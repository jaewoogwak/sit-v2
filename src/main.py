import os
import csv
import argparse
import numpy as np
import random
import sys
import time
from tqdm import tqdm
# import ipdb
import pickle

import torch
import torch.nn as nn

from Configs.builder import get_configs
from Models.builder import get_models
from Datasets.builder import get_datasets
from Opts.builder import get_opts
from Losses.builder import get_losses
from Validations.builder import get_validations

from Utils.basic_utils import AverageMeter, BigFile, read_dict, log_config
from Utils.utils import set_seed, set_log, gpu, save_ckpt, load_ckpt
from itertools import islice



root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser(description="Partially Relevant Video Retrieval")
parser.add_argument(
    '-d', '--dataset_name', default='tvr', type=str, metavar='DATASET', help='dataset name', 
    choices=['tvr', 'act', 'cha', 'tvr_clip', 'tvr_frames', 'act_frames', 'tvr_internvideo', 'act_clip', 'tvr_hd', 'msrvtt', 'webvid', 'webvid_dummy', 'webvid_dummy_18', 'webvid-10m', 'webvid_10m', 'webvid10m']
)
parser.add_argument(
    '--gpu', default = '0', type = str, help = 'specify gpu device'
    )
parser.add_argument('--eval', action='store_true')
parser.add_argument('--search', action='store_true', help='In eval mode, use train+test videos as search context')
parser.add_argument('--search_limit', type=int, default=0, help='Cap number of train videos included in search context (0 = no cap)')
parser.add_argument('--eval_query_limit', type=int, default=0, help='Limit number of query captions to load for evaluation (0 = all)')
parser.add_argument('--timing_topk', type=int, default=0, help='If >0, measure per-query timing using top-k instead of full argsort (timing only)')
parser.add_argument('--timing_on_gpu', action='store_true', help='Measure per-query sorting (top-k/argsort) on GPU instead of CPU (timing only)')
parser.add_argument('--eval_debug_vid', default='', type=str, help='If set, print per-query debug info for this video id during eval')
parser.add_argument('--eval_debug_topk', type=int, default=10, help='Top-K videos to print for eval debug')
parser.add_argument('--eval_debug_segment_topk', type=int, default=5, help='Top-K segments to print per query for eval debug')
parser.add_argument('--eval_debug_frames_root', type=str, default='', help='Frames root for eval debug segment visualization')
parser.add_argument('--eval_debug_segment_frames', type=int, default=8, help='Frames per segment to show in eval debug visualization')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--steps_per_epoch', type=int, default=-1,
                    help='limit number of training steps per epoch (cap); set -1 to disable')
parser.add_argument('--max_epoch', type=int, default=-1,
                    help='hard cap on number of epochs to run (does not change cfg n_epoch)')
parser.add_argument('--amp', action='store_true', help='enable mixed precision (CUDA AMP)')
parser.add_argument('--accum_steps', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--grad_clip', type=float, default=1.0, help='max grad norm (0 to disable)')
parser.add_argument('--train_shard_size', type=int, default=0,
                    help='if >0, train on a moving shard of this many samples per epoch (rotates across dataset)')
parser.add_argument('--debug_shapes', action='store_true',
                    help='print one sample feature shapes (raw HDF5 and pre-model batch); forces num_workers=0')
parser.add_argument('--deterministic', action='store_true',
                    help='enable deterministic (reproducible) CUDA/CPU behavior; slower')
parser.add_argument('--backbone', default='base', type=str, choices=['base', 'large'],
                    help='CLIP backbone size for frame features (e.g., tvr_frames): base=ViT-B/32, large=ViT-L/14')

args = parser.parse_args()

def log_trainable_layers(model, logger):
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    total = sum(param.numel() for param in model.parameters() if param.requires_grad)
    header = f"Trainable layers: {len(names)} params={total:,}"
    if names:
        logger.info(header + "\n" + "\n".join(names))
    else:
        logger.info(header + " (none)")


def _format_loss_terms(criterion):
    terms = getattr(criterion, "_last_loss_terms", None)
    if not isinstance(terms, dict):
        return ""
    return (
        f" cN:{terms.get('clip_nce', 0.0):.3f}"
        f" cT:{terms.get('clip_trip', 0.0):.3f}"
        f" fN:{terms.get('frame_nce', 0.0):.3f}"
        f" fT:{terms.get('frame_trip', 0.0):.3f}"
        f" q:{terms.get('qdl', 0.0):.3f}"
        f" h:{terms.get('hier', 0.0):.3f}"
        f" s:{terms.get('selector', 0.0):.3f}"
    )


def train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer):

    criterion.cfg['_current_epoch'] = int(epoch)

    if epoch >= cfg['hard_negative_start_epoch']:
        criterion.cfg['use_hard_negative'] = True
    else:
        criterion.cfg['use_hard_negative'] = False

    loss_meter = AverageMeter()

    model.train()
    
    # 에폭당 스텝 제한 적용
    max_steps = getattr(cfg, 'steps_per_epoch', None)
    if max_steps is None:
        max_steps = getattr(args, 'steps_per_epoch', -1)
    use_cap = (max_steps is not None and max_steps > 0)
    
     # AMP / Grad Accum 준비
    use_amp = getattr(args, 'amp', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum_steps = max(1, getattr(args, 'accum_steps', 1))
    grad_clip = getattr(args, 'grad_clip', 1.0)

    total = max_steps if use_cap else len(train_loader)
    train_bar = tqdm(train_loader, desc=f"epoch {epoch}", total=total,
                     unit="batch", dynamic_ncols=True, mininterval=1.0)


    optimizer.zero_grad(set_to_none=True)
    for idx, batch in enumerate(train_bar):
        if use_cap and idx >= max_steps:
            break

        batch = gpu(batch)

        with torch.cuda.amp.autocast(enabled=use_amp):
            input_list = model(batch)
            loss = criterion(input_list, batch) / accum_steps

        scaler.scale(loss).backward()

        if (idx + 1) % accum_steps == 0:
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_meter.update(loss.detach().cpu().item() * accum_steps)
        term_str = _format_loss_terms(criterion)
        train_bar.set_description(
            f'exp: {cfg["model_name"]} epoch:{epoch:2d} iter:{idx:3d} loss:{loss_meter.avg:.4f}{term_str}'
        )
        if cfg.get('debug_hier_loss', False):
            every = int(cfg.get('debug_hier_loss_every', 50) or 50)
            if every <= 0:
                every = 1
            if (idx % every) == 0:
                stats = getattr(criterion, "_last_hier_stats", None)
                if isinstance(stats, dict):
                    train_bar.write(f"[hier_loss] {stats}")

    # 누적 후 남은 그라드 처리
    if ((idx + 1) % accum_steps) != 0:
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return loss_meter.avg

def val_one_epoch(epoch, context_dataloader, query_eval_loader, model, val_criterion, cfg, optimizer, best_val, loss_meter, logger):

    val_meter = val_criterion(model, context_dataloader, query_eval_loader)

    if val_meter[4] > best_val[4]:
        es = False
        sc = 'New Best Model !!!'
        best_val = val_meter
        save_ckpt(model, optimizer, cfg, os.path.join(cfg['model_root'], 'best.ckpt'), epoch, best_val)
    else:
        es = True
        sc = 'A Relative Failure Epoch'
                
    logger.info('==========================================================================================================')
    logger.info('Epoch: {:2d}    {}'.format(epoch, sc))
    logger.info('Average Loss: {:.4f}'.format(loss_meter))
    logger.info('R@1: {:.1f}'.format(val_meter[0]))
    logger.info('R@5: {:.1f}'.format(val_meter[1]))
    logger.info('R@10: {:.1f}'.format(val_meter[2]))
    logger.info('R@100: {:.1f}'.format(val_meter[3]))
    logger.info('Rsum: {:.1f}'.format(val_meter[4]))
    logger.info('Best: R@1: {:.1f} R@5: {:.1f} R@10: {:.1f} R@100: {:.1f} Rsum: {:.1f}'.format(best_val[0], best_val[1], best_val[2], best_val[3], best_val[4]))
    logger.info('==========================================================================================================')

    sweep_csv = os.getenv('GMMFORMER_SWEEP_CSV', '').strip()
    if sweep_csv:
        os.makedirs(os.path.dirname(sweep_csv), exist_ok=True)
        write_header = not os.path.exists(sweep_csv)
        with open(sweep_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "exp_name",
                    "epoch",
                    "loss",
                    "r1",
                    "r5",
                    "r10",
                    "r100",
                    "rsum",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "exp_name": cfg.get("model_name", ""),
                    "epoch": epoch,
                    "loss": float(loss_meter),
                    "r1": float(val_meter[0]),
                    "r5": float(val_meter[1]),
                    "r10": float(val_meter[2]),
                    "r100": float(val_meter[3]),
                    "rsum": float(val_meter[4]),
                }
            )
        
    return val_meter, best_val, es


def validation(context_dataloader, query_eval_loader, model, val_criterion, cfg, logger, resume):

    val_meter = val_criterion(model, context_dataloader, query_eval_loader)
    
    logger.info('==========================================================================================================')
    logger.info('Testing from: {}'.format(resume))
    logger.info('R@1: {:.1f}'.format(val_meter[0]))
    logger.info('R@5: {:.1f}'.format(val_meter[1]))
    logger.info('R@10: {:.1f}'.format(val_meter[2]))
    logger.info('R@100: {:.1f}'.format(val_meter[3]))
    logger.info('Rsum: {:.1f}'.format(val_meter[4]))
    logger.info('==========================================================================================================')


def main():
    cfg = get_configs(args.dataset_name)

    # set logging
    logger = set_log(cfg['model_root'], 'log.txt')
    logger.info('Partially Relevant Video Retrieval Training: {}'.format(cfg['dataset_name']))

    # optional backbone override for frame-feature datasets
    backbone = str(getattr(args, 'backbone', 'base') or 'base').lower()
    if backbone == 'large' and cfg.get('visual_feature') in ['tvr_frames', 'act_frames']:
        if cfg.get('visual_feature') == 'tvr_frames':
            cfg['frame_feature_dir'] = '/dev/ssd1/gjw/vcmr/TOT-CVPR22/tvr_dataset/features_CLIP-L14'
        else:
            cfg['frame_feature_dir'] = '/dev/hdd2/gjw/datasets/activitynet/features_CLIP-L14'
        logger.info(f"backbone override: {backbone} -> frame_feature_dir={cfg['frame_feature_dir']}")

    # debug: print shapes once (dataset + gpu() hook)
    if bool(getattr(args, 'debug_shapes', False)):
        os.environ['GMMFORMER_DEBUG_SHAPES'] = '1'
        try:
            cfg['debug_shapes'] = True
        except Exception:
            pass
        logger.info('DEBUG: enabled shape printing (raw HDF5 + pre-model batch); forcing num_workers=0 in dataset builder')

    # hyper parameter
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_ids = range(torch.cuda.device_count())
    logger.info('used gpu: {}'.format(args.gpu))

    # set seed (optionally deterministic)
    set_seed(cfg['seed'], cuda_deterministic=bool(getattr(args, 'deterministic', False)))
    if getattr(args, 'deterministic', False):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    logger.info('set seed: {}'.format(cfg['seed']))

    logger.info('Hyper Parameter ......')
    logger.info(cfg)

    # dataset
    logger.info('Loading Data ......')
    # propagate runtime flags to cfg so dataset builder can optimize
    try:
        cfg['eval'] = bool(args.eval)
        cfg['search'] = bool(args.search)
        cfg['search_limit'] = int(args.search_limit or 0)
        cfg['eval_query_limit'] = int(args.eval_query_limit or 0)
        cfg['timing_topk'] = int(args.timing_topk or 0)
        cfg['timing_on_gpu'] = bool(args.timing_on_gpu)
        cfg['eval_debug_vid'] = str(args.eval_debug_vid or '').strip()
        cfg['eval_debug_frames_root'] = str(getattr(args, 'eval_debug_frames_root', '') or '').strip()
        cfg['eval_debug_segment_frames'] = int(getattr(args, 'eval_debug_segment_frames', 0) or 0)
        cfg['eval_debug_topk'] = int(args.eval_debug_topk or 0)
        cfg['eval_debug_segment_topk'] = int(args.eval_debug_segment_topk or 0)
        cfg['debug_shapes'] = bool(getattr(args, 'debug_shapes', False))
    except Exception:
        pass
    # propagate shard_size into cfg for dataset builder
    if args.train_shard_size and args.train_shard_size > 0:
        cfg['train_shard_size'] = int(args.train_shard_size)
    cfg, train_loader, context_dataloader, query_eval_loader, test_context_dataloader, test_query_eval_loader = get_datasets(cfg)

    # model
    logger.info('Loading Model ......') 
    model = get_models(cfg)
    log_trainable_layers(model, logger)


    cfg['steps_per_epoch'] = args.steps_per_epoch
   
    # initial
    current_epoch = -1
    es_cnt = 0
    best_val = [0., 0., 0., 0., 0.]
    if args.resume != '':
        logger.info('Resume from {}'.format(args.resume))
        _, model_state_dict, optimizer_state_dict, current_epoch, best_val = load_ckpt(args.resume)
        model.load_state_dict(model_state_dict)
    model = model.cuda()
    if len(device_ids) > 1:
        model = nn.DataParallel(model)
    
    criterion = get_losses(cfg)
    val_criterion = get_validations(cfg)

    if args.eval:
        # Evaluate even without a checkpoint (use randomly initialized weights)
        if args.resume == '':
            logger.info('Eval with untrained model (random init)')
        else:
            logger.info('Eval with checkpoint: {}'.format(args.resume))

        # Build optional train+test search context
        eval_context_loader = test_context_dataloader
        if args.search:
            try:
                from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader, Subset
                from Datasets.data_provider import read_video_ids, read_video_ids_limited, collate_frame_val
                from Datasets.msrvtt_dataset import read_video_ids_msrvtt, read_video_ids_msrvtt_limited

                # Reuse readers from test context dataset
                test_vid_ds = test_context_dataloader.dataset
                visual_reader = getattr(test_vid_ds, 'visual_feat', None)
                video2frames = getattr(test_vid_ds, 'video2frames', None)
                ds_cls = test_vid_ds.__class__

                # Determine timing/search configuration
                measure_search = bool(cfg.get("measure_search", False))
                search_limit = int(args.search_limit or 0)

                # Optionally cap the amount of test context when measuring search timing
                test_dataset = test_vid_ds
                if measure_search and search_limit > 0 and search_limit < len(test_vid_ds):
                    test_dataset = Subset(test_vid_ds, range(search_limit))

                test_used = len(test_dataset)
                # Decide how many train videos are required
                train_cap = None
                if measure_search and search_limit > 0:
                    if search_limit <= test_used:
                        train_cap = 0
                    else:
                        train_cap = search_limit - test_used
                elif search_limit > 0:
                    train_cap = search_limit

                # Derive train caption path
                rootpath = cfg['data_root']
                collection = cfg['collection']
                if collection == 'msrvtt':
                    trainCollection = 'msrvtttrain'
                elif collection == 'webvid':
                    trainCollection = 'webvidtrain'
                else:
                    trainCollection = f'{collection}train'
                train_cap_path = os.path.join(rootpath, collection, 'TextData', f'{trainCollection}.caption.txt')

                # Build train video dataset (avoid scanning massive caption files for webvid)
                train_vid_ds = None
                if train_cap is None or train_cap > 0:
                    if collection == 'webvid':
                        # Determine total videos from the video reader
                        total_videos = None
                        try:
                            if hasattr(visual_reader, 'shape'):
                                total_videos = int(visual_reader.shape()[0])
                            elif hasattr(visual_reader, 'nr_of_images'):
                                total_videos = int(visual_reader.nr_of_images)
                        except Exception:
                            total_videos = None
                        if not total_videos or total_videos <= 0:
                            raise RuntimeError('Could not determine total WebVid train videos without reading captions')

                        class _LazyVidIds:
                            def __init__(self, n):
                                self.n = int(n)
                            def __len__(self):
                                return self.n
                            def __getitem__(self, idx):
                                return f"vid_{int(idx)}"

                        use_n = total_videos
                        if train_cap is not None:
                            use_n = min(use_n, max(int(train_cap), 0))
                        if use_n > 0:
                            train_vid_ds = ds_cls(visual_reader, video2frames, cfg, video_ids=_LazyVidIds(use_n))
                    else:
                        # Prefer caption file if exists; otherwise fallback to video2frames keys
                        train_video_ids_list = None
                        if os.path.exists(train_cap_path):
                            # Read capped number of unique train video IDs
                            limit = 0 if train_cap is None else max(int(train_cap), 0)
                            if collection == 'msrvtt':
                                train_video_ids_list = read_video_ids_msrvtt_limited(train_cap_path, limit)
                            else:
                                train_video_ids_list = read_video_ids_limited(train_cap_path, limit)
                        else:
                            # Fallback: derive from video2frames if available
                            if video2frames is not None:
                                all_keys = list(video2frames.keys())
                                # Avoid duplicating videos already present in test dataset if possible
                                try:
                                    test_ids = list(getattr(test_vid_ds, 'video_ids', []))
                                    test_set = set(test_ids)
                                    all_keys = [k for k in all_keys if k not in test_set]
                                except Exception:
                                    pass
                                limit = 0 if train_cap is None else max(int(train_cap), 0)
                                if limit and limit > 0:
                                    train_video_ids_list = all_keys[:limit]
                                else:
                                    train_video_ids_list = all_keys
                            else:
                                raise RuntimeError(f"No caption file {train_cap_path} and no video2frames available to build train search context")

                        # Instantiate a train video dataset of the same class as test
                        if train_video_ids_list:
                            train_vid_ds = ds_cls(visual_reader, video2frames, cfg, video_ids=train_video_ids_list)

                # Combine train + test (skip train if not required)
                datasets = []
                if train_vid_ds is not None and len(train_vid_ds) > 0:
                    datasets.append(train_vid_ds)
                if test_dataset is not None and len(test_dataset) > 0:
                    datasets.append(test_dataset)
                if not datasets:
                    raise RuntimeError('No videos available for search context')
                combined_ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
                eval_context_loader = TorchDataLoader(
                    combined_ds,
                    collate_fn=collate_frame_val,
                    batch_size=cfg['eval_context_bsz'],
                    num_workers=cfg['num_workers'],
                    shuffle=False,
                    pin_memory=cfg['pin_memory']
                )
                # Logging: indicate source and limit
                src_note = 'caption_file' if os.path.exists(train_cap_path) else 'video2frames_fallback'
                lim_note = int(args.search_limit or 0)
                train_count = len(train_vid_ds) if train_vid_ds is not None else 0
                test_count = len(test_dataset)
                logger.info(f'Using search context (train+test): train={train_count} test={test_count} total={train_count + test_count} (src={src_note}, search_limit={lim_note})')
            except Exception as e:
                logger.info(f'Could not build search context (train+test): {e}. Falling back to test-only context.')

        with torch.no_grad():
            resume_label = args.resume if args.resume != '' else 'untrained (random init)'
            validation(eval_context_loader, test_query_eval_loader, model, val_criterion, cfg, logger, resume_label)
        exit(0)

    optimizer = get_opts(cfg, model, train_loader)
    if args.resume != '':
        optimizer.load_state_dict(optimizer_state_dict)

    max_epoch = getattr(args, 'max_epoch', -1)
    end_epoch = cfg['n_epoch']
    if max_epoch is not None and max_epoch > 0:
        end_epoch = min(end_epoch, int(max_epoch))
        logger.info('Max epoch cap: {}'.format(end_epoch))

    for epoch in range(current_epoch + 1, end_epoch):

        # If using a shard-aware sampler, advance shard with epoch
        try:
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
        except Exception:
            pass

        ############## train
        loss_meter = train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer)

        ############## val
        with torch.no_grad():
            val_meter, best_val, es = val_one_epoch(epoch, context_dataloader, query_eval_loader, model, 
                    val_criterion, cfg, optimizer, best_val, loss_meter, logger)

        ############## early stop
        if not es:
            es_cnt = 0
        else:
            es_cnt += 1
            if cfg['max_es_cnt'] != -1 and es_cnt > cfg['max_es_cnt']:  # early stop
                logger.info('Early Stop !!!') 
                exit(0)


if __name__ == '__main__':
    main()
