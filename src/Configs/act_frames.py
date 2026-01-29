import os
import yaml


cfg = {}

# identity + paths
cfg['model_name'] = 'SiT-act-frames-gmm-softmil-c7-f3-level-complete'
cfg['dataset_name'] = 'act_frames'
cfg['seed'] = 9527
cfg['root'] = '/dev/ssd1/gjw/prvr/semantic-transformer-v2'
cfg['data_root'] = '/dev/ssd1/gjw/prvr/dataset'

# features + scoring
cfg['visual_feature'] = 'act_frames'
cfg['text_feature'] = 'clip'
cfg['collection'] = 'activitynet'
cfg['map_size'] = 32
cfg['clip_scale_w'] = 0.7
cfg['frame_scale_w'] = 0.3

# boundaries + segmenting
cfg['frame_feature_dir'] = '/dev/hdd2/gjw/datasets/activitynet/features'
cfg['boundary_train_path'] = '/dev/ssd1/gjw/prvr/semantic-transformer-v2/boundary_detection/output/boundaries_act_train.json'
cfg['boundary_val_path'] = '/dev/ssd1/gjw/prvr/semantic-transformer-v2/boundary_detection/output/boundaries_act_val.json'
cfg['boundary_level'] = 'fine+levels'
cfg['use_last_level_as_frame'] = True
cfg['dedupe_segments'] = False
cfg['video2frames_path'] = '/dev/ssd1/gjw/prvr/dataset/activitynet/FeatureData/i3d/video2frames.txt'

cfg['model_root'] = os.path.join(cfg['root'], 'results', cfg['dataset_name'], cfg['model_name'])
cfg['ckpt_path'] = os.path.join(cfg['model_root'], 'ckpt')


# dataset + loader
cfg['num_workers'] = 32
cfg['no_core_driver'] = False
cfg['no_pin_memory'] = False
cfg['batchsize'] = 128
cfg['persistent_workers'] = True
cfg['prefetch_factor'] = 2


# opt
cfg['lr'] = 0.00025
cfg['lr_warmup_proportion'] = 0.01
cfg['wd'] = 0.01
cfg['margin'] = 0.1


# train
cfg['n_epoch'] = 100
cfg['max_es_cnt'] = 10
cfg['hard_negative_start_epoch'] = 20
cfg['hard_pool_size'] = 20
cfg['use_hard_negative'] = True
cfg['loss_factor'] = [0.02, 0.04, 0.015, 0.035]
cfg['neg_factor'] = [0.2, 32]
cfg['hier_parent_pow'] = 2
cfg['debug_hier_loss'] = True
cfg['debug_hier_loss_every'] = 20


# eval
cfg['eval_query_bsz'] = 50
cfg['eval_context_bsz'] = 100


# model
cfg['max_desc_l'] = 64
cfg['max_ctx_l'] = 256
cfg['segment_max_l'] = 32
cfg['q_feat_size'] = 512
cfg['visual_feat_dim'] = 512
cfg['max_position_embeddings'] = 300
cfg['hidden_size'] = 512
cfg['n_heads'] = 4
cfg['input_drop'] = 0.2
cfg['drop'] = 0.2
cfg['initializer_range'] = 0.02
cfg['segment_batch_size'] = 32
cfg['segment_merge_ratio'] = 0.85
cfg['segment_merge_target'] = 32
cfg['context_encoder_type'] = 'gmm'
cfg['std_transformer_layers'] = 4
cfg['std_transformer_heads'] = 8
cfg['std_transformer_ffn_dim'] = 2048

# soft MIL (requires release paths)
cfg['use_soft_mil'] = True
cfg['release_train_path'] = '/dev/hdd2/gjw/datasets/activitynet/activitynet_train.jsonl'
cfg['release_val_path'] = '/dev/hdd2/gjw/datasets/activitynet/activitynet_val.jsonl'
cfg['soft_mil_sanity_max'] = 0


def _env_list(name, cast=float):
    raw = os.getenv(name, '').strip()
    if not raw:
        return None
    if (raw.startswith('[') and raw.endswith(']')) or (raw.startswith('(') and raw.endswith(')')):
        raw = raw[1:-1].strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    if not parts:
        return None
    return [cast(p) for p in parts]


def _env_scalar(name, cast):
    raw = os.getenv(name, '').strip()
    if not raw:
        return None
    return cast(raw)


def _env_bool(name):
    raw = os.getenv(name, '').strip().lower()
    if not raw:
        return None
    return raw in ('1', 'true', 'yes', 'y', 'on')


def _apply_env_overrides(cfg):
    model_name = os.getenv('GMMFORMER_MODEL_NAME', '').strip()
    if model_name:
        cfg['model_name'] = model_name

    lr = _env_scalar('GMMFORMER_LR', float)
    if lr is not None:
        cfg['lr'] = lr

    wd = _env_scalar('GMMFORMER_WD', float)
    if wd is not None:
        cfg['wd'] = wd

    lr_warmup = _env_scalar('GMMFORMER_LR_WARMUP', float)
    if lr_warmup is not None:
        cfg['lr_warmup_proportion'] = lr_warmup

    n_epoch = _env_scalar('GMMFORMER_N_EPOCH', float)
    if n_epoch is not None:
        cfg['n_epoch'] = int(n_epoch)

    max_es_cnt = _env_scalar('GMMFORMER_MAX_ES_CNT', float)
    if max_es_cnt is not None:
        cfg['max_es_cnt'] = int(max_es_cnt)

    batchsize = _env_scalar('GMMFORMER_BATCHSIZE', float)
    if batchsize is not None:
        cfg['batchsize'] = int(batchsize)

    num_workers = _env_scalar('GMMFORMER_NUM_WORKERS', float)
    if num_workers is not None:
        cfg['num_workers'] = int(num_workers)

    prefetch_factor = _env_scalar('GMMFORMER_PREFETCH_FACTOR', float)
    if prefetch_factor is not None:
        cfg['prefetch_factor'] = int(prefetch_factor)

    persistent_workers = _env_bool('GMMFORMER_PERSISTENT_WORKERS')
    if persistent_workers is not None:
        cfg['persistent_workers'] = persistent_workers

    use_hard_negative = _env_bool('GMMFORMER_USE_HARD_NEG')
    if use_hard_negative is not None:
        cfg['use_hard_negative'] = use_hard_negative

    hard_negative_start_epoch = _env_scalar('GMMFORMER_HARD_NEG_START', float)
    if hard_negative_start_epoch is not None:
        cfg['hard_negative_start_epoch'] = int(hard_negative_start_epoch)

    hard_pool_size = _env_scalar('GMMFORMER_HARD_POOL_SIZE', float)
    if hard_pool_size is not None:
        cfg['hard_pool_size'] = int(hard_pool_size)

    loss_factor = _env_list('GMMFORMER_LOSS_FACTOR', cast=float)
    if loss_factor:
        cfg['loss_factor'] = loss_factor

    neg_factor = _env_list('GMMFORMER_NEG_FACTOR', cast=float)
    if neg_factor:
        cfg['neg_factor'] = neg_factor

    margin = _env_scalar('GMMFORMER_MARGIN', float)
    if margin is not None:
        cfg['margin'] = margin

    clip_scale_w = _env_scalar('GMMFORMER_CLIP_SCALE_W', float)
    if clip_scale_w is not None:
        cfg['clip_scale_w'] = clip_scale_w

    frame_scale_w = _env_scalar('GMMFORMER_FRAME_SCALE_W', float)
    if frame_scale_w is not None:
        cfg['frame_scale_w'] = frame_scale_w

    input_drop = _env_scalar('GMMFORMER_INPUT_DROP', float)
    if input_drop is not None:
        cfg['input_drop'] = input_drop

    drop = _env_scalar('GMMFORMER_DROP', float)
    if drop is not None:
        cfg['drop'] = drop

    segment_merge_ratio = _env_scalar('GMMFORMER_SEGMENT_MERGE_RATIO', float)
    if segment_merge_ratio is not None:
        cfg['segment_merge_ratio'] = segment_merge_ratio

    segment_merge_target = _env_scalar('GMMFORMER_SEGMENT_MERGE_TARGET', float)
    if segment_merge_target is not None:
        cfg['segment_merge_target'] = int(segment_merge_target)

    model_root = os.getenv('GMMFORMER_MODEL_ROOT', '').strip()
    if model_root:
        cfg['model_root'] = model_root
    else:
        cfg['model_root'] = os.path.join(cfg['root'], 'results', cfg['dataset_name'], cfg['model_name'])
    cfg['ckpt_path'] = os.path.join(cfg['model_root'], 'ckpt')


_apply_env_overrides(cfg)


cfg['num_workers'] = 1 if cfg['no_core_driver'] else cfg['num_workers']
cfg['pin_memory'] = not cfg['no_pin_memory']


if not os.path.exists(cfg['model_root']):
    os.makedirs(cfg['model_root'], exist_ok=True)
if not os.path.exists(cfg['ckpt_path']):
    os.makedirs(cfg['ckpt_path'], exist_ok=True)


def get_cfg_defaults():
    with open(os.path.join(cfg['model_root'], 'hyperparams.yaml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file)
    return cfg
