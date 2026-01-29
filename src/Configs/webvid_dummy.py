import os
import yaml
import torch

cfg = {}


cfg['model_name'] = 'gmmformer'
cfg['dataset_name'] = 'webvid_dummy'
cfg['seed'] = 9527
cfg['root'] = '/dev/ssd1/gjw/GMMFormer'
cfg['data_root'] = '/dev/hdd2/gjw/webvid-10M'

# webvid_dummy는 i3d/roberta 샤드를 사용
cfg['visual_feature'] = 'i3d'
cfg['visual_feat_dim'] = 1024
cfg['collection'] = 'webvid_dummy'
cfg['frames_per_video'] = 12
cfg['map_size'] = 32
cfg['clip_scale_w'] = 0.7
cfg['frame_scale_w'] = 0.3

cfg['model_root'] = os.path.join(
    cfg['root'], 'results', cfg['dataset_name'], cfg['model_name'])
cfg['ckpt_path'] = os.path.join(cfg['model_root'], 'ckpt')


# dataset
cfg['num_workers'] = 4
cfg['no_core_driver'] = False
cfg['no_pin_memory'] = False
cfg['batchsize'] = 128


# opt
cfg['lr'] = 0.0003
cfg['lr_warmup_proportion'] = 0.01
cfg['wd'] = 0.01
cfg['margin'] = 0.1


# train
cfg['n_epoch'] = 550
cfg['max_es_cnt'] = 10
cfg['hard_negative_start_epoch'] = 30
cfg['hard_pool_size'] = 20
cfg['use_hard_negative'] = False
cfg['loss_factor'] = [0.05, 0.04, 0.001]
cfg['neg_factor'] = [0.15, 32]


# eval
cfg['eval_query_bsz'] = 1
cfg['eval_context_bsz'] = 100
cfg['measure_search'] = True
cfg['timing_warmup_batches'] = 3
cfg['pre_normalize_context'] = True
cfg['timing_topk'] = 10

# model
cfg['max_desc_l'] = 30
cfg['max_ctx_l'] = 128
cfg['q_feat_size'] = 1024
cfg['max_position_embeddings'] = 300
cfg['hidden_size'] = 384
cfg['n_heads'] = 4
cfg['input_drop'] = 0.2
cfg['drop'] = 0.2
cfg['initializer_range'] = 0.02

# HD-specific settings
cfg['model_type'] = ''
cfg['hd_dim'] = 3008
cfg['hd_activation'] = 'tanh'
cfg['hd_temperature'] = 0.07

cfg['use_binary_inference'] = False
cfg['dual_hd_eval'] = False

cfg['clip_only'] = False
cfg['binary_dim'] = 3008
cfg['binary_act'] = 'tanh'
cfg['binary_temp'] = 0.07
cfg['hypervector_weight'] = 1.0


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


def print_param_stats(model: torch.nn.Module, label: str = "Model"):
    n = count_trainable_params(model)
    print(f"[{label}] trainable params: {n/1e6:.3f}M  ({n:,})")


def count_trainable_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
