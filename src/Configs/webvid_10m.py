import os
import yaml


cfg = {}


cfg['model_name'] = 'gmmformer'
cfg['dataset_name'] = 'webvid-10m'
cfg['seed'] = 9527
cfg['root'] = '/home/jaewoo/GMMFormer'
cfg['data_root'] = '/mnt/data/jaewoo/webvid-10M'

cfg['visual_feature'] = 'i3d'
cfg['collection'] = 'webvid_dummy'
cfg['map_size'] = 32
cfg['clip_scale_w'] = 0.7
cfg['frame_scale_w'] = 0.3

cfg['model_root'] = os.path.join(cfg['root'], 'results', cfg['dataset_name'], cfg['model_name'])
cfg['ckpt_path'] = os.path.join(cfg['model_root'], 'ckpt')


# dataset
cfg['num_workers'] = 4
cfg['no_core_driver'] = False
cfg['no_pin_memory'] = False
cfg['batchsize'] = 128


# opt
cfg['lr'] = 0.00025
cfg['lr_warmup_proportion'] = 0.01
cfg['wd'] = 0.01
cfg['margin'] = 0.2


# train
cfg['n_epoch'] = 100
cfg['max_es_cnt'] = 10
cfg['hard_negative_start_epoch'] = 20
cfg['hard_pool_size'] = 20
cfg['use_hard_negative'] = False
cfg['loss_factor'] = [0.02, 0.04, 0.015]
cfg['neg_factor'] = [0.2, 32]


# eval
cfg['eval_query_bsz'] = 50
cfg['eval_context_bsz'] = 100
cfg['measure_search'] = True  # Enable search timing measurement

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

# binary projection settings
cfg['model_type'] = ''
cfg['hd_dim'] = 3008                # HD dimension (must be divisible by 64 for binary packing)
cfg['hd_activation'] = 'tanh'       # Activation function for HD projection
cfg['hd_temperature'] = 0.07        # Temperature for HD contrastive learning


# HD evaluation settings
cfg['use_binary_inference'] = False  # Use float HD for inference (set True for binary)
cfg['dual_hd_eval'] = False          # Evaluate both float and binary HD modes

# Legacy settings (kept for compatibility)
cfg['clip_only'] = False            # Not used when model_type is specified
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
