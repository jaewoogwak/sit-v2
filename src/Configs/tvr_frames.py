import os
import yaml


cfg = {}

# identity + paths
cfg['model_name'] = 'SiT-tvr-frames-gmm-softmil-c7-f3-level-complete'
cfg['dataset_name'] = 'tvr_frames'
cfg['seed'] = 9527
cfg['root'] = '/dev/ssd1/gjw/prvr/semantic-transformer-v2'
cfg['data_root'] = '/dev/ssd1/gjw/prvr/dataset'

# features + scoring
cfg['visual_feature'] = 'tvr_frames'
cfg['text_feature'] = 'clip'
cfg['text_mask_path'] = ''
cfg['collection'] = 'tvr'
cfg['map_size'] = 32
cfg['clip_scale_w'] = 0.7
cfg['frame_scale_w'] = 0.3

# boundaries + segmenting
cfg['frame_feature_dir'] = '/dev/ssd1/gjw/vcmr/TOT-CVPR22/tvr_dataset/features'
cfg['boundary_train_path'] = '/dev/ssd1/gjw/prvr/semantic-transformer-v2/boundary_detection/output/boundaries_tvr_train.json'
cfg['boundary_val_path'] = '/dev/ssd1/gjw/prvr/semantic-transformer-v2/boundary_detection/output/boundaries_tvr_val.json'
cfg['boundary_level'] = 'fine+levels'
# When True, frame branch uses last-level segments instead of raw frame sequence.
cfg['use_last_level_as_frame'] = True
# When True, dedupe identical (start, end) segments across fine/levels.
cfg['dedupe_segments'] = False

cfg['model_root'] = os.path.join(cfg['root'], 'results', cfg['dataset_name'], cfg['model_name'])
cfg['ckpt_path'] = os.path.join(cfg['model_root'], 'ckpt')


# dataset + loader
cfg['num_workers'] = 16
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
cfg['loss_factor'] = [0.02, 0.04, 0.015]
cfg['neg_factor'] = [0.2, 32]
cfg['hier_parent_pow'] = 0
cfg['debug_hier_loss'] = False
cfg['debug_hier_loss_every'] = 20


# eval
cfg['eval_query_bsz'] = 50
cfg['eval_context_bsz'] = 100
cfg['eval_debug_slot_sim'] = False
cfg['eval_debug_slot_topk'] = 5


# model
cfg['max_desc_l'] = 30
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
cfg['use_seg_token_selector'] = False
cfg['seg_token_K'] = 8
cfg['seg_token_proj'] = True
cfg['seg_token_bertattn_layers'] = 1
cfg['seg_slot_temp'] = 0.07
cfg['seg_slot_dropout'] = 0.1
cfg['seg_diversity_weight'] = 0.2
cfg['seg_diversity_type'] = 'cosine'
cfg['seg_diversity_margin'] = 0.2
cfg['seg_ts_overlap_thr'] = 0.5
cfg['seg_infonce_temp'] = 0.07
cfg['seg_infonce_weight'] = 1.0
cfg['seg_infer_hard_topk'] = True
cfg['seg_infer_topk'] = cfg['seg_token_K']
cfg['seg_debug_mask_print'] = False
cfg['seg_debug_mask_every'] = 20
cfg['seg_debug_mask_max_print'] = 30

# soft MIL (requires release paths)
cfg['use_soft_mil'] = True
cfg['tvr_release_train_path'] = '/dev/ssd1/gjw/prvr/dataset/tvr/TextData/tvr_train_release.jsonl'
cfg['tvr_release_val_path'] = '/dev/ssd1/gjw/prvr/dataset/tvr/TextData/tvr_val_release.jsonl'
cfg['release_train_path'] = '/dev/ssd1/gjw/prvr/dataset/tvr/TextData/tvr_train_release.jsonl'
cfg['release_val_path'] = '/dev/ssd1/gjw/prvr/dataset/tvr/TextData/tvr_val_release.jsonl'

cfg['soft_mil_sanity_max'] = 0



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
