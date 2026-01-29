from easydict import EasyDict as EDict

from Models.gmmformer.model_timing import GMMFormer_Net
from Models.gmmformer.clip_only_model import GMMFormer_ClipOnly_Net
from Models.gmmformer.clip_only_hd_model import GMMFormer_ClipOnly_HD_Net

def get_models(cfg):
    model_config = EDict(
        visual_input_size=cfg['visual_feat_dim'],
        query_input_size=cfg['q_feat_size'],
        hidden_size=cfg['hidden_size'],  # hidden dimension
        max_ctx_l=cfg['max_ctx_l'],
        max_desc_l=cfg['max_desc_l'],
        map_size=cfg['map_size'],
        input_drop=cfg['input_drop'],
        drop=cfg['drop'],
        n_heads=cfg['n_heads'],  # self-att heads
        initializer_range=cfg['initializer_range'],  # for linear layer
        segment_max_l=cfg.get('segment_max_l', cfg.get('max_ctx_l', 128)),
        max_segments=cfg.get('max_segments', None),
        segment_merge_ratio=cfg.get('segment_merge_ratio', None),
        segment_merge_target=cfg.get('segment_merge_target', None),
        margin=cfg['margin'],  # margin for ranking loss
        use_hard_negative=False,  # reset at each epoch
        hard_pool_size=cfg['hard_pool_size'],
        # Binary projection settings
        clip_only=cfg.get('clip_only', False),
        binary_dim=cfg.get('binary_dim', 3008),
        binary_act=cfg.get('binary_act', 'tanh'),
        # HD projection settings
        hd_dim=cfg.get('hd_dim', 3008),
        hd_activation=cfg.get('hd_activation', 'tanh'),
        pure_block=cfg.get('pure_block', False),
        pure_block_ffn=cfg.get('pure_block_ffn', True),
        context_encoder_type=cfg.get('context_encoder_type', 'gmm'),
        std_transformer_layers=cfg.get('std_transformer_layers', 4),
        std_transformer_heads=cfg.get('std_transformer_heads', 8),
        std_transformer_ffn_dim=cfg.get('std_transformer_ffn_dim', None),
        use_soft_mil=cfg.get('use_soft_mil', False),
    )

    # Choose model based on model type
    model_type = cfg.get('model_type', 'original')
    
    if model_type == 'clip_only_hd':
        model = GMMFormer_ClipOnly_HD_Net(model_config)
        print("Using GMMFormer HD-enhanced clip-only model")
    elif cfg.get('clip_only', False) or model_type == 'clip_only':
        model = GMMFormer_ClipOnly_Net(model_config)
        print("Using GMMFormer clip-only model")
    else:
        model = GMMFormer_Net(model_config)
        print("Using original GMMFormer_Net")
    
    return model
