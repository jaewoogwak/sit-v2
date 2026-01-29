from Validations.validations_timinig import validations
from Validations.clip_only_validation import clip_only_validations
from Validations.clip_only_hd_validation import clip_only_hd_validations, dual_mode_hd_validations
# from Validations.unified_hd_validation import UnifiedHDValidation

def get_validations(cfg):
    
    # Choose validation based on model type
    model_type = cfg.get('model_type', 'original')
    
    if model_type == 'clip_only_hd':
        # Choose validation mode for HD model
        if cfg.get('use_unified_hd_validation', True):  # Default to unified validation
            print("Using Unified HD validation (train_unified.py style)")
            
        elif cfg.get('dual_hd_eval', False):
            print("Using dual-mode HD validations (float + binary)")
            return dual_mode_hd_validations(cfg)
        else:
            print("Using HD-enhanced clip-only validations")
            return clip_only_hd_validations(cfg)
    elif cfg.get('clip_only', False) or model_type == 'clip_only':
        print("Using clip-only validations")
        return clip_only_validations(cfg)
    else:
        print("Using original validations")
        return validations(cfg)