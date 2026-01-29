from Losses.loss import loss
from Losses.clip_only_loss import clip_only_loss
from Losses.clip_only_hd_loss import clip_only_hd_loss

def get_losses(cfg):

   

 # Choose loss based on model type
    model_type = cfg.get('model_type', 'original')
    
    if model_type == 'clip_only_hd':
        print("Using HD-enhanced clip-only loss")
        return clip_only_hd_loss(cfg)
    elif cfg.get('clip_only', False) or model_type == 'clip_only':
        print("Using clip-only loss")
        return clip_only_loss(cfg)        
    else:
        print("Using original loss")
        return loss(cfg)