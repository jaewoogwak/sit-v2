import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Models.gmmformer.model_timing import GMMFormer_Net


class BinaryProjection(nn.Module):
    """Binary projection layer similar to new_tvr.py Projection class"""
    def __init__(self, in_dim=384, out_dim=3008, act="tanh"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.act = dict(
            tanh=nn.Tanh(), 
            relu=nn.ReLU(),
            gelu=nn.GELU(), 
            sigmoid=nn.Sigmoid()
        )[act]
        # Initialize similar to new_tvr.py
        nn.init.normal_(self.linear.weight, 0, 0.02)

    def forward(self, x):
        return self.act(self.linear(x))


class GMMFormerBinary(GMMFormer_Net):
    """Hybrid model: GMMFormer + Binary Projection for efficient retrieval"""
    
    def __init__(self, config):
        super(GMMFormerBinary, self).__init__(config)
        self.config = config
        
        # Binary projection layers
        self.binary_dim = getattr(config, 'binary_dim', 3008)
        self.binary_act = getattr(config, 'binary_act', 'tanh')
        self.binary_temp = getattr(config, 'binary_temp', 0.07)
        
        # Text and video binary projections
        self.text_binary_proj = BinaryProjection(
            config.hidden_size, self.binary_dim, self.binary_act
        )
        self.video_binary_proj = BinaryProjection(
            config.hidden_size, self.binary_dim, self.binary_act
        )
        
        # Binary packing utilities (from new_tvr.py)
        self.pack_dim = self.binary_dim // 64  # 3008 // 64 = 47
        
    def _pack_binary(self, x):
        """Pack binary representations for efficient hamming distance computation"""
        bits = (x > 0).long().view(x.size(0), self.binary_dim // 64, 64)
        shifts = torch.arange(64, dtype=torch.int64, device=x.device)
        return (bits << shifts).sum(-1).contiguous()
    
    def _pack_min_max(self, x, dim_stats=None):
        """Min-Max distance-based binarization from new_tvr.py"""
        if dim_stats is not None:
            dim_min, dim_max = dim_stats['min'], dim_stats['max']
        else:
            dim_min = x.min(dim=0)[0]
            dim_max = x.max(dim=0)[0]
            
        dist_to_min = torch.abs(x - dim_min)
        dist_to_max = torch.abs(x - dim_max)
        bits = (dist_to_max < dist_to_min).long()
        
        bits = bits.view(bits.size(0), self.binary_dim // 64, 64)
        shifts = torch.arange(64, dtype=torch.int64, device=x.device)
        return (bits << shifts).sum(-1).contiguous()
    
    def forward(self, batch):
        """Forward pass with both GMMFormer and binary outputs"""
        # Original GMMFormer forward
        gmmformer_output = super().forward(batch)
        clip_scale_scores, clip_scale_scores_, label_dict, frame_scale_scores, frame_scale_scores_, video_query = gmmformer_output
        
        # Get video representations for binary projection
        clip_video_feat = batch['clip_video_features']
        frame_video_feat = batch['frame_video_features']
        frame_video_mask = batch['videos_mask']
        last_level_segments = batch.get('last_level_segments')
        
        # Encode context (reuse from parent class)
        encoded_frame_feat, vid_proposal_feat, _ = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask, last_level_segments=last_level_segments
        )
        
        # Apply binary projections
        text_binary = self.text_binary_proj(video_query)  # (N, binary_dim)
        
        # For video: use clip-level representation (vid_proposal_feat)
        # Sum over clip dimension to get video-level representation
        video_repr = vid_proposal_feat.sum(1)  # (N, hidden_size)
        video_binary = self.video_binary_proj(video_repr)  # (N, binary_dim)
        
        # Normalize for contrastive learning
        text_binary_norm = F.normalize(text_binary, dim=-1)
        video_binary_norm = F.normalize(video_binary, dim=-1)
        
        return {
            'gmmformer_output': gmmformer_output,
            'text_binary': text_binary,
            'video_binary': video_binary,
            'text_binary_norm': text_binary_norm,
            'video_binary_norm': video_binary_norm,
            'text_packed': self._pack_binary(text_binary),
            'video_packed': self._pack_binary(video_binary)
        }
    
    def encode_for_retrieval(self, batch, mode='binary'):
        """Encode queries/videos for retrieval evaluation"""
        if 'text_feat' in batch:
            # Text query encoding
            query_feat = batch['text_feat']
            query_mask = batch['text_mask']
            video_query = self.encode_query(query_feat, query_mask)
            
            if mode == 'binary':
                text_binary = self.text_binary_proj(video_query)
                return {
                    'embedding': video_query,
                    'binary': text_binary,
                    'binary_norm': F.normalize(text_binary, dim=-1),
                    'packed': self._pack_binary(text_binary)
                }
            else:
                return {'embedding': video_query}
                
        elif 'clip_video_features' in batch:
            # Video encoding
            clip_video_feat = batch['clip_video_features']
            frame_video_feat = batch['frame_video_features']
            frame_video_mask = batch['videos_mask']
            last_level_segments = batch.get('last_level_segments')
            
            encoded_frame_feat, vid_proposal_feat, _ = self.encode_context(
                clip_video_feat, frame_video_feat, frame_video_mask, last_level_segments=last_level_segments
            )
            
            video_repr = vid_proposal_feat.sum(1)
            
            if mode == 'binary':
                video_binary = self.video_binary_proj(video_repr)
                return {
                    'embedding': video_repr,
                    'binary': video_binary,
                    'binary_norm': F.normalize(video_binary, dim=-1),
                    'packed': self._pack_binary(video_binary)
                }
            else:
                return {'embedding': video_repr}
    
    def compute_contrastive_loss(self, text_binary_norm, video_binary_norm, labels=None):
        """Compute contrastive loss similar to new_tvr.py"""
        logits = video_binary_norm @ text_binary_norm.T
        
        if labels is None:
            labels = torch.arange(len(text_binary_norm), device=text_binary_norm.device)
            
        loss = (F.cross_entropy(logits / self.binary_temp, labels) +
                F.cross_entropy(logits.T / self.binary_temp, labels)) / 2
        
        return loss
