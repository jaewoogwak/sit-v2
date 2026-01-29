import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.gmmformer.model_components import BertAttention, LinearLayer, \
                                            TrainablePositionalEncoding, GMMBlock

import ipdb


class HDProjection(nn.Module):
    """
    Hyperdimensional projection layer that maps input features to high-dimensional space.
    Based on the reference implementation for HD computing.
    """
    def __init__(self, in_dim=384, out_dim=3008, act="tanh"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.act = dict(tanh=nn.Tanh(), relu=nn.ReLU(),
                        gelu=nn.GELU(), sigmoid=nn.Sigmoid())[act]
        # Initialize with small random weights
        nn.init.normal_(self.linear.weight, 0, 0.02)

    def forward(self, x):
        """
        Args:
            x: Input tensor (..., in_dim)
        Returns:
            HD projected tensor (..., out_dim)
        """
        return self.act(self.linear(x))


class GMMFormer_ClipOnly_HD_Net(nn.Module):
    """
    HD-enhanced GMMFormer that uses hyperdimensional computing for clip-level video retrieval.
    
    Key differences from clip-only model:
    1. Projects both query and clip features to HD space (3008-dim)
    2. Aggregates 32 clips into single video representation via summation
    3. Computes similarity in HD space instead of max-pooling over clips
    """
    def __init__(self, config):
        super(GMMFormer_ClipOnly_HD_Net, self).__init__()
        self.config = config

        # Query encoder components
        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))

        # Clip-level video encoder components
        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.clip_encoder = GMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop,
                                                 pure_block=getattr(config, "pure_block", False),
                                                 pure_block_ffn=getattr(config, "pure_block_ffn", True)))
        self.clip_encoder_2 = GMMBlock(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop,
                                                 pure_block=getattr(config, "pure_block", False),
                                                 pure_block_ffn=getattr(config, "pure_block_ffn", True)))
                    
        # Modular attention mapping for query aggregation
        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)

        # HD projection layers
        self.hd_dim = getattr(config, 'hd_dim', 3008)
        self.hd_activation = getattr(config, 'hd_activation', 'tanh')
        
        self.query_hd_proj = HDProjection(in_dim=config.hidden_size, 
                                         out_dim=self.hd_dim, 
                                         act=self.hd_activation)
        self.clip_hd_proj = HDProjection(in_dim=config.hidden_size, 
                                        out_dim=self.hd_dim, 
                                        act=self.hd_activation)

        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def forward(self, batch):
        """
        Forward pass for HD-enhanced clip-only model.
        Returns HD-projected features and similarity scores.
        """
        clip_video_feat = batch['clip_video_features']  # (batch, 32, 384)
        query_feat = batch['text_feat']
        query_mask = batch['text_mask']
        query_labels = batch['text_labels']

        # Encode clip-level context and project to HD space
        video_hd_repr = self.encode_clip_context_hd(clip_video_feat)  # (batch, hd_dim)
        
        # Get HD-based predictions
        clip_scale_scores, clip_scale_scores_ = self.get_pred_from_raw_query_hd(
            query_feat, query_mask, query_labels, video_hd_repr, return_query_feats=True)

        # Create label dictionary for loss computation
        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        # Get HD query embeddings for diversity loss
        video_query_hd = self.encode_query_hd(query_feat, query_mask)

        # Return HD-based outputs
        return [clip_scale_scores, clip_scale_scores_, label_dict, video_query_hd]

    def encode_query_hd(self, query_feat, query_mask):
        """Encode text query into HD embedding"""
        # First encode with GMMFormer layers
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        
        # Apply modular attention to get single query representation
        video_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D)
        
        # Project to HD space
        video_query_hd = self.query_hd_proj(video_query)  # (N, hd_dim)
        
        return video_query_hd

    def encode_clip_context_hd(self, clip_video_feat):
        """
        Encode clip-level video features and aggregate into HD video representation.
        
        Args:
            clip_video_feat: (batch, 32, visual_input_size)
        Returns:
            video_hd_repr: (batch, hd_dim) - aggregated HD video representation
        """
        # Encode clips with GMMFormer layers
        encoded_clip_feat = self.encode_input(clip_video_feat, None, self.clip_input_proj, self.clip_encoder,
                                               self.clip_pos_embed)  # (batch, 32, hidden_size)
        encoded_clip_feat = self.clip_encoder_2(encoded_clip_feat, None)  # (batch, 32, hidden_size)

        # Project each clip to HD space
        clip_hd_feat = self.clip_hd_proj(encoded_clip_feat)  # (batch, 32, hd_dim)
        
        # Aggregate clips into single video representation via summation (HD superposition)
        video_hd_repr = torch.sum(clip_hd_feat, dim=1)  # (batch, hd_dim)
        
        return video_hd_repr

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Generic input encoding function.
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask):
        """
        Apply modular attention to aggregate query features.
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 1, D)
        return modular_queries.squeeze()

    @staticmethod
    def get_hd_similarity_scores(query_hd, video_hd, normalize=True):
        """
        Compute similarity scores in HD space.
        
        Args:
            query_hd: (N_queries, hd_dim)
            video_hd: (N_videos, hd_dim)  
            normalize: Whether to normalize before computing similarity
        Returns:
            similarity_scores: (N_queries, N_videos)
        """
        if normalize:
            query_hd = F.normalize(query_hd, dim=-1)
            video_hd = F.normalize(video_hd, dim=-1)
        
        # Compute cosine similarity in HD space
        similarity_scores = torch.matmul(query_hd, video_hd.t())  # (N_queries, N_videos)
        
        return similarity_scores

    def get_pred_from_raw_query_hd(self, query_feat, query_mask, query_labels=None,
                                   video_hd_repr=None, return_query_feats=False):
        """
        Get HD-based predictions from raw query.
        
        Args:
            query_feat: Raw query features
            query_mask: Query mask
            query_labels: Query labels (optional)
            video_hd_repr: HD video representations (batch, hd_dim)
            return_query_feats: Whether to return unnormalized scores
        Returns:
            HD similarity scores
        """
        # Encode query to HD space
        query_hd = self.encode_query_hd(query_feat, query_mask)  # (N_queries, hd_dim)

        # Compute HD similarity scores (normalized)
        hd_similarity_scores = self.get_hd_similarity_scores(query_hd, video_hd_repr, normalize=True)

        if return_query_feats:
            # Also return unnormalized scores for NCE loss
            hd_similarity_scores_ = self.get_hd_similarity_scores(query_hd, video_hd_repr, normalize=False)
            return hd_similarity_scores, hd_similarity_scores_
        else:
            return hd_similarity_scores


def mask_logits(target, mask):
    """Apply mask to logits for attention computation"""
    return target * mask + (1 - mask) * (-1e10)
