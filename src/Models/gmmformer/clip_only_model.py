import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.gmmformer.model_components import BertAttention, LinearLayer, \
                                            TrainablePositionalEncoding, GMMBlock

import ipdb


class GMMFormer_ClipOnly_Net(nn.Module):
    """
    Simplified GMMFormer that only uses clip-level features for efficient training.
    Removes frame-level processing to reduce computational overhead.
    """
    def __init__(self, config):
        super(GMMFormer_ClipOnly_Net, self).__init__()
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
        Forward pass for clip-only model.
        Returns only clip-level scores and removes frame-level computation.
        """
        clip_video_feat = batch['clip_video_features']
        query_feat = batch['text_feat']
        query_mask = batch['text_mask']
        query_labels = batch['text_labels']
        
        # Only encode clip-level context (no frame processing)
        vid_proposal_feat = self.encode_clip_context(clip_video_feat)
        
        # Get clip-level predictions
        clip_scale_scores, clip_scale_scores_ = self.get_pred_from_raw_query(
            query_feat, query_mask, query_labels, vid_proposal_feat, return_query_feats=True)

        # Create label dictionary for loss computation
        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        # Get query embeddings for diversity loss
        video_query = self.encode_query(query_feat, query_mask)

        # Return only clip-level outputs (remove frame outputs)
        return [clip_scale_scores, clip_scale_scores_, label_dict, video_query]

    def encode_query(self, query_feat, query_mask):
        """Encode text query into embedding"""
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        video_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D)
        return video_query

    def encode_clip_context(self, clip_video_feat):
        """
        Encode only clip-level video features.
        Simplified version that removes frame-level processing.
        """
        encoded_clip_feat = self.encode_input(clip_video_feat, None, self.clip_input_proj, self.clip_encoder,
                                               self.clip_pos_embed)
        encoded_clip_feat = self.clip_encoder_2(encoded_clip_feat, None)  # [bs, 32, 384]
        return encoded_clip_feat

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
    def get_clip_scale_scores(modularied_query, context_feat):
        """
        Compute normalized clip-level similarity scores.
        """
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
        query_context_scores, indices = torch.max(clip_level_query_context_scores, dim=1)
        
        return query_context_scores

    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):
        """
        Compute unnormalized clip-level similarity scores for NCE loss.
        """
        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
        output_query_context_scores, indices = torch.max(query_context_scores, dim=1)
        return output_query_context_scores

    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None, return_query_feats=False):
        """
        Get clip-level predictions from raw query.
        Simplified version that only returns clip-level scores.
        """
        video_query = self.encode_query(query_feat, query_mask)

        # Get clip-level retrieval scores
        clip_scale_scores = self.get_clip_scale_scores(video_query, video_proposal_feat)

        if return_query_feats:
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            return clip_scale_scores, clip_scale_scores_
        else:
            return clip_scale_scores


def mask_logits(target, mask):
    """Apply mask to logits for attention computation"""
    return target * mask + (1 - mask) * (-1e10)
