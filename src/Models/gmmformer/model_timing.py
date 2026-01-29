import copy
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.gmmformer.model_components import BertAttention, LinearLayer, \
                                            TrainablePositionalEncoding, GMMBlock, \
                                            StandardTransformerEncoder, IdentityEncoder, \
                                            tome_merge_tokens

import ipdb
import time

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class GMMFormer_Net(nn.Module):
    def __init__(self, config):
        super(GMMFormer_Net, self).__init__()
        self.config = config
        self._debug_last_level_printed = False

        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        self.segment_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=getattr(config, "segment_max_l", config.max_ctx_l),
            hidden_size=config.hidden_size, dropout=config.input_drop
        )
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)

        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))


        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        use_std_encoder = getattr(config, "context_encoder_type", "gmm").lower() == "transformer"
        if use_std_encoder:
            std_heads = getattr(config, "std_transformer_heads", 8)
            std_layers = getattr(config, "std_transformer_layers", 4)
            std_ffn = getattr(config, "std_transformer_ffn_dim", config.hidden_size * 4)
            std_cfg = edict(hidden_size=config.hidden_size, intermediate_size=std_ffn,
                            hidden_dropout_prob=config.drop, num_attention_heads=std_heads,
                            attention_probs_dropout_prob=config.drop)
            self.clip_encoder = StandardTransformerEncoder(std_cfg, num_layers=std_layers)
            self.clip_encoder_2 = IdentityEncoder()
        else:
            gmm_cfg = edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                            hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                            attention_probs_dropout_prob=config.drop,
                            pure_block=getattr(config, "pure_block", False),
                            pure_block_ffn=getattr(config, "pure_block_ffn", True))
            self.clip_encoder = GMMBlock(gmm_cfg)
            self.clip_encoder_2 = GMMBlock(gmm_cfg)

        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                             dropout=config.input_drop, relu=True)
        if use_std_encoder:
            self.frame_encoder_1 = StandardTransformerEncoder(std_cfg, num_layers=std_layers)
            self.frame_encoder_2 = IdentityEncoder()
        else:
            self.frame_encoder_1 = GMMBlock(gmm_cfg)
            self.frame_encoder_2 = GMMBlock(gmm_cfg)
                    
        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)
        self.modular_vector_mapping_2 = nn.Linear(config.hidden_size, out_features=1, bias=False)


        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
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

        clip_video_feat = batch['clip_video_features']
        query_feat = batch['text_feat']
        query_mask = batch['text_mask']
        query_labels = batch['text_labels']

        frame_video_feat = batch['frame_video_features']
        frame_video_mask = batch['videos_mask']
        last_level_segments = batch.get('last_level_segments')

        encoded_frame_feat, vid_proposal_feat, clip_mask = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask, last_level_segments=last_level_segments)
        
        # print(f"##### [Train] Video proposal feat:{vid_proposal_feat.shape} Video feat:{encoded_frame_feat.shape}")
        clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_ \
            = self.get_pred_from_raw_query(
            query_feat, query_mask, query_labels, vid_proposal_feat, encoded_frame_feat,
            clip_mask=clip_mask, return_query_feats=True)

        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)

        video_query = self.encode_query(query_feat, query_mask)

        output = [clip_scale_scores, clip_scale_scores_, label_dict, frame_scale_scores, frame_scale_scores_, video_query]
        if bool(getattr(self.config, "use_soft_mil", False)):
            output.extend([vid_proposal_feat, clip_mask])
        return output


    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        if query_mask is not None:
            mask = query_mask.unsqueeze(1)
        

        video_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 1

        return video_query


    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None, last_level_segments=None):

        clip_mask = None
        if isinstance(clip_video_feat, (list, tuple)):
            encoded_clip_feat, clip_mask = self.encode_segments(clip_video_feat)
        else:
            encoded_clip_feat = self.encode_input(clip_video_feat, None, self.clip_input_proj, self.clip_encoder,
                                                  self.clip_pos_embed)
            encoded_clip_feat = self.clip_encoder_2(encoded_clip_feat, None)                # [bs, 32, 384]

        use_last_level = bool(getattr(self.config, "use_last_level_as_frame", False))
        boundary_level = str(getattr(self.config, "boundary_level", "")).lower()
        auto_last_level = ("+" in boundary_level and "fine" in boundary_level and "levels" in boundary_level)
        if not use_last_level and auto_last_level:
            try:
                explicit = "use_last_level_as_frame" in self.config
            except Exception:
                explicit = False
            if not explicit:
                use_last_level = True

        if (not self._debug_last_level_printed) and os.environ.get('GMMFORMER_DEBUG_LAST_LEVEL', '').strip() not in ('', '0', 'false', 'False'):
            logging.getLogger().info(
                "DEBUG[last_level] "
                f"use_last_level={use_last_level} "
                f"auto_last_level={auto_last_level} "
                f"last_level_segments={'yes' if last_level_segments is not None else 'no'} "
                f"boundary_level={boundary_level}"
            )
            self._debug_last_level_printed = True

        if use_last_level and last_level_segments is not None:
            encoded_frame_feat = self.encode_last_level_segments(last_level_segments)
        else:
            encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj,
                                                    self.frame_encoder_1,
                                                    self.frame_pos_embed)                   # [bs, N, 384]
            encoded_frame_feat = self.frame_encoder_2(encoded_frame_feat, video_mask.unsqueeze(1))

            encoded_frame_feat = self.get_modularized_frames(encoded_frame_feat, video_mask)


        return encoded_frame_feat, encoded_clip_feat, clip_mask

    def encode_segments(self, segments_list):
        seg_counts = []
        flat_segments = []
        merge_ratio = getattr(self.config, "segment_merge_ratio", None)
        merge_target = getattr(self.config, "segment_merge_target", None)
        for segs in segments_list:
            seg_counts.append(len(segs))
            for seg in segs:
                if seg.dim() == 1:
                    seg = seg.unsqueeze(0)
                max_seg_len = getattr(self.config, "segment_max_l", None)
                if max_seg_len and seg.size(0) > max_seg_len:
                    idx = torch.linspace(
                        0,
                        seg.size(0) - 1,
                        steps=int(max_seg_len),
                        device=seg.device,
                    )
                    seg = seg[idx.long()]
                seg = tome_merge_tokens(seg, ratio=merge_ratio, target_len=merge_target)
                flat_segments.append(seg)

        if not flat_segments:
            device = torch.device("cpu")
            dtype = torch.float32
            hsz = self.config.hidden_size
            padded = torch.zeros((len(segments_list), 1, hsz), device=device, dtype=dtype)
            mask = torch.ones((len(segments_list), 1), device=device, dtype=torch.float32)
            return padded, mask

        device = flat_segments[0].device
        lengths = torch.tensor([seg.size(0) for seg in flat_segments], device=device)
        sorted_idx = torch.argsort(lengths, descending=True)
        chunk_size = getattr(self.config, "segment_batch_size", None)
        if not chunk_size or chunk_size <= 0:
            chunk_size = len(flat_segments)

        encoded_segments = torch.empty(
            (len(flat_segments), self.config.hidden_size),
            device=device,
            dtype=flat_segments[0].dtype,
        )

        for start in range(0, len(flat_segments), chunk_size):
            end = min(start + chunk_size, len(flat_segments))
            chunk_idx = sorted_idx[start:end]
            seg_list = [flat_segments[i] for i in chunk_idx.tolist()]
            chunk_lengths = lengths[chunk_idx]
            max_len = int(chunk_lengths.max().item())
            seg_batch = torch.nn.utils.rnn.pad_sequence(seg_list, batch_first=True)
            mask_batch = (torch.arange(max_len, device=device).unsqueeze(0) < chunk_lengths.unsqueeze(1)).to(torch.float32)
            seg_batch = self.clip_input_proj(seg_batch)
            seg_batch = self.segment_pos_embed(seg_batch)
            attn_mask = mask_batch.unsqueeze(1)
            seg_batch = self.clip_encoder(seg_batch, attn_mask)
            seg_batch = self.clip_encoder_2(seg_batch, attn_mask)
            denom = mask_batch.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            seg_batch = (seg_batch * mask_batch.unsqueeze(-1)).sum(dim=1) / denom
            encoded_segments[chunk_idx] = seg_batch

        batch_embs = []
        idx = 0
        for count in seg_counts:
            if count == 0:
                batch_embs.append(torch.zeros((1, self.config.hidden_size),
                                              device=device, dtype=encoded_segments.dtype))
            else:
                batch_embs.append(encoded_segments[idx:idx + count])
                idx += count

        max_segments = getattr(self.config, "max_segments", None)
        max_in_batch = max(x.size(0) for x in batch_embs) if batch_embs else 1
        if max_segments is None or max_segments <= 0:
            max_segments = max_in_batch
        elif max_segments < max_in_batch:
            max_segments = max_in_batch

        hsz = batch_embs[0].size(1)
        device = batch_embs[0].device
        dtype = batch_embs[0].dtype
        padded = torch.zeros((len(batch_embs), max_segments, hsz), device=device, dtype=dtype)
        mask = torch.zeros((len(batch_embs), max_segments), device=device, dtype=torch.float32)
        for i, segs in enumerate(batch_embs):
            s = segs.size(0)
            padded[i, :s, :] = segs
            mask[i, :s] = 1.0
        return padded, mask

    def encode_last_level_segments(self, segments_list):
        encoded_segments, seg_mask = self.encode_segments_with_encoder(
            segments_list,
            self.frame_input_proj,
            self.segment_pos_embed,
            self.frame_encoder_1,
            self.frame_encoder_2,
        )
        denom = seg_mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        return (encoded_segments * seg_mask.unsqueeze(-1)).sum(dim=1) / denom

    def encode_segments_with_encoder(self, segments_list, input_proj_layer, pos_embed_layer,
                                     encoder_layer, encoder_layer_2):
        seg_counts = []
        flat_segments = []
        merge_ratio = getattr(self.config, "segment_merge_ratio", None)
        merge_target = getattr(self.config, "segment_merge_target", None)
        for segs in segments_list:
            seg_counts.append(len(segs))
            for seg in segs:
                if seg.dim() == 1:
                    seg = seg.unsqueeze(0)
                max_seg_len = getattr(self.config, "segment_max_l", None)
                if max_seg_len and seg.size(0) > max_seg_len:
                    idx = torch.linspace(
                        0,
                        seg.size(0) - 1,
                        steps=int(max_seg_len),
                        device=seg.device,
                    )
                    seg = seg[idx.long()]
                seg = tome_merge_tokens(seg, ratio=merge_ratio, target_len=merge_target)
                flat_segments.append(seg)

        if not flat_segments:
            device = torch.device("cpu")
            dtype = torch.float32
            hsz = self.config.hidden_size
            padded = torch.zeros((len(segments_list), 1, hsz), device=device, dtype=dtype)
            mask = torch.ones((len(segments_list), 1), device=device, dtype=torch.float32)
            return padded, mask

        device = flat_segments[0].device
        lengths = torch.tensor([seg.size(0) for seg in flat_segments], device=device)
        sorted_idx = torch.argsort(lengths, descending=True)
        chunk_size = getattr(self.config, "segment_batch_size", None)
        if not chunk_size or chunk_size <= 0:
            chunk_size = len(flat_segments)

        encoded_segments = torch.empty(
            (len(flat_segments), self.config.hidden_size),
            device=device,
            dtype=flat_segments[0].dtype,
        )

        for start in range(0, len(flat_segments), chunk_size):
            end = min(start + chunk_size, len(flat_segments))
            chunk_idx = sorted_idx[start:end]
            seg_list = [flat_segments[i] for i in chunk_idx.tolist()]
            chunk_lengths = lengths[chunk_idx]
            max_len = int(chunk_lengths.max().item())
            seg_batch = torch.nn.utils.rnn.pad_sequence(seg_list, batch_first=True)
            mask_batch = (torch.arange(max_len, device=device).unsqueeze(0) < chunk_lengths.unsqueeze(1)).to(torch.float32)
            seg_batch = input_proj_layer(seg_batch)
            seg_batch = pos_embed_layer(seg_batch)
            attn_mask = mask_batch.unsqueeze(1)
            seg_batch = encoder_layer(seg_batch, attn_mask)
            seg_batch = encoder_layer_2(seg_batch, attn_mask)
            denom = mask_batch.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            seg_batch = (seg_batch * mask_batch.unsqueeze(-1)).sum(dim=1) / denom
            encoded_segments[chunk_idx] = seg_batch

        batch_embs = []
        idx = 0
        for count in seg_counts:
            if count == 0:
                batch_embs.append(torch.zeros((1, self.config.hidden_size),
                                              device=device, dtype=encoded_segments.dtype))
            else:
                batch_embs.append(encoded_segments[idx:idx + count])
                idx += count

        max_segments = getattr(self.config, "max_segments", None)
        max_in_batch = max(x.size(0) for x in batch_embs) if batch_embs else 1
        if max_segments is None or max_segments <= 0:
            max_segments = max_in_batch
        elif max_segments < max_in_batch:
            max_segments = max_in_batch

        hsz = batch_embs[0].size(1)
        device = batch_embs[0].device
        dtype = batch_embs[0].dtype
        padded = torch.zeros((len(batch_embs), max_segments, hsz), device=device, dtype=dtype)
        mask = torch.zeros((len(batch_embs), max_segments), device=device, dtype=torch.float32)
        for i, segs in enumerate(batch_embs):
            s = segs.size(0)
            padded[i, :s, :] = segs
            mask[i, :s] = 1.0
        return padded, mask

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
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
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()

    def get_modularized_frames(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping_2(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()


    def get_clip_scale_scores(self, modularied_query, context_feat, clip_mask=None, return_timing=False):

        modularied_query = F.normalize(modularied_query, dim=-1)
        # If context was pre-normalized upstream, skip re-normalization to avoid duplication
        if not getattr(self.config, 'context_already_normalized', False):
            context_feat = F.normalize(context_feat, dim=-1)
        
        if return_timing:
            pass
            # [act] context feat shape: (4430, 32, 384) modularied_query shape: (50, 384)
            # [tvr] context feat shape: ()
        
        matmul_result = torch.matmul(context_feat, modularied_query.t())
        # print(f"clip scale matmul: query {tuple(modularied_query.shape)} video {tuple(context_feat.shape)} -> {(t_mm1 - t_mm0):.6f}s",
        #       flush=True)
        # 일관된 출력 모양을 보장: (B, K, V)
        if matmul_result.dim() == 2:  # (V, K) when batch=1
            # (V, K) -> (K, V) -> (1, K, V)
            clip_level_query_context_scores = matmul_result.permute(1, 0).unsqueeze(0)
        else:  # (V, K, B) -> (B, K, V)
            clip_level_query_context_scores = matmul_result.permute(2, 1, 0)

        if clip_mask is not None:
            if clip_mask.dim() == 1:
                clip_mask = clip_mask.unsqueeze(0)
            mask = clip_mask.transpose(0, 1).unsqueeze(0)
            clip_level_query_context_scores = clip_level_query_context_scores.masked_fill(mask == 0, -1e10)

        query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                  dim=1)
        
        return query_context_scores


    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat, clip_mask=None):

        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
        if clip_mask is not None:
            if clip_mask.dim() == 1:
                clip_mask = clip_mask.unsqueeze(0)
            mask = clip_mask.transpose(0, 1).unsqueeze(0)
            query_context_scores = query_context_scores.masked_fill(mask == 0, -1e10)

        output_query_context_scores, indices = torch.max(query_context_scores, dim=1)

        return output_query_context_scores


    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None, encoded_frame_feat=None,
                                clip_mask=None,
                                return_query_feats=False, return_timing=False):
        
        
        if return_timing:
            _sync(); t_enc0 = time.perf_counter()
        
        # Encdoing & Normalize
        video_query = self.encode_query(query_feat, query_mask) # 포함
        normalized_video_query = F.normalize(video_query, dim=-1) # 얘네위로올리기
        # If context was pre-normalized upstream, reuse as-is
        if getattr(self.config, 'context_already_normalized', False):
            normalized_frame_feat = encoded_frame_feat
        else:
            normalized_frame_feat = F.normalize(encoded_frame_feat, dim=-1)
        
        if return_timing:
            _sync(); t_enc1 = time.perf_counter()
        
        
        # get clip-level retrieval scores
        if return_timing:
            _sync(); t_all0 = time.perf_counter()  # start 2: similarity check starts here
            _sync(); t_c0 = time.perf_counter()
        clip_scale_scores = self.get_clip_scale_scores(
            video_query, video_proposal_feat, clip_mask=clip_mask, return_timing=return_timing)

        if return_timing:
            _sync(); t_c1 = time.perf_counter()
            _sync(); t_f0 = time.perf_counter()
            # breakpoint()
        # Handle batch size 1 case for frame_scale_scores
        
        if normalized_video_query.dim() == 1:  # batch size 1
            normalized_video_query = normalized_video_query.unsqueeze(0)
        

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fmm0 = time.perf_counter()
        matmul_result = torch.matmul(normalized_frame_feat, normalized_video_query.t())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fmm1 = time.perf_counter()
        # print(f"frame scale matmul: query {tuple(normalized_video_query.shape)} video {tuple(normalized_frame_feat.shape)} -> {(t_fmm1 - t_fmm0):.6f}s",
        #       flush=True)
        if matmul_result.dim() == 1:
            # (V,) when batch size is 1 -> (1, V)
            frame_scale_scores = matmul_result.unsqueeze(0)
        else:
            # (V, B) -> (B, V)
            frame_scale_scores = matmul_result.permute(1, 0)
        if return_timing:
            _sync(); t_f1 = time.perf_counter()
            _sync(); t_all1 = time.perf_counter()
            timing = {
                "encoding_normalize_ms": (t_enc1 - t_enc0) * 1000.0,
                "search_total_ms": (t_all1 - t_all0) * 1000.0,
                "clip_ms": (t_c1 - t_c0) * 1000.0,
                "frame_ms": (t_f1 - t_f0) * 1000.0,
                "similarity_start_time": t_all0,  # Pass start time for cross-function timing
            }
        
        if return_query_feats:
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat, clip_mask=clip_mask)
            frame_scale_scores_ = torch.matmul(encoded_frame_feat, video_query.t()).permute(1, 0)
            if return_timing:
                return clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_, timing
            else:
                return clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_
        else:
            if return_timing:
                return clip_scale_scores, frame_scale_scores, timing
            else:
                return clip_scale_scores, frame_scale_scores

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
