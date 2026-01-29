import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

import ipdb

import numpy as np
import logging
import torch.backends.cudnn as cudnn
import os
import pickle

from tqdm import tqdm
import torch

from Utils.utils import gpu
import json

# Add path for bp_hamming import
import sys
import pathlib as _pl
clip_test_root = _pl.Path(__file__).resolve().parents[3] / "clip-test"  # /disk/gjw/clip-test/
sys.path.append(str(clip_test_root))


def _pack_binary(x, binary_dim=3008):
    """Pack binary representations for efficient hamming distance computation"""
    bits = (x > 0).long().view(x.size(0), binary_dim // 64, 64)
    shifts = torch.arange(64, dtype=torch.int64, device=x.device)
    return (bits << shifts).sum(-1).contiguous()


def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, v2t_gt in enumerate(v2t_gt):
        for t_gt in v2t_gt:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt


def eval_q2m(scores, q2m_gts):
    n_q, n_m = scores.shape

    gt_ranks = torch.zeros((n_q), dtype=torch.int32).cuda()
    aps = torch.zeros(n_q).cuda()
    for i in range(n_q):
        s = scores[i]
        sorted_idxs = torch.argsort(s)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = torch.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(torch.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(torch.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(torch.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(torch.where(gt_ranks <= 100)[0]) / n_q

    return (r1, r5, r10, r100)


def cal_perf(t2v_all_errors, t2v_gt):
    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100) = eval_q2m(t2v_all_errors, t2v_gt)

    return (t2v_r1, t2v_r5, t2v_r10, t2v_r100)


class binary_validations(nn.Module):
    """Binary validation that maintains original interface but adds binary evaluation"""
    
    def __init__(self, cfg):
        super(binary_validations, self).__init__()
        self.cfg = cfg
        self.use_binary = getattr(cfg, 'use_binary', False)
        self.binary_dim = getattr(cfg, 'binary_dim', 3008)

    def forward(self, model, context_dataloader, query_eval_loader):
        """Main validation function - maintains original interface"""
        model.eval()

        # Run original validation
        context_info = self.compute_context_info(model, context_dataloader)
        score_sum, query_metas = self.compute_query2ctx_info(model,
                                                             query_eval_loader,
                                                             context_info)
        video_metas = context_info['video_metas']

        v2t_gt, t2v_gt = get_gt(video_metas, query_metas)

        t2v_r1, t2v_r5, t2v_r10, t2v_r100 = cal_perf(-1 * score_sum, t2v_gt)
        t2v_rsum = 0
        t2v_rsum += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)

        # If binary is enabled, also run binary evaluation
        print(f"self.use_binary {self.use_binary}")
        
        print("\n--- Running Binary Evaluation ---")
        binary_results = self.binary_evaluation(model, context_dataloader, query_eval_loader)
        
        print(f"Original  - R@1: {t2v_r1:.1f}, R@5: {t2v_r5:.1f}, R@10: {t2v_r10:.1f}, R@100: {t2v_r100:.1f}, Sum: {t2v_rsum:.1f}")
        print(f"Binary    - R@1: {binary_results[0]:.1f}, R@5: {binary_results[1]:.1f}, R@10: {binary_results[2]:.1f}, R@100: {binary_results[3]:.1f}, Sum: {binary_results[4]:.1f}")

        return [t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_rsum]

    def compute_query2ctx_info(self, model, query_eval_loader, ctx_info):
        """Original query to context computation"""
        query_metas = []
        score_sum = []
        for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):

            batch = gpu(batch)
            query_metas.extend(batch[-1])
            query_feat = batch[0]
            query_mask = batch[1]
            _clip_scale_scores, _frame_scale_scores = model.get_pred_from_raw_query(
                query_feat, query_mask, None, ctx_info["video_proposal_feat"], ctx_info["video_feat"],
                clip_mask=ctx_info.get("clip_mask"))
            _score_sum = self.cfg['clip_scale_w'] * _clip_scale_scores + self.cfg['frame_scale_w'] * _frame_scale_scores

            score_sum.append(_score_sum)

        score_sum = torch.cat(score_sum, dim=0)

        return score_sum, query_metas

    def compute_context_info(self, model, context_dataloader):
        """Original context computation"""
        n_total_vid = len(context_dataloader.dataset)
        bsz = self.cfg['eval_context_bsz']
        metas = []  # list(dicts)
        vid_proposal_feat = []
        clip_mask = []
        frame_feat, frame_mask = [], []
        for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
                            total=len(context_dataloader)):

            batch = gpu(batch)
            metas.extend(batch[-1])
            clip_video_feat_ = batch[0]
            frame_video_feat_ = batch[1]
            frame_mask_ = batch[2]
            last_level_segments = batch[3] if len(batch) >= 6 else None
            _frame_feat, _video_proposal_feat, _clip_mask = model.encode_context(
                clip_video_feat_, frame_video_feat_, frame_mask_, last_level_segments=last_level_segments
            )

            frame_feat.append(_frame_feat)
            frame_mask.append(frame_mask_)

            vid_proposal_feat.append(_video_proposal_feat)
            if _clip_mask is None:
                clip_mask.append(torch.ones(_video_proposal_feat.size(0), _video_proposal_feat.size(1),
                                           device=_video_proposal_feat.device))
            else:
                clip_mask.append(_clip_mask)

        vid_proposal_feat = torch.cat(vid_proposal_feat, dim=0)
        clip_mask = torch.cat(clip_mask, dim=0) if clip_mask else None

        def cat_tensor(tensor_list):
            if len(tensor_list) == 0:
                return None
            else:
                seq_l = [e.shape[1] for e in tensor_list]
                b_sizes = [e.shape[0] for e in tensor_list]
                b_sizes_cumsum = np.cumsum([0] + b_sizes)
                if len(tensor_list[0].shape) == 3:
                    hsz = tensor_list[0].shape[2]
                    res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
                elif len(tensor_list[0].shape) == 2:
                    res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
                else:
                    raise ValueError("Only support 2/3 dimensional tensors")
                for i, e in enumerate(tensor_list):
                    res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i+1], :seq_l[i]] = e
                return res_tensor
                
        return dict(
            video_metas=metas,  # list(dict) (N_videos)
            video_proposal_feat=vid_proposal_feat,
            video_feat=cat_tensor(frame_feat),
            video_mask=cat_tensor(frame_mask),
            clip_mask=clip_mask
            )

    def binary_evaluation(self, model, context_dataloader, query_eval_loader):
        """Hypervector-based binary evaluation using hamming distance or cosine similarity"""
        
        # Compute hypervector video embeddings
        print("Computing hypervector video embeddings...")
        clip_hypervectors = []
        frame_hypervectors = []
        video_metas = []
        
        for batch in tqdm(context_dataloader, desc="Hypervector video encoding"):
            batch = gpu(batch)
            video_metas.extend(batch[-1])
            
            clip_video_feat = batch[0]
            frame_video_feat = batch[1]
            frame_mask = batch[2]
            
            # Get hypervector representations (with binarization for hamming distance)
            binary_out = model.get_binary_video_embedding(
                clip_video_feat, frame_video_feat, frame_mask, use_binarization=True
            )
            
            clip_hypervectors.append(binary_out['clip_hypervector'])
            frame_hypervectors.append(binary_out['frame_hypervector'])
        
        clip_hypervectors = torch.cat(clip_hypervectors, dim=0)  # (N_videos, 3008)
        frame_hypervectors = torch.cat(frame_hypervectors, dim=0)  # (N_videos, 3008)
        
        # Compute hypervector query embeddings
        print("Computing hypervector query embeddings...")
        text_hypervectors = []
        query_metas = []
        
        for batch in tqdm(query_eval_loader, desc="Hypervector query encoding"):
            batch = gpu(batch)
            query_metas.extend(batch[-1])
            
            query_feat = batch[0]
            query_mask = batch[1]
            
            binary_out = model.get_binary_query_embedding(
                query_feat, query_mask, use_binarization=True
            )
            text_hypervectors.append(binary_out['text_hypervector'])
        
        text_hypervectors = torch.cat(text_hypervectors, dim=0)  # (N_queries, 3008)
        
        # Evaluate both clip-level and frame-level hypervectors
        print("Evaluating hypervector similarities...")
        
        # Clip-level evaluation
        clip_results = self._evaluate_hypervector_similarity(
            text_hypervectors, clip_hypervectors, video_metas, query_metas, "clip"
        )
        
        # Frame-level evaluation  
        frame_results = self._evaluate_hypervector_similarity(
            text_hypervectors, frame_hypervectors, video_metas, query_metas, "frame"
        )
        
        # Combined evaluation (weighted sum like original GMMFormer)
        clip_weight = self.cfg.get('clip_scale_w', 0.7)
        frame_weight = self.cfg.get('frame_scale_w', 0.3)
        
        combined_results = []
        for i in range(len(clip_results)):
            combined_results.append(clip_weight * clip_results[i] + frame_weight * frame_results[i])
        
        print(f"Clip-level   - R@1: {clip_results[0]:.1f}, R@5: {clip_results[1]:.1f}, R@10: {clip_results[2]:.1f}, R@100: {clip_results[3]:.1f}, Sum: {clip_results[4]:.1f}")
        print(f"Frame-level  - R@1: {frame_results[0]:.1f}, R@5: {frame_results[1]:.1f}, R@10: {frame_results[2]:.1f}, R@100: {frame_results[3]:.1f}, Sum: {frame_results[4]:.1f}")
        print(f"Combined     - R@1: {combined_results[0]:.1f}, R@5: {combined_results[1]:.1f}, R@10: {combined_results[2]:.1f}, R@100: {combined_results[3]:.1f}, Sum: {combined_results[4]:.1f}")
        
        return combined_results

    def _evaluate_hypervector_similarity(self, text_hypervectors, video_hypervectors, video_metas, query_metas, level_name=""):
        """Evaluate hypervector similarity using both hamming distance and cosine similarity"""
        
        # Try binary hamming distance first
        try:
            from Validations.binary_index import bp_hamming
            
            # Pack hypervectors for hamming distance  
            text_packed = _pack_binary(text_hypervectors, self.binary_dim)
            video_packed = _pack_binary(video_hypervectors, self.binary_dim)
            
            # Create hamming index
            index = bp_hamming(self.binary_dim)
            index.add(video_packed)
            
            # Compute hamming distances
            binary_scores = []
            for i in range(text_packed.size(0)):
                distances, indices = index.search(text_packed[i:i+1], video_packed.size(0))
                # Convert hamming distances to similarities (lower distance = higher similarity)
                similarities = -distances[0].float()
                binary_scores.append(similarities)
            
            binary_scores = torch.stack(binary_scores)  # (N_queries, N_videos)
            print(f"Using binary hamming distance for {level_name} level")
            
        except ImportError:
            # Fallback to cosine similarity on hypervectors
            print(f"Binary index not available, using cosine similarity for {level_name} level")
            text_hv_norm = F.normalize(text_hypervectors, dim=-1)
            video_hv_norm = F.normalize(video_hypervectors, dim=-1)
            binary_scores = text_hv_norm @ video_hv_norm.T  # (N_queries, N_videos)
        
        # Evaluate using original metrics
        v2t_gt, t2v_gt = get_gt(video_metas, query_metas)
        
        t2v_r1, t2v_r5, t2v_r10, t2v_r100 = cal_perf(-1 * binary_scores, t2v_gt)
        t2v_rsum = t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100
        
        return [t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_rsum]

    def detailed_binary_analysis(self, model, context_dataloader, query_eval_loader):
        """Detailed analysis comparing original vs binary performance"""
        print("\n--- Detailed Binary Analysis ---")
        
        # This can be called separately for more detailed analysis
        context_info = self.compute_context_info(model, context_dataloader)
        
        # Get original scores
        original_scores, query_metas = self.compute_query2ctx_info(
            model, query_eval_loader, context_info
        )
        
        # Get binary scores  
        binary_results = self.binary_evaluation(model, context_dataloader, query_eval_loader)
        
        video_metas = context_info['video_metas']
        v2t_gt, t2v_gt = get_gt(video_metas, query_metas)
        
        # Compare performance per query type, etc.
        # This can be extended for more detailed analysis
        
        return {
            'original': original_scores,
            'binary': binary_results,
            'video_metas': video_metas,
            'query_metas': query_metas
        }
