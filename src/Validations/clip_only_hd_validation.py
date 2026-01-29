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


def get_gt(video_metas, query_metas):
    """
    Get ground truth mappings between videos and queries.
    """
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt


def eval_q2m(scores, q2m_gts):
    """
    Evaluate query-to-media retrieval performance.
    """
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
    """
    Calculate retrieval performance metrics.
    """
    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100) = eval_q2m(t2v_all_errors, t2v_gt)

    return (t2v_r1, t2v_r5, t2v_r10, t2v_r100)


def compute_hamming_distance(query_binary, video_binary):
    """
    Compute hamming distance between binary HD embeddings.
    
    Args:
        query_binary: (N_queries, hd_dim) binary embeddings
        video_binary: (N_videos, hd_dim) binary embeddings
    Returns:
        hamming_distances: (N_queries, N_videos) hamming distances
    """
    # Expand dimensions for broadcasting
    query_expanded = query_binary.unsqueeze(1)  # (N_queries, 1, hd_dim)
    video_expanded = video_binary.unsqueeze(0)  # (1, N_videos, hd_dim)
    
    # XOR to find differing bits, then sum
    xor_result = torch.logical_xor(query_expanded > 0, video_expanded > 0)
    hamming_distances = xor_result.float().sum(dim=-1)  # (N_queries, N_videos)
    
    return hamming_distances


def _pack_hd_embeddings(x):
    """
    Binarize and pack HD embeddings using reference method.
    Identical to new_tvr.py _pack function.
    
    Args:
        x: (N, hd_dim) float embeddings
    Returns:
        packed_embeddings: (N, hd_dim//64) packed binary embeddings
    """
    hd_dim = x.shape[-1]
    return ((x > 0).long().view(x.size(0), hd_dim // 64, 64)
            << torch.arange(64, dtype=torch.int64, device=x.device)).sum(-1).contiguous()




class clip_only_hd_validations(nn.Module):
    """
    Validation for HD-enhanced clip-only model.
    Supports both float HD similarity and binary hamming distance evaluation.
    """
    def __init__(self, cfg):
        super(clip_only_hd_validations, self).__init__()
        self.cfg = cfg
        self.use_binary = cfg.get('use_binary_inference', False)

    def forward(self, model, context_dataloader, query_eval_loader):
        """
        Main validation function for HD clip-only model.
        """
        model.eval()

        context_info = self.compute_context_info_hd(model, context_dataloader)
        
        if self.use_binary:
            # Binary HD evaluation - returns recall metrics directly
            recall_results, query_metas = self.compute_query2ctx_info_binary(model,
                                                                           query_eval_loader,
                                                                           context_info)
            return recall_results  # [R@1, R@5, R@10, R@100, RSum] already computed
        else:
            # Float HD evaluation - use traditional similarity matrix approach
            score_sum, query_metas = self.compute_query2ctx_info_hd(model,
                                                                   query_eval_loader,
                                                                   context_info)
        
            video_metas = context_info['video_metas']
            v2t_gt, t2v_gt = get_gt(video_metas, query_metas)
            t2v_r1, t2v_r5, t2v_r10, t2v_r100 = cal_perf(-1 * score_sum, t2v_gt)
            t2v_rsum = t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100

            return [t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_rsum]

    def compute_query2ctx_info_hd(self, model, query_eval_loader, ctx_info):
        """
        Compute query-to-context similarity scores using HD embeddings (float mode).
        """
        query_metas = []
        query_hd_embeddings = []
        
        for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing HD query embeddings", total=len(query_eval_loader)):
            batch = gpu(batch)
            query_metas.extend(batch[-1])
            query_feat = batch[0]
            query_mask = batch[1]
            
            # Get HD query embeddings
            query_hd = model.encode_query_hd(query_feat, query_mask)  # (batch, hd_dim)
            query_hd_embeddings.append(query_hd)

        query_hd_embeddings = torch.cat(query_hd_embeddings, dim=0)  # (N_queries, hd_dim)
        
        # Compute HD similarity scores
        video_hd_embeddings = ctx_info["video_hd_embeddings"]  # (N_videos, hd_dim)
        
        # Normalize embeddings for cosine similarity
        query_hd_norm = F.normalize(query_hd_embeddings, dim=-1)
        video_hd_norm = F.normalize(video_hd_embeddings, dim=-1)
        
        # Compute similarity matrix
        similarity_scores = torch.matmul(query_hd_norm, video_hd_norm.t())  # (N_queries, N_videos)

        # Debug: Check score distribution
        print(f"Float HD similarity scores - Min: {similarity_scores.min().item():.4f}, "
              f"Max: {similarity_scores.max().item():.4f}, "
              f"Mean: {similarity_scores.mean().item():.4f}, "
              f"Std: {similarity_scores.std().item():.4f}")

        return similarity_scores, query_metas

    def compute_query2ctx_info_binary(self, model, query_eval_loader, ctx_info):
        """
        Compute query-to-context similarity using binary HD embeddings with bp_hamming.
        Uses reference-style direct recall calculation without reordering.
        """
        try:
            from Validations.binary_index import bp_hamming
        except ImportError:
            # Fallback to direct import if in same directory
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from binary_index import bp_hamming
        
        query_metas = []
        query_hd_embeddings = []
        
        for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing binary HD query embeddings", total=len(query_eval_loader)):
            batch = gpu(batch)
            query_metas.extend(batch[-1])
            query_feat = batch[0]
            query_mask = batch[1]
            
            # Get HD query embeddings and pack
            query_hd = model.encode_query_hd(query_feat, query_mask)  # (batch, hd_dim)
            query_packed = _pack_hd_embeddings(query_hd)  # (batch, hd_dim//64)
            query_hd_embeddings.append(query_packed)

        query_packed_embeddings = torch.cat(query_hd_embeddings, dim=0)  # (N_queries, hd_dim//64)
        
        # Get packed video embeddings and video metadata
        video_packed_embeddings = ctx_info["video_packed_embeddings"]  # (N_videos, hd_dim//64)
        video_metas = ctx_info['video_metas']
        
        # Create video_id to index mapping for GT lookup
        vid_to_index = {meta_dict['name'] if isinstance(meta_dict, dict) else meta_dict: i 
                       for i, meta_dict in enumerate(video_metas)}
        
        # Use bp_hamming for fast retrieval  
        hd_dim = query_packed_embeddings.shape[-1] * 64  # Recover original HD dimension
        index = bp_hamming(hd_dim)
        index.add(video_packed_embeddings.cuda())
        
        # Direct recall calculation (reference style)
        Ks = [1, 5, 10, 100]
        hits = {k: 0 for k in Ks}
        n_queries = len(query_metas)
        
        for i in range(n_queries):
            query_single = query_packed_embeddings[i:i+1].cuda()
            
            # Get Top-K results directly from bp_hamming
            _, topk_indices = index.search(query_single, Ks[-1])  # Get Top-100
            topk_video_indices = topk_indices[0].cpu().tolist()
            
            # Find GT video index for this query
            query_meta = query_metas[i]
            if isinstance(query_meta, dict):
                # Extract video ID from query meta (format may vary)
                if 'vid_name' in query_meta:
                    gt_video_id = query_meta['vid_name'] 
                elif 'video_id' in query_meta:
                    gt_video_id = query_meta['video_id']
                else:
                    # Fallback: assume query_meta contains video identifier
                    gt_video_id = str(query_meta).split('#')[0] if '#' in str(query_meta) else str(query_meta)
            else:
                # Handle string format: "video_id#query_info"
                gt_video_id = str(query_meta).split('#')[0] if '#' in str(query_meta) else str(query_meta)
            
            # Check if GT video exists in video index
            if gt_video_id in vid_to_index:
                gt_video_idx = vid_to_index[gt_video_id]
                
                # Check recall at different K values
                for k in Ks:
                    if gt_video_idx in topk_video_indices[:k]:
                        hits[k] += 1
            else:
                print(f"Warning: GT video '{gt_video_id}' not found in video index for query {i}")
        
        # Calculate recall metrics
        recalls = []
        for k in Ks:
            recall_at_k = 100.0 * hits[k] / n_queries if n_queries > 0 else 0.0
            recalls.append(recall_at_k)
            print(f"Binary HD R@{k:<3}= {recall_at_k:.1f}% ({hits[k]}/{n_queries})")
        
        recall_sum = sum(recalls)
        print(f"Binary HD SumR = {recall_sum:.1f}")
        
        # Return in format expected by validation framework
        return recalls + [recall_sum], query_metas

    def compute_context_info_hd(self, model, context_dataloader):
        """
        Compute HD context information for all videos.
        """
        n_total_vid = len(context_dataloader.dataset)
        bsz = self.cfg['eval_context_bsz']
        metas = []  # list(dicts)
        video_hd_embeddings = []
        video_binary_embeddings = []
        
        for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing HD video embeddings",
                            total=len(context_dataloader)):

            batch = gpu(batch)
            metas.extend(batch[-1])
            clip_video_feat_ = batch[0]  # (batch, 32, visual_input_size)
            
            # Encode to HD space and aggregate
            video_hd = model.encode_clip_context_hd(clip_video_feat_)  # (batch, hd_dim)
            video_hd_embeddings.append(video_hd)
            
            # Also compute packed binary embeddings if needed
            if self.use_binary:
                video_packed = _pack_hd_embeddings(video_hd)
                video_binary_embeddings.append(video_packed)

        video_hd_embeddings = torch.cat(video_hd_embeddings, dim=0)  # (N_videos, hd_dim)
        
        result_dict = dict(
            video_metas=metas,  # list(dict) (N_videos)
            video_hd_embeddings=video_hd_embeddings,
        )
        
        if self.use_binary:
            video_packed_embeddings = torch.cat(video_binary_embeddings, dim=0)
            result_dict["video_packed_embeddings"] = video_packed_embeddings
                
        return result_dict


class dual_mode_hd_validations(nn.Module):
    """
    Validation that runs both float HD and binary HD evaluation for comparison.
    """
    def __init__(self, cfg):
        super(dual_mode_hd_validations, self).__init__()
        self.cfg = cfg
        self.float_validator = clip_only_hd_validations(cfg)
        
        # Create binary validator
        binary_cfg = cfg.copy()
        binary_cfg['use_binary_inference'] = True
        self.binary_validator = clip_only_hd_validations(binary_cfg)

    def forward(self, model, context_dataloader, query_eval_loader):
        """
        Run both float and binary evaluation efficiently by sharing context computation.
        """
        model.eval()
        
        print("\n=== Computing shared HD context (once) ===")
        # Compute context info once with both float and binary embeddings
        shared_context_info = self.compute_shared_context_info(model, context_dataloader)
        
        print("\n=== Float HD Evaluation ===")
        # Float evaluation using shared context
        float_score_sum, query_metas = self.float_validator.compute_query2ctx_info_hd(
            model, query_eval_loader, shared_context_info)
        
        video_metas = shared_context_info['video_metas']
        v2t_gt, t2v_gt = get_gt(video_metas, query_metas)
        t2v_r1, t2v_r5, t2v_r10, t2v_r100 = cal_perf(-1 * float_score_sum, t2v_gt)
        t2v_rsum = t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100
        float_results = [t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_rsum]
        
        print("\n=== Binary HD Evaluation ===")  
        # Binary evaluation using shared context (with binary embeddings)
        binary_results, _ = self.binary_validator.compute_query2ctx_info_binary(
            model, query_eval_loader, shared_context_info)
        
        print(f"\n=== Comparison ===")
        print(f"Float HD  - R@1: {float_results[0]:.1f}, R@5: {float_results[1]:.1f}, R@10: {float_results[2]:.1f}, R@100: {float_results[3]:.1f}, RSum: {float_results[4]:.1f}")
        print(f"Binary HD - R@1: {binary_results[0]:.1f}, R@5: {binary_results[1]:.1f}, R@10: {binary_results[2]:.1f}, R@100: {binary_results[3]:.1f}, RSum: {binary_results[4]:.1f}")
        
        # Return float results as primary metric for logging
        return float_results
    
    def compute_shared_context_info(self, model, context_dataloader):
        """
        Compute context information once for both float and binary evaluation.
        """
        n_total_vid = len(context_dataloader.dataset)
        metas = []
        video_hd_embeddings = []
        video_binary_embeddings = []
        
        for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing shared HD video embeddings",
                            total=len(context_dataloader)):

            batch = gpu(batch)
            metas.extend(batch[-1])
            clip_video_feat_ = batch[0]  # (batch, 32, visual_input_size)
            
            # Encode to HD space and aggregate (compute once)
            video_hd = model.encode_clip_context_hd(clip_video_feat_)  # (batch, hd_dim)
            video_hd_embeddings.append(video_hd)
            
            # Always compute binary embeddings for dual-mode
            video_packed = _pack_hd_embeddings(video_hd)
            video_binary_embeddings.append(video_packed)

        video_hd_embeddings = torch.cat(video_hd_embeddings, dim=0)  # (N_videos, hd_dim)
        video_packed_embeddings = torch.cat(video_binary_embeddings, dim=0)  # (N_videos, hd_dim//64)
        
        return dict(
            video_metas=metas,
            video_hd_embeddings=video_hd_embeddings,
            video_packed_embeddings=video_packed_embeddings,
        )