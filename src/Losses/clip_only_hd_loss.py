import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.gmmformer.model_components import clip_nce

import ipdb


class query_diverse_loss(nn.Module):
    """
    Query diversity loss to encourage different queries for the same video to be diverse.
    Modified for HD space.
    """
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.mrg = config['neg_factor'][0]
        self.alpha = config['neg_factor'][1]
        
    def forward(self, x, label_dict):
        """
        Args:
            x: HD query embeddings (batch_size, hd_dim)
            label_dict: Label dictionary for grouping queries
        """
        bs = x.shape[0]
        x = F.normalize(x, dim=-1)
        cos = torch.matmul(x, x.t())

        N_one_hot = torch.zeros((bs, bs))
        for i, label in label_dict.items():
            N_one_hot[label[0]:(label[-1]+1), label[0]:(label[-1]+1)] = torch.ones((len(label), len(label)))
        N_one_hot = N_one_hot - torch.eye(bs)
        N_one_hot = N_one_hot.cuda()
    
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        neg_term = torch.log(1 + N_sim_sum).sum() / bs
        
        return neg_term


class hd_contrastive_loss(nn.Module):
    """
    HD-specific contrastive loss for video-text retrieval in hyperdimensional space.
    Applies temperature scaling appropriate for HD similarities.
    """
    def __init__(self, config):
        super(hd_contrastive_loss, self).__init__()
        self.temperature = config.get('hd_temperature', 0.07)
        
    def forward(self, query_hd, video_hd, labels, label_dict):
        """
        Args:
            query_hd: HD query embeddings (N_queries, hd_dim)
            video_hd: HD video embeddings (N_videos, hd_dim) 
            labels: Query labels for matching
            label_dict: Label dictionary for grouping
        """
        # Normalize embeddings
        query_hd = F.normalize(query_hd, dim=-1)
        video_hd = F.normalize(video_hd, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(query_hd, video_hd.t()) / self.temperature  # (N_queries, N_videos)
        
        # Create positive pairs mask
        batch_size = query_hd.shape[0]
        positive_mask = torch.zeros(batch_size, batch_size).to(query_hd.device)
        
        for i, label in enumerate(labels):
            positive_mask[i, label] = 1.0
            
        # Compute contrastive loss (InfoNCE style)
        # Text-to-Video direction
        t2v_logits = similarity_matrix  # (N_queries, N_videos)
        t2v_labels = torch.tensor(labels).to(query_hd.device)
        t2v_loss = F.cross_entropy(t2v_logits, t2v_labels)
        
        # Video-to-Text direction  
        v2t_logits = similarity_matrix.t()  # (N_videos, N_queries)
        v2t_loss = 0
        for i, label_group in label_dict.items():
            if len(label_group) > 0:
                # For each video, compute loss against its positive queries
                pos_logits = v2t_logits[i, label_group]
                # Create target (first positive query as target)
                target = torch.zeros(1).long().to(query_hd.device)
                # Compute loss for this video
                v2t_loss += F.cross_entropy(pos_logits.unsqueeze(0), target)
        
        v2t_loss = v2t_loss / len(label_dict) if len(label_dict) > 0 else 0
        
        return (t2v_loss + v2t_loss) / 2


class clip_only_hd_loss(nn.Module):
    """
    Loss function for HD-enhanced clip-only model.
    Combines HD contrastive loss, triplet loss, and query diversity loss.
    """
    def __init__(self, cfg):
        super(clip_only_hd_loss, self).__init__()
        self.cfg = cfg
        
        # Traditional NCE loss for compatibility (operates on HD similarity scores)
        self.clip_nce_criterion = clip_nce(reduction='mean')
        
        # Query diversity loss (works in HD space)
        self.qdl = query_diverse_loss(cfg)

    def forward(self, input_list, batch):
        '''
        HD loss forward pass.
        Args:
            input_list: [hd_similarity_scores, hd_similarity_scores_, label_dict, query_hd]
            batch: batch data containing query_labels
        '''
        query_labels = batch['text_labels']
        
        hd_similarity_scores = input_list[0]      # Normalized HD similarity scores
        hd_similarity_scores_ = input_list[1]     # Unnormalized HD similarity scores  
        label_dict = input_list[2]                # Label dictionary for grouping
        query_hd = input_list[3]                  # HD query embeddings

        # Compute loss components
        
        # 1. HD NCE Loss (using unnormalized scores)
        hd_nce_loss = self.cfg['loss_factor'][0] * self.clip_nce_criterion(query_labels, label_dict, hd_similarity_scores_)
        
        # 2. HD-based Triplet Loss (adapted for HD similarities)
        hd_triplet_loss = self.get_hd_triplet_loss(hd_similarity_scores, query_labels)
        
        # 3. Query Diversity Loss in HD space
        qdl_loss = self.cfg['loss_factor'][2] * self.qdl(query_hd, label_dict)

        # Total loss (HD-based)
        total_loss = hd_nce_loss + hd_triplet_loss + qdl_loss

        return total_loss

    def get_hd_triplet_loss(self, hd_similarity_scores, labels):
        """
        Compute triplet loss using HD similarity scores.
        Adapted from clip_only_loss.get_clip_triplet_loss()
        """
        v2t_scores = hd_similarity_scores.t()  # (N_videos, N_queries)
        t2v_scores = hd_similarity_scores      # (N_queries, N_videos)
        labels = np.array(labels)

        # Video-to-Text loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            # Find positive queries for this video
            pos_indices = np.where(labels == i)[0]
            if len(pos_indices) == 0:
                continue
                
            pos_pair_scores = torch.mean(v2t_scores[i][pos_indices])

            # Find negative queries
            neg_indices = np.where(labels != i)[0]
            if len(neg_indices) == 0:
                continue
                
            neg_pair_scores, _ = torch.sort(v2t_scores[i][neg_indices], descending=True)
            
            if self.cfg['use_hard_negative']:
                sample_neg_pair_scores = neg_pair_scores[0]  # Hardest negative
            else:
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]

            v2t_loss += (self.cfg['margin'] + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # Text-to-Video loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999  # Mask positive scores
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
        
        t2v_sample_max_idx = min(1 + self.cfg['hard_pool_size'],
                                 t2v_scores.shape[1]) if self.cfg['use_hard_negative'] else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]

        t2v_neg_scores = t2v_scores[text_indices, sample_indices]
        t2v_loss = (self.cfg['margin'] + t2v_neg_scores - t2v_pos_scores).clamp(min=0)

        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / max(1, len(v2t_scores))



class binary_hd_loss(nn.Module):
    """
    Optional binary HD loss for training with binary quantization.
    Can be used for end-to-end binary training if desired.
    """
    def __init__(self, cfg):
        super(binary_hd_loss, self).__init__()
        self.cfg = cfg
        self.temperature = cfg.get('hd_temperature', 0.07)
        
    def forward(self, query_hd, video_hd, labels):
        """
        Binary HD contrastive loss using hamming distance.
        """
        # Binarize HD embeddings (sign function with straight-through estimator)
        query_binary = self.binarize_hd(query_hd)
        video_binary = self.binarize_hd(video_hd)
        
        # Compute hamming distance (normalized to [0,1])
        hamming_dist = self.compute_hamming_distance(query_binary, video_binary)
        
        # Convert to similarity (1 - normalized_hamming_distance)
        similarity = 1.0 - hamming_dist
        
        # Apply temperature scaling
        logits = similarity / self.temperature
        
        # Standard contrastive loss
        targets = torch.tensor(labels).to(query_hd.device)
        loss = F.cross_entropy(logits, targets)
        
        return loss
    
    def binarize_hd(self, x):
        """Binarize HD embeddings using sign function with straight-through estimator"""
        # Forward: sign function, Backward: identity (straight-through)
        return (torch.sign(x) - x).detach() + x
    
    def compute_hamming_distance(self, query_binary, video_binary):
        """Compute normalized hamming distance between binary embeddings"""
        # XOR operation to find differing bits
        xor_result = torch.logical_xor(query_binary > 0, video_binary.unsqueeze(0) > 0)
        
        # Count differing bits and normalize by total bits
        hamming_dist = xor_result.float().mean(dim=-1)  # (N_queries, N_videos)
        
        return hamming_dist