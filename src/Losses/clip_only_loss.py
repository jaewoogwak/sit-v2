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
    """
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.mrg = config['neg_factor'][0]
        self.alpha = config['neg_factor'][1]
        
    def forward(self, x, label_dict):
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


class clip_only_loss(nn.Module):
    """
    Simplified loss function for clip-only model.
    Only computes clip-level NCE loss, triplet loss, and query diversity loss.
    """
    def __init__(self, cfg):
        super(clip_only_loss, self).__init__()
        self.cfg = cfg
        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.qdl = query_diverse_loss(cfg)

    def forward(self, input_list, batch):
        '''
        Simplified forward for clip-only model.
        Args:
            input_list: [clip_scale_scores, clip_scale_scores_, label_dict, video_query]
            batch: batch data containing query_labels
        '''
        query_labels = batch['text_labels']
        
        clip_scale_scores = input_list[0]    # Normalized scores for triplet loss
        clip_scale_scores_ = input_list[1]   # Unnormalized scores for NCE loss
        label_dict = input_list[2]           # Label dictionary for grouping
        query = input_list[3]                # Query embeddings for diversity loss

        # Compute loss components
        clip_nce_loss = self.cfg['loss_factor'][0] * self.clip_nce_criterion(query_labels, label_dict, clip_scale_scores_)
        clip_trip_loss = self.get_clip_triplet_loss(clip_scale_scores, query_labels)
        qdl_loss = self.cfg['loss_factor'][2] * self.qdl(query, label_dict)

        # Total loss (only clip-level terms)
        total_loss = clip_nce_loss + clip_trip_loss + qdl_loss

        return total_loss

    def get_clip_triplet_loss(self, query_context_scores, labels):
        """
        Compute triplet loss for clip-level retrieval.
        Supports both hard negative mining and random negative sampling.
        """
        v2t_scores = query_context_scores.t()
        t2v_scores = query_context_scores
        labels = np.array(labels)

        # Video-to-Text loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])

            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
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

        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)