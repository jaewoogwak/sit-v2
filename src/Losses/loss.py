import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.gmmformer.model_components import clip_nce, frame_nce

import ipdb


class query_diverse_loss(nn.Module):
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


class loss(nn.Module):
    def __init__(self, cfg):
        super(loss, self).__init__()
        self.cfg = cfg
        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.video_nce_criterion = clip_nce(reduction='mean')

        self.qdl = query_diverse_loss(cfg)
        self._hier_eps = 1e-6

    def forward(self, input_list, batch):
        '''
        param: query_labels: List[int]
        param: clip_scale_scores.shape = [5*bs,bs]
        param: frame_scale_scores.shape = [5*bs,5*bs]
        param: clip_scale_scores_.shape = [5*bs,bs]
        param: frame_scale_scores_.shape = [5*bs,5*bs]
        param: label_dict: Dict[List]
        '''

        query_labels = batch['text_labels']
        
        clip_scale_scores = input_list[0]
        clip_scale_scores_ = input_list[1]
        label_dict = input_list[2]
        frame_scale_scores = input_list[3]
        frame_scale_scores_ = input_list[4]

        use_soft_mil = bool(self.cfg.get('use_soft_mil', False))
        if use_soft_mil:
            query = input_list[5]
        else:
            query = input_list[-1]

        if use_soft_mil and len(input_list) >= 8 and 'segment_bounds' in batch and 'text_ts' in batch:
            encoded_clip_feat = input_list[6]
            clip_mask = input_list[7]
            soft_mil_scores = self.get_soft_mil_scores(
                encoded_clip_feat, clip_mask, query, query_labels, batch
            )
            clip_nce_loss = self.cfg['loss_factor'][0] * self.clip_nce_criterion(
                query_labels, label_dict, soft_mil_scores
            )
        else:
            clip_nce_loss = self.cfg['loss_factor'][0] * self.clip_nce_criterion(
                query_labels, label_dict, clip_scale_scores_
            )
        clip_trip_loss = self.get_clip_triplet_loss(clip_scale_scores, query_labels)

        frame_nce_loss = self.cfg['loss_factor'][1] * self.video_nce_criterion(query_labels, label_dict, frame_scale_scores_)
        frame_trip_loss = self.get_clip_triplet_loss(frame_scale_scores, query_labels)

        qdl_loss = self.cfg['loss_factor'][2] * self.qdl(query, label_dict)

        loss = clip_nce_loss + clip_trip_loss + frame_nce_loss + frame_trip_loss + qdl_loss

        # Optional: segment hierarchy loss (parent-attract, sibling-repel within same parent)
        hier_w = 0.0
        if isinstance(self.cfg.get('loss_factor'), (list, tuple)) and len(self.cfg['loss_factor']) > 3:
            hier_w = float(self.cfg['loss_factor'][3])
        if hier_w > 0 and use_soft_mil and len(input_list) >= 8 and 'segment_bounds' in batch:
            segment_bounds = batch.get('segment_bounds')
            segment_bounds_mask = batch.get('segment_bounds_mask')
            if segment_bounds is not None:
                hier_loss, hier_stats = self.get_segment_hierarchy_loss(
                    encoded_clip_feat, clip_mask, segment_bounds, segment_bounds_mask
                )
                self._last_hier_stats = hier_stats
                loss = loss + hier_w * hier_loss
            else:
                self._last_hier_stats = {"enabled": False, "reason": "no_segment_bounds"}
        else:
            self._last_hier_stats = {"enabled": False, "reason": "disabled"}

        return loss

    def get_segment_hierarchy_loss(self, encoded_clip_feat, clip_mask, segment_bounds, segment_bounds_mask=None):
        """
        Parent-attract / sibling-repel loss within the same parent.
        encoded_clip_feat: (Nv, S, D)
        clip_mask: (Nv, S) or None
        segment_bounds: (Nv, S, 2) in seconds
        segment_bounds_mask: (Nv, S) or None
        """
        # Align lengths if segment bounds and clip features disagree in S
        s_feat = encoded_clip_feat.size(1)
        s_bounds = segment_bounds.size(1)
        if s_feat != s_bounds:
            s_min = min(s_feat, s_bounds)
            encoded_clip_feat = encoded_clip_feat[:, :s_min]
            segment_bounds = segment_bounds[:, :s_min]
            if clip_mask is not None:
                clip_mask = clip_mask[:, :s_min]
            if segment_bounds_mask is not None:
                segment_bounds_mask = segment_bounds_mask[:, :s_min]

        if clip_mask is None and segment_bounds_mask is None:
            valid_mask = torch.ones(encoded_clip_feat.shape[:2], device=encoded_clip_feat.device, dtype=torch.bool)
        else:
            if clip_mask is None:
                valid_mask = segment_bounds_mask > 0
            elif segment_bounds_mask is None:
                valid_mask = clip_mask > 0
            else:
                valid_mask = (clip_mask > 0) & (segment_bounds_mask > 0)

        margin = float(self.cfg.get('margin', 0.1))
        total = encoded_clip_feat.new_zeros(())
        count = 0
        vids_used = 0
        anchors = 0
        skipped_no_parent = 0
        skipped_no_sibling = 0
        eps = self._hier_eps

        for v in range(encoded_clip_feat.size(0)):
            vmask = valid_mask[v]
            if vmask.sum().item() < 2:
                continue
            feats = encoded_clip_feat[v][vmask]
            bounds = segment_bounds[v][vmask]
            if feats.numel() == 0 or bounds.numel() == 0:
                continue
            vids_used += 1

            feats = F.normalize(feats, dim=-1)
            sim = torch.matmul(feats, feats.t())
            s = bounds[:, 0]
            e = bounds[:, 1]
            seg_len = torch.clamp(e - s, min=eps)

            s_j = s[:, None]
            e_j = e[:, None]
            s_i = s[None, :]
            e_i = e[None, :]
            len_j = seg_len[:, None]
            len_i = seg_len[None, :]

            contains = (s_j <= s_i) & (e_i <= e_j)
            equal_bounds = (torch.abs(s_j - s_i) < eps) & (torch.abs(e_j - e_i) < eps)
            proper = (len_j > len_i) | equal_bounds
            parent = contains & proper
            parent.fill_diagonal_(False)

            n = sim.size(0)
            for i in range(n):
                anchors += 1
                parent_idx = torch.nonzero(parent[:, i], as_tuple=False).squeeze(1)
                if parent_idx.numel() == 0:
                    skipped_no_parent += 1
                    continue

                p_len = seg_len[parent_idx]
                parent_pow = float(self.cfg.get('hier_parent_pow', 1.0))
                w = 1.0 / torch.clamp(p_len, min=eps).pow(parent_pow)
                w = w / torch.clamp(w.sum(), min=eps)
                pos = (sim[i, parent_idx] * w).sum()

                sib_mask = torch.zeros(n, device=sim.device, dtype=torch.bool)
                for j in parent_idx.tolist():
                    sib_mask |= parent[j]
                sib_mask[i] = False
                if not sib_mask.any():
                    skipped_no_sibling += 1
                    continue
                neg_scores = sim[i, sib_mask]
                loss_i = F.relu(margin + neg_scores - pos).mean()
                total = total + loss_i
                count += 1

        stats = {
            "enabled": True,
            "count": int(count),
            "vids": int(vids_used),
            "anchors": int(anchors),
            "skipped_no_parent": int(skipped_no_parent),
            "skipped_no_sibling": int(skipped_no_sibling),
        }
        if count == 0:
            stats["loss"] = 0.0
            return total, stats
        stats["loss"] = float((total / float(count)).detach().cpu().item())
        return total / float(count), stats

    def get_soft_mil_scores(self, encoded_clip_feat, clip_mask, query, labels, batch):
        """
        Compute soft-MIL aggregated clip scores using GT timestamps.
        encoded_clip_feat: (Nv, S, D)
        clip_mask: (Nv, S)
        query: (Nq, D)
        labels: list[int] length Nq (query -> video index in batch)
        """
        if clip_mask is None:
            clip_mask = torch.ones(encoded_clip_feat.shape[:2], device=encoded_clip_feat.device)
        if clip_mask.dim() == 1:
            clip_mask = clip_mask.unsqueeze(0)

        if query.dim() == 1:
            query = query.unsqueeze(0)

        seg_scores = torch.matmul(encoded_clip_feat, query.t()).permute(2, 0, 1)  # (Nq, Nv, S)

        clip_mask_exp = clip_mask.unsqueeze(0).to(seg_scores.device)
        seg_scores_masked = seg_scores.masked_fill(clip_mask_exp == 0, -1e10)
        base_scores = torch.logsumexp(seg_scores_masked, dim=-1)  # (Nq, Nv)

        segment_bounds = batch['segment_bounds'].to(seg_scores.device)
        text_ts = batch['text_ts'].to(seg_scores.device)
        text_ts_mask = batch.get('text_ts_mask', None)
        if text_ts_mask is None:
            text_ts_mask = torch.ones(text_ts.shape[0], device=text_ts.device)
        text_ts_mask = text_ts_mask.to(seg_scores.device)

        pos_indices = torch.tensor(labels, device=seg_scores.device, dtype=torch.long)
        q_indices = torch.arange(seg_scores.shape[0], device=seg_scores.device)

        pos_seg_scores = seg_scores[q_indices, pos_indices]  # (Nq, S)
        pos_seg_mask = clip_mask[pos_indices]  # (Nq, S)
        pos_bounds = segment_bounds[pos_indices]  # (Nq, S, 2)
        if pos_bounds.size(1) != pos_seg_scores.size(1):
            min_len = min(pos_bounds.size(1), pos_seg_scores.size(1))
            pos_seg_scores = pos_seg_scores[:, :min_len]
            pos_seg_mask = pos_seg_mask[:, :min_len]
            pos_bounds = pos_bounds[:, :min_len, :]

        ts_start = text_ts[:, 0].unsqueeze(1)
        ts_end = text_ts[:, 1].unsqueeze(1)
        seg_start = pos_bounds[:, :, 0]
        seg_end = pos_bounds[:, :, 1]

        inter = torch.clamp(torch.min(seg_end, ts_end) - torch.max(seg_start, ts_start), min=0.0)
        seg_len = torch.clamp(seg_end - seg_start, min=1e-6)
        weights = inter / seg_len
        weights = weights * pos_seg_mask

        fallback_weights = pos_seg_mask / torch.clamp(pos_seg_mask.sum(dim=1, keepdim=True), min=1.0)
        weights = torch.where(text_ts_mask.unsqueeze(1) > 0, weights, fallback_weights)

        weight_sum = weights.sum(dim=1, keepdim=True)
        weights = torch.where(weight_sum > 0, weights / weight_sum, fallback_weights)

        log_w = torch.where(weights > 0, torch.log(weights), torch.full_like(weights, -1e10))
        pos_scores_masked = pos_seg_scores.masked_fill(pos_seg_mask == 0, -1e10)
        pos_score = torch.logsumexp(pos_scores_masked + log_w, dim=1)  # (Nq,)

        base_scores = base_scores.clone()
        base_scores[q_indices, pos_indices] = pos_score
        return base_scores

    def get_clip_triplet_loss(self, query_context_scores, labels):
        v2t_scores = query_context_scores.t()
        t2v_scores = query_context_scores
        labels = np.array(labels)

        # cal_v2t_loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])


            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
            if self.cfg['use_hard_negative']:
                sample_neg_pair_scores = neg_pair_scores[0]
            else:
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]

            v2t_loss += (self.cfg['margin'] + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # cal_t2v_loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
        t2v_sample_max_idx = min(1 + self.cfg['hard_pool_size'],
                                 t2v_scores.shape[1]) if self.cfg['use_hard_negative'] else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]

        t2v_neg_scores = t2v_scores[text_indices, sample_indices]

        t2v_loss = (self.cfg['margin'] + t2v_neg_scores - t2v_pos_scores).clamp(min=0)

        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)

    def get_frame_trip_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """

        bsz = len(query_context_scores)

        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx + loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """

        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)

        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)

        sample_min_idx = 1  # skip the masked positive

        sample_max_idx = min(sample_min_idx + self.cfg['hard_pool_size'], bsz) if self.cfg['use_hard_negative'] else bsz

        # sample_max_idx = 2

        # (N, )
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]

        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        return torch.clamp(self.cfg['margin'] + neg_score - pos_score, min=0).sum() / len(pos_score)
