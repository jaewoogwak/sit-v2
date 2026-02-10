import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

def onehot(indexes, N=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().long().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    return output


def tome_merge_tokens(tokens, ratio=None, target_len=None, eps=1e-6):
    """Merge tokens by similarity with weighted averaging based on merge counts."""
    if tokens.dim() != 2:
        return tokens
    length = tokens.size(0)
    if length <= 1:
        return tokens

    target_from_ratio = None
    if ratio is not None and 0.0 < ratio < 1.0:
        target_from_ratio = max(1, int(math.ceil(length * ratio)))
    if target_len is not None and target_len > 0:
        target = min(length, int(target_len))
        if target_from_ratio is not None:
            target = min(target, target_from_ratio)
    elif target_from_ratio is not None:
        target = target_from_ratio
    else:
        return tokens

    weights = torch.ones(length, device=tokens.device, dtype=tokens.dtype)
    merged_tokens = tokens
    merged_weights = weights

    while merged_tokens.size(0) > target and merged_tokens.size(0) > 1:
        length = merged_tokens.size(0)
        merges_needed = min(length - target, length // 2)

        with torch.no_grad():
            normed = merged_tokens / (merged_tokens.norm(dim=-1, keepdim=True) + eps)
            sim = torch.matmul(normed, normed.t())
            sim.fill_diagonal_(-float("inf"))
            best_match = sim.argmax(dim=1)
            best_score = sim[torch.arange(length, device=tokens.device), best_match]
            order = torch.argsort(best_score, descending=True)

        used = torch.zeros(length, dtype=torch.bool, device=tokens.device)
        pairs = {}
        merges = 0
        for idx in order.tolist():
            if merges >= merges_needed:
                break
            jdx = int(best_match[idx].item())
            if used[idx] or used[jdx] or idx == jdx:
                continue
            pairs[idx] = jdx
            used[idx] = True
            used[jdx] = True
            merges += 1

        new_tokens = []
        new_weights = []
        i = 0
        while i < length:
            if i in pairs:
                j = pairs[i]
                wi = merged_weights[i]
                wj = merged_weights[j]
                merged = (merged_tokens[i] * wi + merged_tokens[j] * wj) / (wi + wj)
                new_tokens.append(merged)
                new_weights.append(wi + wj)
                i += 1
                continue
            if used[i]:
                i += 1
                continue
            new_tokens.append(merged_tokens[i])
            new_weights.append(merged_weights[i])
            i += 1

        merged_tokens = torch.stack(new_tokens, dim=0)
        merged_weights = torch.stack(new_weights, dim=0)

        if merged_tokens.size(0) == length:
            break

    if merged_tokens.size(0) > target:
        merged_tokens = merged_tokens[:target]
    return merged_tokens


class clip_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(clip_nce, self).__init__()
        self.reduction = reduction

    def forward(self, labels, label_dict, q2ctx_scores=None, contexts=None, queries=None):

        query_bsz = q2ctx_scores.shape[0]
        vid_bsz = q2ctx_scores.shape[1]
        diagnoal = torch.arange(query_bsz).to(q2ctx_scores.device)
        t2v_nominator = q2ctx_scores[diagnoal, labels]

        t2v_nominator = torch.logsumexp(t2v_nominator.unsqueeze(1), dim=1)
        t2v_denominator = torch.logsumexp(q2ctx_scores, dim=1)

        v2t_nominator = torch.zeros(vid_bsz).to(q2ctx_scores)
        v2t_denominator = torch.zeros(vid_bsz).to(q2ctx_scores)

        for i, label in label_dict.items():
            v2t_nominator[i] = torch.logsumexp(q2ctx_scores[label, i], dim=0)

            v2t_denominator[i] = torch.logsumexp(q2ctx_scores[:, i], dim=0)
        if self.reduction:
            return torch.mean(t2v_denominator - t2v_nominator) + torch.mean(v2t_denominator - v2t_nominator)
        else:
            return denominator - nominator


class frame_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(frame_nce, self).__init__()
        self.reduction = reduction

    def forward(self, q2ctx_scores=None, contexts=None, queries=None):

        if q2ctx_scores is None:
            assert contexts is not None and queries is not None
            x = torch.matmul(contexts, queries.t())
            device = contexts.device
            bsz = contexts.shape[0]
        else:
            x = q2ctx_scores
            device = q2ctx_scores.device
            bsz = q2ctx_scores.shape[0]

        x = x.view(bsz, bsz, -1)
        nominator = x * torch.eye(x.shape[0], dtype=torch.float32, device=device)[:, :, None]
        nominator = nominator.sum(dim=1)

        nominator = torch.logsumexp(nominator, dim=1)

        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        if self.reduction:
            return torch.mean(denominator - nominator)
        else:
            return denominator - nominator

class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_position_emb(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return input_feat + position_embeddings


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class SegmentQueryTokenSelector(nn.Module):
    """Slot-based token selector for segment-branch late interaction."""
    def __init__(self, config):
        super(SegmentQueryTokenSelector, self).__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_slots = int(getattr(config, "seg_token_K", 8))
        self.use_proj = bool(getattr(config, "seg_token_proj", True))
        self.temperature = float(getattr(config, "seg_slot_temp", 0.07))
        self.slot_dropout = nn.Dropout(float(getattr(config, "seg_slot_dropout", 0.0)))
        self.infer_hard_topk = bool(getattr(config, "seg_infer_hard_topk", True))
        self.diversity_type = str(getattr(config, "seg_diversity_type", "cosine")).lower()
        self.diversity_margin = float(getattr(config, "seg_diversity_margin", 0.2))
        self.debug_mask_print = bool(getattr(config, "seg_debug_mask_print", False))
        self.debug_mask_every = int(getattr(config, "seg_debug_mask_every", 20))
        self.debug_mask_max_print = int(getattr(config, "seg_debug_mask_max_print", 30))
        self._debug_step = 0
        self._debug_printed = 0
        infer_topk = int(getattr(config, "seg_infer_topk", self.num_slots))
        self.infer_topk = infer_topk if infer_topk > 0 else self.num_slots

        if self.use_proj:
            self.token_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Dropout(float(getattr(config, "seg_slot_dropout", 0.0))),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
            )
        else:
            self.token_proj = nn.Identity()

        self.num_attn_layers = int(getattr(config, "seg_token_bertattn_layers", 1))
        self.context_layers = nn.ModuleList()
        if self.num_attn_layers > 0:
            attn_cfg = type("AttnCfg", (), {})()
            attn_cfg.hidden_size = self.hidden_size
            attn_cfg.intermediate_size = self.hidden_size
            attn_cfg.hidden_dropout_prob = float(getattr(config, "drop", 0.1))
            attn_cfg.num_attention_heads = int(getattr(config, "n_heads", 4))
            attn_cfg.attention_probs_dropout_prob = float(getattr(config, "drop", 0.1))
            for _ in range(self.num_attn_layers):
                self.context_layers.append(BertAttention(attn_cfg, wid=None))

        self.slot_queries = nn.Parameter(torch.empty(self.num_slots, self.hidden_size))
        self.token_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)
        if isinstance(self.token_proj, nn.Sequential):
            for module in self.token_proj:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        module.bias.data.zero_()
        nn.init.normal_(self.token_key.weight, mean=0.0, std=0.02)

    def _build_token_score(self, logits, token_mask):
        masked_logits = logits.masked_fill(token_mask.unsqueeze(1) <= 0, -1e10)
        token_score = masked_logits.max(dim=1).values
        return token_score

    def _hard_select(self, token_feats, token_score, token_mask):
        bsz, seq_len, hsz = token_feats.shape
        k = min(self.infer_topk, seq_len)
        if k <= 0:
            k = min(self.num_slots, seq_len)

        masked_score = token_score.masked_fill(token_mask <= 0, -1e10)
        topk_idx = torch.topk(masked_score, k=k, dim=-1).indices
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, hsz)
        selected = torch.gather(token_feats, dim=1, index=gather_idx)

        if selected.size(1) < self.num_slots:
            pad = torch.zeros(
                (bsz, self.num_slots - selected.size(1), hsz),
                device=selected.device,
                dtype=selected.dtype,
            )
            selected = torch.cat([selected, pad], dim=1)
        elif selected.size(1) > self.num_slots:
            selected = selected[:, :self.num_slots, :]
        return selected

    def forward(self, token_feats, token_mask):
        self._debug_step += 1
        token_mask = token_mask.to(token_feats.device).float()
        token_nonzero = (token_feats.abs().sum(dim=-1) > 0).float()
        token_feats = self.token_proj(token_feats)
        attn_mask = token_mask.unsqueeze(1)  # (B, 1, L) to match BertSelfAttention masking path
        for layer in self.context_layers:
            token_feats = layer(token_feats, attn_mask)

        token_keys = self.token_key(token_feats)
        slot_q = self.slot_queries.unsqueeze(0).expand(token_feats.size(0), -1, -1)
        logits = torch.einsum("bkd,bld->bkl", slot_q, token_keys) / max(self.temperature, 1e-6)
        token_score = self._build_token_score(logits, token_mask)

        if (not self.training) and self.infer_hard_topk:
            slot_feats = self._hard_select(token_feats, token_score, token_mask)
            attn_weights = None
        else:
            logits = logits.masked_fill(token_mask.unsqueeze(1) <= 0, -1e10)
            attn_weights = F.softmax(logits, dim=-1)
            slot_feats = torch.einsum("bkl,bld->bkd", attn_weights, token_feats)
            slot_feats = self.slot_dropout(slot_feats)

        if self.debug_mask_print and self._debug_printed < self.debug_mask_max_print:
            every = max(1, self.debug_mask_every)
            if self._debug_step % every == 0:
                with torch.no_grad():
                    show_n = min(2, token_mask.size(0))
                    valid_count = token_mask.sum(dim=-1)
                    nonzero_count = token_nonzero.sum(dim=-1)
                    token_norm = token_feats.norm(dim=-1)
                    last_k = min(8, token_mask.size(1))
                    head_norm = token_norm[:, :last_k].mean(dim=-1)
                    tail_norm = token_norm[:, -last_k:].mean(dim=-1)
                    if attn_weights is not None:
                        invalid_attn_mass = (attn_weights * (1.0 - token_mask.unsqueeze(1))).sum(dim=-1).mean(dim=-1)
                        tail_attn = attn_weights[:, :, -last_k:].mean(dim=(1, 2))
                    else:
                        invalid_attn_mass = torch.zeros_like(valid_count)
                        tail_attn = torch.zeros_like(valid_count)
                    print(
                        "[seg_mask_debug] "
                        f"step={self._debug_step} "
                        f"valid_count={valid_count[:show_n].detach().cpu().tolist()} "
                        f"nonzero_count={nonzero_count[:show_n].detach().cpu().tolist()} "
                        f"head_norm={head_norm[:show_n].detach().cpu().tolist()} "
                        f"tail_norm={tail_norm[:show_n].detach().cpu().tolist()} "
                        f"invalid_attn_mass={invalid_attn_mass[:show_n].detach().cpu().tolist()} "
                        f"tail_attn={tail_attn[:show_n].detach().cpu().tolist()}"
                    )
                    self._debug_printed += 1

        slot_norm = F.normalize(slot_feats, dim=-1)
        sim = torch.matmul(slot_norm, slot_norm.transpose(1, 2))
        eye = torch.eye(self.num_slots, device=sim.device, dtype=torch.bool).unsqueeze(0)
        off_diag = sim.masked_select(~eye)
        if off_diag.numel() == 0:
            diversity_loss = slot_feats.new_zeros(())
        elif self.diversity_type == "orthogonality":
            diversity_loss = (off_diag ** 2).mean()
        else:
            diversity_loss = F.relu(off_diag - self.diversity_margin).mean()

        return {
            "slot_feats": slot_feats,
            "token_score": token_score,
            "attn_weights": attn_weights,
            "diversity_loss": diversity_loss,
        }



class GMMBlock(nn.Module):
    def __init__(self, config):
        super(GMMBlock, self).__init__()
        self.pure_block = getattr(config, "pure_block", False)
        self.pure_block_ffn = getattr(config, "pure_block_ffn", True)
        if self.pure_block:
            self.attn = BertAttention(config, wid=None)
            if self.pure_block_ffn:
                self.intermediate = BertIntermediate(config)
                self.output = BertOutput(config)
        else:
            self.attn0 = BertAttention(config)
            self.attn1 = BertAttention(config, wid=0.1)
            self.attn2 = BertAttention(config, wid=1.0)
            self.attn3 = BertAttention(config, wid=5.0)
            # self.ca = BertCrossAttention(config)
            # self.agg_fc1 = nn.Linear(config.hidden_size, config.hidden_size)
            # self.agg_fc2 = nn.Linear(config.hidden_size, 4)
        
    def forward(self, input_tensor, attention_mask=None, weight_token=None):
        if self.pure_block:
            attention_output = self.attn(input_tensor, attention_mask)
            if self.pure_block_ffn:
                intermediate_output = self.intermediate(attention_output)
                return self.output(intermediate_output, attention_output)
            return attention_output

        o0 = self.attn0(input_tensor, attention_mask).unsqueeze(-1)
        o1 = self.attn1(input_tensor, attention_mask).unsqueeze(-1)
        o2 = self.attn2(input_tensor, attention_mask).unsqueeze(-1)
        o3 = self.attn3(input_tensor, attention_mask).unsqueeze(-1)
    
        oo = torch.cat([o0, o1, o2, o3], dim=-1)
        # if weight_token is not None:
        #     # Dynamic aggregation over 4 Gaussian branches with variable sequence length.
        #     token = weight_token.expand(input_tensor.size(0), -1, -1)
        #     agg_feat = self.ca(input_tensor, token, token, attention_mask=None)
        #     agg_logits = self.agg_fc2(F.gelu(self.agg_fc1(agg_feat)))
        #     agg_weight = F.softmax(agg_logits, dim=-1).unsqueeze(2)
        #     out = torch.sum(oo * agg_weight, dim=-1)
        # else:
        out = torch.mean(oo, dim=-1)

        return out


class StandardTransformerLayer(nn.Module):
    def __init__(self, config):
        super(StandardTransformerLayer, self).__init__()
        self.attn = BertAttention(config, wid=None)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intermediate_act_fn = F.gelu

    def forward(self, input_tensor, attention_mask=None):
        attention_output = self.attn(input_tensor, attention_mask)
        intermediate_output = self.intermediate_act_fn(self.intermediate(attention_output))
        output = self.output_dense(intermediate_output)
        output = self.dropout(output)
        return self.output_layer_norm(output + attention_output)


class StandardTransformerEncoder(nn.Module):
    def __init__(self, config, num_layers=4):
        super(StandardTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([StandardTransformerLayer(config) for _ in range(num_layers)])

    def forward(self, input_tensor, attention_mask=None):
        for layer in self.layers:
            input_tensor = layer(input_tensor, attention_mask)
        return input_tensor


class IdentityEncoder(nn.Module):
    def forward(self, input_tensor, attention_mask=None):
        return input_tensor


class BertAttention(nn.Module):
    def __init__(self, config, wid=None):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config, wid=wid)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertCrossAttention(nn.Module):
    def __init__(self, config, wid=None):
        super(BertCrossAttention, self).__init__()
        self.self = BertSelfAttention(config, wid=wid)
        self.output = BertSelfOutput(config)

    def forward(self, query_tensor, key_tensor, value_tensor, attention_mask=None):
        self_output = self.self(query_tensor, key_tensor, value_tensor, attention_mask)
        attention_output = self.output(self_output, query_tensor)
        return attention_output


class BertSelfAttention(nn.Module):
    def __init__(self, config, wid=None):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.wid = wid
        

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def generate_gauss_weight(self, props_len, width):

        center = torch.arange(props_len).cuda() / props_len
        width = width*torch.ones(props_len).cuda()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / 9

        w = 0.3989422804014327

        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)
        """

        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        # transpose
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores_ori = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)

        attention_scores_ori = attention_scores_ori / math.sqrt(self.attention_head_size)


        attention_scores = attention_scores_ori
        if self.wid is not None:
            gmm_mask = self.generate_gauss_weight(attention_scores.shape[-1], self.wid)
            gmm_mask = gmm_mask.unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores_ori * gmm_mask
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
            attention_scores = attention_scores + attention_mask
        # attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # compute output context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        intermediate_size = config.hidden_size * 4
        self.dense = nn.Linear(config.hidden_size, intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        intermediate_size = config.hidden_size * 4
        self.dense = nn.Linear(intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
