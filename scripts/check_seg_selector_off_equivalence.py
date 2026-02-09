#!/usr/bin/env python3
import copy
import os
import sys
import torch
from easydict import EasyDict as EDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.Models.gmmformer.model_timing import GMMFormer_Net


def _make_config():
    return EDict(
        visual_input_size=512,
        query_input_size=512,
        hidden_size=512,
        max_ctx_l=32,
        max_desc_l=77,
        map_size=32,
        input_drop=0.2,
        drop=0.2,
        n_heads=4,
        initializer_range=0.02,
        segment_max_l=32,
        max_segments=None,
        segment_merge_ratio=0.85,
        segment_merge_target=32,
        margin=0.1,
        use_hard_negative=False,
        hard_pool_size=20,
        pure_block=False,
        pure_block_ffn=True,
        context_encoder_type="gmm",
        std_transformer_layers=4,
        std_transformer_heads=8,
        std_transformer_ffn_dim=2048,
        use_soft_mil=False,
        use_seg_token_selector=False,
    )


def _make_batch(device):
    b_v = 3
    b_q = 6
    n_seg = 9
    n_frm = 24
    lq = 16
    d = 512
    return {
        "clip_video_features": torch.randn(b_v, n_seg, d, device=device),
        "frame_video_features": torch.randn(b_v, n_frm, d, device=device),
        "videos_mask": torch.ones(b_v, n_frm, device=device),
        "text_feat": torch.randn(b_q, lq, d, device=device),
        "text_mask": torch.ones(b_q, lq, device=device),
        "text_labels": [0, 0, 1, 1, 2, 2],
    }


def _assert_equal_tensor(a, b, name):
    if not torch.equal(a, b):
        max_diff = (a - b).abs().max().item()
        raise AssertionError(f"{name} mismatch (max_abs_diff={max_diff})")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_a = _make_config()
    cfg_b = copy.deepcopy(cfg_a)
    cfg_b.seg_token_K = 12
    cfg_b.seg_token_proj = True
    cfg_b.seg_token_bertattn_layers = 2
    cfg_b.seg_slot_temp = 0.09
    cfg_b.seg_slot_dropout = 0.11
    cfg_b.seg_diversity_weight = 0.35
    cfg_b.seg_diversity_type = "orthogonality"
    cfg_b.seg_diversity_margin = 0.1
    cfg_b.seg_ts_overlap_thr = 0.4
    cfg_b.seg_infonce_temp = 0.05
    cfg_b.seg_infonce_weight = 0.8
    cfg_b.seg_infer_hard_topk = False
    cfg_b.seg_infer_topk = 10
    cfg_b.use_seg_token_selector = False

    torch.manual_seed(1234)
    model_a = GMMFormer_Net(cfg_a).to(device).eval()
    torch.manual_seed(1234)
    model_b = GMMFormer_Net(cfg_b).to(device).eval()

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    if state_a.keys() != state_b.keys():
        raise AssertionError("state_dict keys changed when selector is disabled")
    for k in state_a.keys():
        _assert_equal_tensor(state_a[k], state_b[k], f"state_dict[{k}]")

    batch = _make_batch(device)
    with torch.no_grad():
        out_a = model_a(batch)
        out_b = model_b(batch)

    if len(out_a) != len(out_b):
        raise AssertionError(f"output length mismatch: {len(out_a)} vs {len(out_b)}")
    for idx, (va, vb) in enumerate(zip(out_a, out_b)):
        if torch.is_tensor(va):
            _assert_equal_tensor(va, vb, f"forward[{idx}]")
        else:
            if va != vb:
                raise AssertionError(f"forward[{idx}] mismatch")

    print("PASS: selector-disabled path is numerically identical for initialization and forward outputs.")


if __name__ == "__main__":
    main()
