#!/usr/bin/env python3
"""
Compare ActivityNet text features in two HDF5 files against fresh CLIP ViT-B/32 encoding.

Expected files:
  - query_h5: key -> (512,)
  - token_h5: key -> (77, 512)

This script reports numeric agreement between:
  1) query_h5[key] and CLIP pooled text feature (encode_text, normalize=False)
  2) token_h5[key] and CLIP token features after ln_final + text_projection
  3) query_h5[key] and EOT token vector from token_h5[key]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Tuple

import numpy as np
try:
    import h5py
except Exception:
    h5py = None

try:
    import torch
except Exception:
    torch = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare ActivityNet text HDF5 features with CLIP ViT-B/32 re-encoding."
    )
    p.add_argument("--desc_id", type=str, required=True, help="Key like v_xxx#enc#0")
    p.add_argument("--query_h5", type=str, required=True, help="HDF5 path with (512,) vectors")
    p.add_argument("--token_h5", type=str, required=True, help="HDF5 path with (77,512) vectors")
    p.add_argument(
        "--text",
        type=str,
        default="",
        help="Raw query text. If empty, script tries --caption_txt then --jsonl.",
    )
    p.add_argument(
        "--caption_txt",
        type=str,
        default="dataset/activitynet/activitynetval.caption.txt",
        help="Caption txt file with format: '<desc_id> <text>'",
    )
    p.add_argument(
        "--jsonl",
        type=str,
        nargs="*",
        default=[
            "dataset/activitynet/TextData/activitynet_val.jsonl",
            "dataset/activitynet/TextData/activitynet_train.jsonl",
        ],
        help="JSONL files to search when text is not given.",
    )
    p.add_argument("--model_name", type=str, default="ViT-B-32")
    p.add_argument("--pretrained", type=str, default="openai")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _read_text_from_caption_txt(path: str, desc_id: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(desc_id + " "):
                return line[len(desc_id) + 1 :].strip()
    return None


def _read_text_from_jsonl(paths, desc_id: str) -> Optional[str]:
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("desc_id") == desc_id:
                    desc = obj.get("desc")
                    if desc is not None:
                        return str(desc).strip()
    return None


def _encode_batch_open_clip(model, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pooled = model.encode_text(token_ids, normalize=False)  # (B,D)
    x = model.token_embedding(token_ids)
    x = x + model.positional_embedding

    batch_first = False
    try:
        if hasattr(model.transformer, "batch_first"):
            batch_first = bool(model.transformer.batch_first)
        elif hasattr(model.transformer, "resblocks") and len(model.transformer.resblocks) > 0:
            attn = getattr(model.transformer.resblocks[0], "attn", None)
            if attn is not None and hasattr(attn, "batch_first"):
                batch_first = bool(attn.batch_first)
    except Exception:
        batch_first = False

    if batch_first:
        x = model.transformer(x, attn_mask=model.attn_mask)
    else:
        x = x.permute(1, 0, 2)
        x = model.transformer(x, attn_mask=model.attn_mask)
        x = x.permute(1, 0, 2)
    x = model.ln_final(x)
    if getattr(model, "text_projection", None) is not None:
        x = x @ model.text_projection
    return pooled, x


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = float(np.linalg.norm(a))
    bb = float(np.linalg.norm(b))
    if aa == 0.0 or bb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (aa * bb))


def _diff_stats(a: np.ndarray, b: np.ndarray):
    d = np.abs(a - b)
    return float(d.max()), float(d.mean())


def _load_h5_feat(path: str, key: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"Key not found in {path}: {key}")
        arr = f[key][...]
    return np.asarray(arr)


def main():
    args = parse_args()
    if h5py is None:
        raise RuntimeError("h5py is required: pip install h5py")
    if torch is None:
        raise RuntimeError("torch is required: pip install torch")

    text = args.text.strip() if args.text else ""
    if not text:
        text = _read_text_from_caption_txt(args.caption_txt, args.desc_id) or ""
    if not text:
        text = _read_text_from_jsonl(args.jsonl, args.desc_id) or ""
    if not text:
        raise RuntimeError(
            "Query text not found. Provide --text, or valid --caption_txt/--jsonl containing desc_id."
        )

    query_h5_vec = _load_h5_feat(args.query_h5, args.desc_id).astype(np.float32, copy=False)
    token_h5_mat = _load_h5_feat(args.token_h5, args.desc_id).astype(np.float32, copy=False)

    if query_h5_vec.ndim != 1:
        raise RuntimeError(f"Expected query_h5 shape (D,), got {query_h5_vec.shape}")
    if token_h5_mat.ndim != 2:
        raise RuntimeError(f"Expected token_h5 shape (L,D), got {token_h5_mat.shape}")
    if token_h5_mat.shape[1] != query_h5_vec.shape[0]:
        raise RuntimeError(
            f"Dim mismatch: query_h5 D={query_h5_vec.shape[0]}, token_h5 D={token_h5_mat.shape[1]}"
        )

    try:
        import open_clip
    except Exception as exc:
        raise RuntimeError("open_clip is required: pip install open_clip_torch") from exc

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()

    with torch.no_grad():
        token_ids = tokenizer([text]).to(device)  # (1,77)
        pooled, token_lnproj = _encode_batch_open_clip(model, token_ids)

    pooled_np = pooled[0].detach().float().cpu().numpy()
    token_np = token_lnproj[0].detach().float().cpu().numpy()
    eot_idx = int(token_ids[0].argmax().item())
    eot_vec = token_np[eot_idx]
    h5_eot_vec = token_h5_mat[eot_idx]

    print("=== Input ===")
    print(f"desc_id: {args.desc_id}")
    print(f"text: {text}")
    print()
    print("=== Shapes ===")
    print(f"query_h5: {tuple(query_h5_vec.shape)}")
    print(f"token_h5: {tuple(token_h5_mat.shape)}")
    print(f"clip_pooled: {tuple(pooled_np.shape)}")
    print(f"clip_token_lnproj: {tuple(token_np.shape)}")
    print(f"eot_index: {eot_idx}")
    print()
    print("=== Comparison ===")
    q_clip_cos = _cosine(query_h5_vec, pooled_np)
    q_clip_max, q_clip_mean = _diff_stats(query_h5_vec, pooled_np)
    print(f"query_h5 vs clip_pooled      : cos={q_clip_cos:.8f}  max_abs={q_clip_max:.6e}  mean_abs={q_clip_mean:.6e}")

    t_clip_cos = _cosine(token_h5_mat.reshape(-1), token_np.reshape(-1))
    t_clip_max, t_clip_mean = _diff_stats(token_h5_mat, token_np)
    print(f"token_h5 vs clip_token_lnproj: cos={t_clip_cos:.8f}  max_abs={t_clip_max:.6e}  mean_abs={t_clip_mean:.6e}")

    q_eot_cos = _cosine(query_h5_vec, eot_vec)
    q_eot_max, q_eot_mean = _diff_stats(query_h5_vec, eot_vec)
    print(f"query_h5 vs clip_eot_token   : cos={q_eot_cos:.8f}  max_abs={q_eot_max:.6e}  mean_abs={q_eot_mean:.6e}")

    q_h5eot_cos = _cosine(query_h5_vec, h5_eot_vec)
    q_h5eot_max, q_h5eot_mean = _diff_stats(query_h5_vec, h5_eot_vec)
    print(f"query_h5 vs token_h5[eot]    : cos={q_h5eot_cos:.8f}  max_abs={q_h5eot_max:.6e}  mean_abs={q_h5eot_mean:.6e}")

    print()
    print("=== Interpretation Guide ===")
    print("- If query_h5 vs clip_pooled is near-identical, query_h5 stores encode_text pooled features.")
    print("- If token_h5 vs clip_token_lnproj is near-identical, token_h5 stores per-token ln_final+text_projection.")
    print("- If query_h5 matches token_h5[eot], pooled output corresponds to EOT token representation.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
