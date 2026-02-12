#!/usr/bin/env python3
"""
Build ActivityNet CLIP text pooled-query HDF5 (key -> (512,)).

Uses open_clip encode_text(normalize=False), which corresponds to CLIP pooled
text representation (EOT-based).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Encode ActivityNet descriptions with CLIP and write pooled query HDF5."
    )
    p.add_argument(
        "--jsonl",
        type=str,
        nargs="+",
        required=True,
        help="Input jsonl paths (e.g., activitynet_train.jsonl activitynet_val.jsonl).",
    )
    p.add_argument(
        "--out_query_h5",
        type=str,
        required=True,
        help="Output pooled feature HDF5 path (key -> (512,)).",
    )
    p.add_argument("--model_name", type=str, default="ViT-B-32-quickgelu")
    p.add_argument("--pretrained", type=str, default="openai")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--query_dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Storage dtype for pooled query features.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _check_overwrite(path: str, overwrite: bool):
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise FileExistsError(f"Output exists: {path} (use --overwrite)")


def _to_numpy_dtype(name: str):
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    if name == "float64":
        return np.float64
    raise ValueError(name)


def _load_jsonl_entries(paths: List[str]) -> List[Tuple[str, str]]:
    entries: Dict[str, str] = {}
    total_lines = 0
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing jsonl: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                obj = json.loads(line)
                desc_id = obj.get("desc_id")
                desc = obj.get("desc")
                if not desc_id or desc is None:
                    continue
                if desc_id not in entries:
                    entries[desc_id] = str(desc)
    print(f"[load] jsonl_lines={total_lines} unique_desc_id={len(entries)}")
    return sorted(entries.items(), key=lambda x: x[0])


def main():
    args = parse_args()
    np_dtype = _to_numpy_dtype(args.query_dtype)

    _ensure_parent(args.out_query_h5)
    _check_overwrite(args.out_query_h5, args.overwrite)

    try:
        import open_clip
    except Exception as exc:
        raise RuntimeError("open_clip is required. Install open_clip_torch.") from exc

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")
    print(f"[model] loading open_clip model={args.model_name} pretrained={args.pretrained}")

    model, _, _ = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()

    entries = _load_jsonl_entries(args.jsonl)
    if len(entries) == 0:
        raise RuntimeError("No valid entries loaded from jsonl.")

    context_length = int(getattr(model, "context_length", 77))
    embed_dim = int(getattr(model, "text_projection").shape[-1]) if getattr(model, "text_projection", None) is not None else int(model.token_embedding.weight.shape[-1])
    print(f"[model] context_length={context_length} embed_dim={embed_dim}")

    bs = int(args.batch_size)
    n = len(entries)

    with h5py.File(args.out_query_h5, "w") as f_query:
        f_query.attrs["clip_impl"] = "open_clip"
        f_query.attrs["clip_model"] = args.model_name
        f_query.attrs["pretrained"] = args.pretrained
        f_query.attrs["context_length"] = context_length
        f_query.attrs["embed_dim"] = embed_dim
        f_query.attrs["query_source"] = "encode_text_pooled"
        f_query.attrs["dtype"] = args.query_dtype

        pbar = tqdm(total=n, desc="Encoding pooled text", unit="cap")
        for i in range(0, n, bs):
            chunk = entries[i : i + bs]
            keys = [x[0] for x in chunk]
            texts = [x[1] for x in chunk]
            with torch.no_grad():
                token_ids = tokenizer(texts).to(device)
                pooled = model.encode_text(token_ids, normalize=False)
            pooled_np = pooled.detach().float().cpu().numpy().astype(np_dtype, copy=False)
            for j, key in enumerate(keys):
                f_query.create_dataset(key, data=pooled_np[j], dtype=np_dtype)
            pbar.update(len(chunk))
            if (i // bs) % 20 == 0:
                pbar.set_postfix({"written": i + len(chunk)})
        pbar.close()

    print(f"[done] wrote query_h5: {args.out_query_h5}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
