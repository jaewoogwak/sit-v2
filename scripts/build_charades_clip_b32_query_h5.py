#!/usr/bin/env python3
"""
Build a single Charades CLIP query HDF5 from caption txt files.

Input caption format:
  <cap_id> <caption text>

Output HDF5:
  key=cap_id, value shape=(1, 512) by default
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from typing import Iterable

import h5py
import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Build Charades CLIP query HDF5 from caption txt files.")
    p.add_argument(
        "--caption_txt",
        type=str,
        nargs="+",
        default=[
            "/dev/ssd1/gjw/prvr/dataset/charades/TextData/charadestrain.caption.txt",
            "/dev/ssd1/gjw/prvr/dataset/charades/TextData/charadesval.caption.txt",
            "/dev/ssd1/gjw/prvr/dataset/charades/TextData/charadestest.caption.txt",
        ],
        help="Caption txt files to merge. Duplicate cap_id keeps first occurrence.",
    )
    p.add_argument(
        "--out_h5",
        type=str,
        default="/dev/ssd1/gjw/prvr/dataset/charades/TextData/clip_ViT_B_32_charades_query_feat.hdf5",
    )
    p.add_argument("--model_name", type=str, default="ViT-B-32-quickgelu")
    p.add_argument("--pretrained", type=str, default="openai")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--store_2d",
        action="store_true",
        help="Store each query as (1, D) instead of (D,) for compatibility with legacy files.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _read_caption_entries(paths: Iterable[str]) -> OrderedDict[str, str]:
    entries: OrderedDict[str, str] = OrderedDict()
    total = 0
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                cap_id, text = parts
                total += 1
                if cap_id not in entries:
                    entries[cap_id] = text
    print(f"[load] lines={total} unique_cap_ids={len(entries)}")
    return entries


def main():
    args = parse_args()
    _ensure_parent(args.out_h5)
    if os.path.exists(args.out_h5):
        if not args.overwrite:
            raise FileExistsError(f"{args.out_h5} exists. Use --overwrite.")
        os.remove(args.out_h5)

    entries = _read_caption_entries(args.caption_txt)
    if not entries:
        raise RuntimeError("No valid caption entries found.")

    try:
        import open_clip
    except Exception as exc:
        raise RuntimeError("open_clip_torch is required.") from exc

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")
    print(f"[model] {args.model_name} / {args.pretrained}")

    model, _, _ = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()

    cap_ids = list(entries.keys())
    texts = [entries[k] for k in cap_ids]

    with h5py.File(args.out_h5, "w") as f:
        f.attrs["clip_impl"] = "open_clip"
        f.attrs["clip_model"] = args.model_name
        f.attrs["pretrained"] = args.pretrained
        f.attrs["query_source"] = "encode_text_pooled_normalize_false"
        f.attrs["store_shape"] = "1xD" if args.store_2d else "D"

        bs = int(args.batch_size)
        for i in tqdm(range(0, len(cap_ids), bs), desc="Encoding", unit="cap"):
            chunk_ids = cap_ids[i : i + bs]
            chunk_texts = texts[i : i + bs]
            with torch.no_grad():
                token_ids = tokenizer(chunk_texts).to(device)
                pooled = model.encode_text(token_ids, normalize=False)
            arr = pooled.detach().float().cpu().numpy().astype(np.float32, copy=False)
            for j, cid in enumerate(chunk_ids):
                vec = arr[j]
                if args.store_2d:
                    vec = vec.reshape(1, -1)
                f.create_dataset(cid, data=vec, dtype=np.float32)

    print(f"[done] wrote: {args.out_h5}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

