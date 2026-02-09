#!/usr/bin/env python3
"""
Build ActivityNet CLIP ViT-B/32 text HDF5 files with root-level desc_id keys.

Outputs:
  1) pooled query features: key -> (512,)
  2) token features:        key -> (77, 512)
  3) token masks:           key -> (77,)

The key format matches current files, e.g. "v_xxx#enc#0".
"""

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
    parser = argparse.ArgumentParser(
        description="Encode ActivityNet descriptions with CLIP B/32 and write HDF5 (pooled + tokens + mask)."
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        nargs="+",
        required=True,
        help="Input jsonl paths (e.g., activitynet_train.jsonl activitynet_val.jsonl).",
    )
    parser.add_argument(
        "--out_query_h5",
        type=str,
        default="",
        help="Output pooled feature HDF5 path (key -> (512,)).",
    )
    parser.add_argument(
        "--out_token_h5",
        type=str,
        required=True,
        help="Output token feature HDF5 path (key -> (77,512)).",
    )
    parser.add_argument(
        "--out_mask_h5",
        type=str,
        required=True,
        help="Output token mask HDF5 path (key -> (77,)).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-32",
        help="open_clip model name.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="open_clip pretrained tag.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for text encoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu.",
    )
    parser.add_argument(
        "--token_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for token features.",
    )
    parser.add_argument(
        "--query_dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Storage dtype for pooled query features.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they exist.",
    )
    return parser.parse_args()


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


def _load_jsonl_entries(paths: List[str]) -> List[Tuple[str, str]]:
    entries: Dict[str, str] = {}
    total_lines = 0
    for path in paths:
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
                    entries[desc_id] = desc
    print(f"[load] jsonl_lines={total_lines} unique_desc_id={len(entries)}")
    items = sorted(entries.items(), key=lambda x: x[0])
    return items


def _to_numpy_dtype(name: str):
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    if name == "float64":
        return np.float64
    raise ValueError(name)


def _encode_batch_open_clip(model, token_ids: torch.Tensor):
    """
    Returns:
      pooled: (B, D)  # CLIP pooled text feature
      token_lnproj: (B, L, D)
    """
    # pooled output (same branch as model.encode_text)
    pooled = model.encode_text(token_ids, normalize=False)

    # token-level hidden states with ln_final + text_projection
    x = model.token_embedding(token_ids)
    x = x + model.positional_embedding

    # open_clip versions differ: some text transformers are batch_first, some are seq_first.
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


def main():
    args = parse_args()

    for p in args.jsonl:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing jsonl: {p}")

    _ensure_parent(args.out_token_h5)
    _ensure_parent(args.out_mask_h5)
    _check_overwrite(args.out_token_h5, args.overwrite)
    _check_overwrite(args.out_mask_h5, args.overwrite)

    write_query = bool(args.out_query_h5)
    if write_query:
        _ensure_parent(args.out_query_h5)
        _check_overwrite(args.out_query_h5, args.overwrite)

    try:
        import open_clip
    except Exception as exc:
        raise RuntimeError(
            "open_clip is required. Install it in your env (e.g., pip install open_clip_torch)."
        ) from exc

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

    token_np_dtype = _to_numpy_dtype(args.token_dtype)
    query_np_dtype = _to_numpy_dtype(args.query_dtype)

    context_length = int(getattr(model, "context_length", 77))
    embed_dim = int(getattr(model, "text_projection").shape[-1]) if getattr(model, "text_projection", None) is not None else int(model.token_embedding.weight.shape[-1])
    print(f"[model] context_length={context_length} embed_dim={embed_dim}")

    with h5py.File(args.out_token_h5, "w") as f_token, h5py.File(args.out_mask_h5, "w") as f_mask:
        f_query = h5py.File(args.out_query_h5, "w") if write_query else None
        try:
            # metadata
            for h5f in [f_token, f_mask] + ([f_query] if f_query is not None else []):
                h5f.attrs["clip_impl"] = "open_clip"
                h5f.attrs["clip_model"] = args.model_name
                h5f.attrs["pretrained"] = args.pretrained
                h5f.attrs["context_length"] = context_length
                h5f.attrs["embed_dim"] = embed_dim
            f_token.attrs["token_source"] = "last_hidden_lnproj"
            f_token.attrs["dtype"] = args.token_dtype
            f_mask.attrs["mask_source"] = "token_ids_nonzero"
            f_mask.attrs["dtype"] = "uint8"
            if f_query is not None:
                f_query.attrs["query_source"] = "encode_text_pooled"
                f_query.attrs["dtype"] = args.query_dtype

            bs = int(args.batch_size)
            n = len(entries)
            pbar = tqdm(total=n, desc="Encoding text", unit="cap")
            for i in range(0, n, bs):
                chunk = entries[i:i + bs]
                keys = [x[0] for x in chunk]
                texts = [x[1] for x in chunk]
                with torch.no_grad():
                    token_ids = tokenizer(texts).to(device)  # (B,77)
                    pooled, token_feats = _encode_batch_open_clip(model, token_ids)
                    token_mask = (token_ids != 0).to(torch.uint8)  # (B,77)

                pooled_np = pooled.detach().float().cpu().numpy().astype(query_np_dtype, copy=False)
                token_np = token_feats.detach().float().cpu().numpy().astype(token_np_dtype, copy=False)
                mask_np = token_mask.detach().cpu().numpy().astype(np.uint8, copy=False)

                for j, key in enumerate(keys):
                    f_token.create_dataset(key, data=token_np[j], dtype=token_np_dtype)
                    f_mask.create_dataset(key, data=mask_np[j], dtype=np.uint8)
                    if f_query is not None:
                        f_query.create_dataset(key, data=pooled_np[j], dtype=query_np_dtype)

                pbar.update(len(chunk))
                if (i // bs) % 20 == 0:
                    pbar.set_postfix({"written": i + len(chunk)})
            pbar.close()
        finally:
            if f_query is not None:
                f_query.close()

    print("[done] wrote:")
    if write_query:
        print(f"  query_h5: {args.out_query_h5}")
    print(f"  token_h5: {args.out_token_h5}")
    print(f"  mask_h5 : {args.out_mask_h5}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
