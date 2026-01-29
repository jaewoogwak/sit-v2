import argparse
import json
import os

import h5py
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="Build HDF5 from per-video npy frame embeddings.")
    ap.add_argument("--jsonl_paths", type=str, required=True,
                    help="Comma-separated JSONL paths (used to collect video ids).")
    ap.add_argument("--npy_dir", type=str, required=True,
                    help="Directory containing <video_id>.npy files.")
    ap.add_argument("--h5_out", type=str, required=True,
                    help="Output HDF5 path.")
    ap.add_argument("--tvr_style", action="store_true",
                    help="Enable TVR-style filename resolution (<show>_frames__<video_id>.npy).")
    ap.add_argument("--tvr_prefixes", type=str,
                    default="bbt,castle,friends,grey,house,met",
                    help="Comma-separated TVR show prefixes.")
    ap.add_argument("--tvr_default", type=str, default="bbt",
                    help="Default TVR show prefix when video_id has no show prefix.")
    ap.add_argument("--allow_missing", action="store_true",
                    help="Skip missing npy files instead of failing.")
    ap.add_argument("--dtype", type=str, default="float32",
                    help="Output dtype (default: float32).")
    return ap.parse_args()


def _get_vid(entry):
    for key in ("vid_name", "video_id", "vid", "video", "video_name"):
        if key in entry:
            return entry[key]
    return None


def _iter_video_ids(jsonl_paths):
    seen = set()
    for path in jsonl_paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                vid = _get_vid(entry)
                if vid is None or vid in seen:
                    continue
                seen.add(vid)
                yield vid


def _resolve_npy_path(video_id, npy_dir, tvr_style=False, tvr_prefixes=None, tvr_default="bbt"):
    if not tvr_style:
        return os.path.join(npy_dir, f"{video_id}.npy")

    prefixes = [p for p in (tvr_prefixes or []) if p]
    show = None
    base = video_id
    for prefix in prefixes:
        if video_id.startswith(prefix + "_"):
            show = prefix
            base = video_id[len(prefix) + 1:]
            break

    candidates = []
    if show is None:
        candidates.append(f"{tvr_default}_frames__{video_id}.npy")
        for prefix in prefixes:
            candidates.append(f"{prefix}_frames__{video_id}.npy")
    else:
        candidates.append(f"{show}_frames__{base}.npy")
        candidates.append(f"{show}_frames__{video_id}.npy")

    for name in candidates:
        path = os.path.join(npy_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(npy_dir, f"{video_id}.npy")


def main():
    args = parse_args()
    jsonl_paths = [p for p in args.jsonl_paths.split(",") if p]
    os.makedirs(os.path.dirname(args.h5_out) or ".", exist_ok=True)
    tvr_prefixes = [p.strip() for p in args.tvr_prefixes.split(",") if p.strip()]

    total = 0
    written = 0
    missing = 0

    with h5py.File(args.h5_out, "w") as h5f:
        for vid in _iter_video_ids(jsonl_paths):
            total += 1
            npy_path = _resolve_npy_path(
                vid,
                args.npy_dir,
                tvr_style=args.tvr_style,
                tvr_prefixes=tvr_prefixes,
                tvr_default=args.tvr_default,
            )
            if not os.path.exists(npy_path):
                missing += 1
                if not args.allow_missing:
                    raise FileNotFoundError(f"Missing npy: {npy_path}")
                continue
            feats = np.load(npy_path, mmap_mode="r")
            feats = np.asarray(feats, dtype=args.dtype)
            h5f.create_dataset(vid, data=feats, compression="gzip", compression_opts=4)
            written += 1

    print(f"[build_h5_from_npy] total_vids={total} written={written} missing={missing}")


if __name__ == "__main__":
    main()
