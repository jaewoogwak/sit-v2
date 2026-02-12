#!/usr/bin/env python3
"""
Compare two token-level HDF5 files (key -> (77,512)) for the same desc_id keys.

Reports:
  - key overlap stats
  - per-key cosine/max_abs/mean_abs
  - global summary (mean/min/max)
  - top-K most different keys
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import h5py
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Compare two token_lnproj HDF5 files by shared keys.")
    p.add_argument("--h5_a", type=str, required=True, help="First HDF5 path.")
    p.add_argument("--h5_b", type=str, required=True, help="Second HDF5 path.")
    p.add_argument(
        "--desc_id",
        type=str,
        default="",
        help="If set, compare only this key.",
    )
    p.add_argument(
        "--max_keys",
        type=int,
        default=0,
        help="If >0, compare only first N shared keys (sorted).",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=10,
        help="How many most-different keys to print.",
    )
    return p.parse_args()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = float(np.linalg.norm(a))
    bb = float(np.linalg.norm(b))
    if aa == 0.0 or bb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (aa * bb))


def _diff_stats(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    d = np.abs(a - b)
    return float(d.max()), float(d.mean())


def _load_keys(f: h5py.File) -> List[str]:
    return sorted(list(f.keys()))


def main():
    args = parse_args()

    with h5py.File(args.h5_a, "r") as fa, h5py.File(args.h5_b, "r") as fb:
        keys_a = set(_load_keys(fa))
        keys_b = set(_load_keys(fb))
        shared = sorted(keys_a & keys_b)
        only_a = len(keys_a - keys_b)
        only_b = len(keys_b - keys_a)

        print("=== Key Stats ===")
        print(f"h5_a keys: {len(keys_a)}")
        print(f"h5_b keys: {len(keys_b)}")
        print(f"shared   : {len(shared)}")
        print(f"only_a   : {only_a}")
        print(f"only_b   : {only_b}")
        print()

        if args.desc_id:
            if args.desc_id not in shared:
                raise KeyError(f"desc_id not shared: {args.desc_id}")
            keys = [args.desc_id]
        else:
            keys = shared
            if args.max_keys > 0:
                keys = keys[: args.max_keys]

        if len(keys) == 0:
            raise RuntimeError("No keys to compare.")

        cos_list: List[float] = []
        max_abs_list: List[float] = []
        mean_abs_list: List[float] = []
        rows: List[Tuple[str, float, float, float]] = []

        for k in tqdm(keys, desc="Comparing", unit="key"):
            a = np.asarray(fa[k][...], dtype=np.float32)
            b = np.asarray(fb[k][...], dtype=np.float32)

            if a.shape != b.shape:
                raise RuntimeError(f"Shape mismatch at {k}: {a.shape} vs {b.shape}")
            if a.ndim != 2:
                raise RuntimeError(f"Expected 2D token feature at {k}, got {a.shape}")

            cos = _cosine(a.reshape(-1), b.reshape(-1))
            max_abs, mean_abs = _diff_stats(a, b)

            cos_list.append(cos)
            max_abs_list.append(max_abs)
            mean_abs_list.append(mean_abs)
            rows.append((k, cos, max_abs, mean_abs))

        cos_np = np.asarray(cos_list, dtype=np.float64)
        max_np = np.asarray(max_abs_list, dtype=np.float64)
        mean_np = np.asarray(mean_abs_list, dtype=np.float64)

        print("=== Summary ===")
        print(f"compared_keys: {len(keys)}")
        print(f"cosine mean/min/max   : {cos_np.mean():.8f} / {cos_np.min():.8f} / {cos_np.max():.8f}")
        print(f"max_abs mean/min/max  : {max_np.mean():.6e} / {max_np.min():.6e} / {max_np.max():.6e}")
        print(f"mean_abs mean/min/max : {mean_np.mean():.6e} / {mean_np.min():.6e} / {mean_np.max():.6e}")
        print()

        rows_sorted = sorted(rows, key=lambda x: x[1])  # lowest cosine first
        topk = max(1, int(args.topk))
        print(f"=== Top {topk} Most Different (lowest cosine) ===")
        for k, cos, max_abs, mean_abs in rows_sorted[:topk]:
            print(f"{k}\tcos={cos:.8f}\tmax_abs={max_abs:.6e}\tmean_abs={mean_abs:.6e}")

        if args.desc_id:
            k, cos, max_abs, mean_abs = rows[0]
            print()
            print("=== Single Key Result ===")
            print(f"{k}\tcos={cos:.8f}\tmax_abs={max_abs:.6e}\tmean_abs={mean_abs:.6e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
