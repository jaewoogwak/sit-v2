#!/usr/bin/env python3
"""
Repack text-feature HDF5 by moving datasets under a source group (e.g., /feat)
to root-level keys expected by existing loaders.

Example:
  input : /feat/v_xxx#enc#0  (77, 512)
  output: /v_xxx#enc#0       (77, 512)
"""

import argparse
import os
import sys

import h5py


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repack HDF5 text feature keys from a group to root-level keys."
    )
    parser.add_argument(
        "--src_h5",
        type=str,
        required=True,
        help="Source HDF5 path (e.g., token_lnproj file).",
    )
    parser.add_argument(
        "--dst_h5",
        type=str,
        required=True,
        help="Destination HDF5 path to create.",
    )
    parser.add_argument(
        "--src_group",
        type=str,
        default="feat",
        help="Group name in src_h5 to read datasets from (default: feat).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination file if it exists.",
    )
    return parser.parse_args()


def ensure_parent_group(h5file, key_path):
    parent = os.path.dirname(key_path)
    if parent and parent != "/":
        h5file.require_group(parent)


def count_datasets(group):
    total = 0

    def visitor(_name, obj):
        nonlocal total
        if isinstance(obj, h5py.Dataset):
            total += 1

    group.visititems(visitor)
    return total


def main():
    args = parse_args()
    src_h5 = args.src_h5
    dst_h5 = args.dst_h5
    src_group_name = args.src_group.strip("/")

    if not os.path.exists(src_h5):
        raise FileNotFoundError(f"Source file not found: {src_h5}")

    if os.path.exists(dst_h5):
        if args.overwrite:
            os.remove(dst_h5)
        else:
            raise FileExistsError(
                f"Destination exists: {dst_h5}. Use --overwrite to replace it."
            )

    copied = 0
    total = 0
    next_report = 1
    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        if src_group_name not in src:
            raise KeyError(f"Group '{src_group_name}' not found in {src_h5}")
        src_group = src[src_group_name]
        total = count_datasets(src_group)
        print(f"[repack] total_datasets: {total}")

        # Preserve root attributes for provenance.
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        def visitor(name, obj):
            nonlocal copied, next_report
            if not isinstance(obj, h5py.Dataset):
                return

            # Convert /feat/<name> -> /<name>
            out_key = name
            ensure_parent_group(dst, out_key)
            src_group.copy(obj, dst, name=out_key)
            copied += 1

            if total > 0:
                pct = (copied * 100.0) / float(total)
                if copied >= next_report or copied == total:
                    print(f"[repack] progress: {copied}/{total} ({pct:.1f}%)")
                    step = max(1, total // 20)  # about 5% increments
                    next_report = min(total, copied + step)

        src_group.visititems(visitor)

    print(f"[repack] src: {src_h5}")
    print(f"[repack] dst: {dst_h5}")
    print(f"[repack] src_group: /{src_group_name}")
    print(f"[repack] copied_datasets: {copied}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[repack][error] {exc}", file=sys.stderr)
        sys.exit(1)
