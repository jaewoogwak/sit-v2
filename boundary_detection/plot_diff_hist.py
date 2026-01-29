#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot fine diffs stats and histogram for a single video_id."
    )
    parser.add_argument(
        "--boundaries_json",
        type=str,
        default="/dev/ssd1/gjw/prvr/semantic-transformer-v2/boundary_detection/output/boundaries_val.json",
    )
    parser.add_argument("--video_id", type=str, required=True)
    parser.add_argument(
        "--level",
        type=str,
        choices=("fine", "coarse"),
        default="fine",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="If empty, saves to output/diff_hist_{video_id}.png next to boundaries_json.",
    )
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument(
        "--show_mad",
        action="store_true",
        help="Overlay median and MAD band on the histogram.",
    )
    return parser.parse_args()


def _default_output_path(boundaries_json, video_id):
    base_dir = os.path.dirname(boundaries_json) or "."
    safe_vid = video_id.replace("/", "_")
    return os.path.join(base_dir, f"diff_hist_{safe_vid}.png")


def _load_entry(boundaries_json, video_id, level):
    with open(boundaries_json, "r") as f:
        data = json.load(f)
    entry = data.get(video_id)
    if entry is None:
        raise KeyError(f"video_id not found: {video_id}")
    if isinstance(entry, dict) and level in entry:
        entry = entry[level]
    diffs = entry.get("diffs", []) if isinstance(entry, dict) else []
    return diffs


def main():
    args = parse_args()
    diffs = _load_entry(args.boundaries_json, args.video_id, args.level)
    if not diffs:
        raise ValueError(f"no diffs found for {args.video_id} ({args.level})")

    diffs_arr = np.asarray(diffs, dtype=np.float32)
    stats = {
        "count": int(diffs_arr.size),
        "mean": float(diffs_arr.mean()),
        "std": float(diffs_arr.std()),
        "min": float(diffs_arr.min()),
        "max": float(diffs_arr.max()),
        "median": float(np.median(diffs_arr)),
    }

    print(f"video_id: {args.video_id}")
    print(f"level: {args.level}")
    print(f"count: {stats['count']}")
    print(f"mean: {stats['mean']:.6f}")
    print(f"std: {stats['std']:.6f}")
    print(f"min: {stats['min']:.6f}")
    print(f"max: {stats['max']:.6f}")
    print(f"median: {stats['median']:.6f}")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    sorted_diffs = np.sort(diffs_arr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(diffs_arr, bins=args.bins, color="#4C78A8", edgecolor="black", alpha=0.85)
    mean_val = stats["mean"]
    std_val = stats["std"]
    axes[0].axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label="mean (std)")
    axes[0].axvspan(
        mean_val - std_val,
        mean_val + std_val,
        color="red",
        alpha=0.15,
        label="mean ± std (std)",
    )
    if args.show_mad:
        median_val = stats["median"]
        mad_val = float(np.median(np.abs(diffs_arr - median_val)))
        axes[0].axvline(
            median_val,
            color="#54A24B",
            linestyle="-.",
            linewidth=1.5,
            label="median (mad)",
        )
        axes[0].axvspan(
            median_val - mad_val,
            median_val + mad_val,
            color="#54A24B",
            alpha=0.15,
            label="median ± mad (mad)",
        )
    axes[0].set_title("Diffs Histogram")
    axes[0].set_xlabel("diff value")
    axes[0].set_ylabel("count")
    axes[0].legend(loc="upper right")

    axes[1].plot(sorted_diffs, color="#F58518", linewidth=1.5)
    axes[1].set_title("Sorted Diffs")
    axes[1].set_xlabel("index (sorted)")
    axes[1].set_ylabel("diff value")

    fig.suptitle(f"{args.video_id} ({args.level})")
    fig.tight_layout()

    output_path = args.output_path or _default_output_path(args.boundaries_json, args.video_id)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
