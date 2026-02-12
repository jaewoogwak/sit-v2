#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_boundaries(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _count_segments_from_peaks(peaks: list) -> int:
    return len(peaks) + 1


def _count_segments_from_level(level: dict) -> int:
    edges = level.get("edges") or []
    if edges:
        return max(0, len(edges) - 1)
    peaks = level.get("peaks") or []
    if peaks:
        return _count_segments_from_peaks(peaks)
    return 0


def segment_counts(data: dict, mode: str) -> list:
    counts = []
    for _, info in data.items():
        fine_peaks = info.get("fine", {}).get("peaks", []) or []
        coarse_peaks = info.get("coarse", {}).get("peaks", []) or []
        levels = info.get("levels") or []
        level_segments = sum(_count_segments_from_level(level) for level in levels)
        if mode == "fine":
            counts.append(_count_segments_from_peaks(fine_peaks))
        elif mode == "fine+coarse":
            counts.append(_count_segments_from_peaks(fine_peaks) + _count_segments_from_peaks(coarse_peaks))
        elif mode == "fine+levels":
            counts.append(_count_segments_from_peaks(fine_peaks) + level_segments)
        elif mode == "fine+coarse+levels":
            counts.append(
                _count_segments_from_peaks(fine_peaks)
                + _count_segments_from_peaks(coarse_peaks)
                + level_segments
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return counts


def plot_hist(ax, counts, title, color):
    if not counts:
        ax.set_title(f"{title} (empty)")
        return
    max_count = max(counts)
    bins = list(range(1, max_count + 2))
    ax.hist(counts, bins=bins, rwidth=0.9, align="left", color=color, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Segments per video")
    ax.set_ylabel("Count")
    # Avoid overcrowded x ticks when the range is large.
    tick_step = max(1, (max_count + 9) // 10)
    ax.set_xticks(range(1, max_count + 1, tick_step))
    ax.tick_params(axis="x", labelrotation=0)


def main():
    parser = argparse.ArgumentParser(
        description="Plot segment-count histograms from boundary peaks."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="act_frames",
        choices=("act_frames", "tvr_frames", "cha_frames"),
        help="Dataset config name (maps to boundaries_{act,tvr,cha}_{train,val}.json)",
    )
    parser.add_argument(
        "--boundaries-dir",
        type=Path,
        default=Path("/dev/ssd1/gjw/prvr/semantic-transformer-v2/boundary_detection/output"),
        help="Directory containing boundaries_{train,val}.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path",
    )
    parser.add_argument(
        "--include-final-pooled",
        action="store_true",
        help="Add +1 per video to counts (e.g., include a final pooled/global segment).",
    )
    args = parser.parse_args()

    dataset_key_map = {
        "act_frames": "act",
        "tvr_frames": "tvr",
        "cha_frames": "cha",
    }
    dataset_key = dataset_key_map[args.dataset]
    train_path = args.boundaries_dir / f"boundaries_{dataset_key}_train.json"
    val_path = args.boundaries_dir / f"boundaries_{dataset_key}_val.json"

    if args.output is None:
        args.output = (
            args.boundaries_dir
            / f"segment_count_hist_{dataset_key}_train_val.png"
        )

    train_data = load_boundaries(train_path)
    val_data = load_boundaries(val_path)

    train_fine = segment_counts(train_data, "fine")
    train_combo = segment_counts(train_data, "fine+coarse")
    val_fine = segment_counts(val_data, "fine")
    val_combo = segment_counts(val_data, "fine+coarse")
    has_levels = any(info.get("levels") for info in train_data.values()) or any(
        info.get("levels") for info in val_data.values()
    )
    if has_levels:
        train_levels = segment_counts(train_data, "fine+levels")
        train_combo_levels = segment_counts(train_data, "fine+coarse+levels")
        val_levels = segment_counts(val_data, "fine+levels")
        val_combo_levels = segment_counts(val_data, "fine+coarse+levels")

    if args.include_final_pooled:
        train_fine = [c + 1 for c in train_fine]
        train_combo = [c + 1 for c in train_combo]
        val_fine = [c + 1 for c in val_fine]
        val_combo = [c + 1 for c in val_combo]
        if has_levels:
            train_levels = [c + 1 for c in train_levels]
            train_combo_levels = [c + 1 for c in train_combo_levels]
            val_levels = [c + 1 for c in val_levels]
            val_combo_levels = [c + 1 for c in val_combo_levels]

    if has_levels:
        fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
        plot_hist(
            axes[0, 0],
            train_fine,
            f"Train (n={len(train_fine)}) - Fine only",
            color="#4C78A8",
        )
        plot_hist(
            axes[0, 1],
            train_combo,
            f"Train (n={len(train_combo)}) - Fine + Coarse",
            color="#F58518",
        )
        plot_hist(
            axes[0, 2],
            train_levels,
            f"Train (n={len(train_levels)}) - Fine + Levels",
            color="#72B7B2",
        )
        plot_hist(
            axes[1, 0],
            val_fine,
            f"Val (n={len(val_fine)}) - Fine only",
            color="#54A24B",
        )
        plot_hist(
            axes[1, 1],
            val_combo,
            f"Val (n={len(val_combo)}) - Fine + Coarse",
            color="#E45756",
        )
        plot_hist(
            axes[1, 2],
            val_levels,
            f"Val (n={len(val_levels)}) - Fine + Levels",
            color="#B279A2",
        )
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
        plot_hist(
            axes[0, 0],
            train_fine,
            f"Train (n={len(train_fine)}) - Fine only",
            color="#4C78A8",
        )
        plot_hist(
            axes[0, 1],
            train_combo,
            f"Train (n={len(train_combo)}) - Fine + Coarse",
            color="#F58518",
        )
        plot_hist(
            axes[1, 0],
            val_fine,
            f"Val (n={len(val_fine)}) - Fine only",
            color="#54A24B",
        )
        plot_hist(
            axes[1, 1],
            val_combo,
            f"Val (n={len(val_combo)}) - Fine + Coarse",
            color="#E45756",
        )

    if args.include_final_pooled:
        fig.suptitle("Segment counts (+1 final pooled)", fontsize=14)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)

    print(f"Saved histogram to: {args.output}")
    print(
        "Counts summary: "
        f"train_fine={len(train_fine)}, train_combo={len(train_combo)}, "
        f"val_fine={len(val_fine)}, val_combo={len(val_combo)}"
    )
    print("Mean/std of segments per video:")
    print(
        f"  train fine: {statistics.mean(train_fine):.2f} / {statistics.pstdev(train_fine):.2f}"
    )
    print(
        f"  train fine+coarse: {statistics.mean(train_combo):.2f} / {statistics.pstdev(train_combo):.2f}"
    )
    print(f"  val fine: {statistics.mean(val_fine):.2f} / {statistics.pstdev(val_fine):.2f}")
    print(
        f"  val fine+coarse: {statistics.mean(val_combo):.2f} / {statistics.pstdev(val_combo):.2f}"
    )
    if has_levels:
        print(
            f"  train fine+levels: {statistics.mean(train_levels):.2f} / {statistics.pstdev(train_levels):.2f}"
        )
        print(
            f"  train fine+coarse+levels: {statistics.mean(train_combo_levels):.2f} / {statistics.pstdev(train_combo_levels):.2f}"
        )
        print(
            f"  val fine+levels: {statistics.mean(val_levels):.2f} / {statistics.pstdev(val_levels):.2f}"
        )
        print(
            f"  val fine+coarse+levels: {statistics.mean(val_combo_levels):.2f} / {statistics.pstdev(val_combo_levels):.2f}"
        )
    print("Min/max of segments per video:")
    print(f"  train fine: {min(train_fine)} / {max(train_fine)}")
    print(f"  train fine+coarse: {min(train_combo)} / {max(train_combo)}")
    print(f"  val fine: {min(val_fine)} / {max(val_fine)}")
    print(f"  val fine+coarse: {min(val_combo)} / {max(val_combo)}")
    if has_levels:
        print(f"  train fine+levels: {min(train_levels)} / {max(train_levels)}")
        print(f"  train fine+coarse+levels: {min(train_combo_levels)} / {max(train_combo_levels)}")
        print(f"  val fine+levels: {min(val_levels)} / {max(val_levels)}")
        print(f"  val fine+coarse+levels: {min(val_combo_levels)} / {max(val_combo_levels)}")


if __name__ == "__main__":
    main()
