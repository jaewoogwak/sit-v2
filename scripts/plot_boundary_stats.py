#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter, defaultdict
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def peaks_from_level_entry(level_entry, num_frames):
    if not isinstance(level_entry, dict):
        return []
    edges = level_entry.get("edges") or []
    if edges:
        peaks = []
        for edge in edges[1:-1]:
            try:
                peak = int(edge) - 1
            except Exception:
                continue
            if 0 <= peak < num_frames:
                peaks.append(peak)
        return peaks
    return level_entry.get("peaks", []) or []


def _parse_level_token(token):
    token = (token or "").strip().lower()
    if not token:
        return "", None
    if token == "levels":
        return "levels", None
    if token.startswith("level"):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            return "level", int(digits)
        return "level", None
    return token, None


def count_segments_from_peaks(peaks):
    if not peaks:
        return 1
    return len({int(p) for p in peaks if p is not None}) + 1


def _infer_num_frames(entry):
    if not isinstance(entry, dict):
        return None
    levels = entry.get("levels") or []
    edges_max = None
    for lv in levels:
        if not isinstance(lv, dict):
            continue
        edges = lv.get("edges") or []
        if edges:
            for e in edges:
                try:
                    val = int(e)
                except Exception:
                    continue
                edges_max = val if edges_max is None else max(edges_max, val)
    if edges_max is not None:
        return max(0, edges_max - 1)
    return None


def _bounds_from_peaks(peaks, num_frames):
    clean_peaks = sorted({int(p) for p in peaks if p is not None})
    if num_frames is None:
        num_frames = (max(clean_peaks) + 1) if clean_peaks else 1
    if num_frames <= 0:
        return [(0, 0)]
    if not clean_peaks:
        return [(0, num_frames)]
    bounds = []
    start = 0
    for peak in clean_peaks:
        end = peak + 1
        if end <= start:
            continue
        bounds.append((start, end))
        start = end
    if start < num_frames:
        bounds.append((start, num_frames))
    return bounds or [(0, num_frames)]


def _bounds_from_levels(entry, level_num, num_frames):
    if not isinstance(entry, dict):
        return []
    levels = entry.get("levels") or []
    bounds = []
    for lv in levels:
        if not isinstance(lv, dict):
            continue
        if level_num is not None and lv.get("level") != level_num:
            continue
        peaks = peaks_from_level_entry(lv, num_frames or 10**9)
        bounds.extend(_bounds_from_peaks(peaks, num_frames))
    return bounds


def _get_peaks_by_token(entry, token_name):
    if token_name in ("fine", "coarse"):
        if isinstance(entry, dict) and ("fine" in entry or "coarse" in entry):
            return entry.get(token_name, {}).get("peaks", []) or entry.get("peaks", []) if token_name == "fine" else entry.get(token_name, {}).get("peaks", [])
        if isinstance(entry, dict) and token_name == "fine":
            return entry.get("peaks", []) or []
        if isinstance(entry, list) and token_name == "fine":
            return entry
    return []


def compute_stats(path, boundary_level="fine+levels", dedupe=False):
    with open(path, "r") as f:
        data = json.load(f)

    seg_dist = defaultdict(Counter)
    max_level_dist = Counter()
    seg_counts_by_level = defaultdict(list)

    for _, entry in data.items():
        if isinstance(entry, dict):
            if "fine" in entry or "coarse" in entry:
                fine_peaks = entry.get("fine", {}).get("peaks", []) or entry.get("peaks", []) or []
            else:
                fine_peaks = entry.get("peaks", []) or []
            fine_count = count_segments_from_peaks(fine_peaks)
            seg_dist[0][fine_count] += 1
            seg_counts_by_level[0].append(fine_count)

            levels = entry.get("levels") or []
            level_nums = [
                lv.get("level")
                for lv in levels
                if isinstance(lv, dict) and isinstance(lv.get("level"), (int, float))
            ]
            if level_nums:
                max_level = int(max(level_nums))
                max_level_dist[max_level] += 1
                for lv in levels:
                    if not isinstance(lv, dict) or not isinstance(lv.get("level"), (int, float)):
                        continue
                    level_num = int(lv.get("level"))
                    peaks = peaks_from_level_entry(lv, num_frames=10**9)
                    seg_count = count_segments_from_peaks(peaks)
                    seg_dist[level_num][seg_count] += 1
                    seg_counts_by_level[level_num].append(seg_count)
            else:
                max_level_dist[0] += 1
        else:
            fine_peaks = entry if isinstance(entry, list) else []
            fine_count = count_segments_from_peaks(fine_peaks)
            seg_dist[0][fine_count] += 1
            seg_counts_by_level[0].append(fine_count)
            max_level_dist[0] += 1

        if dedupe:
            num_frames = _infer_num_frames(entry)
            bounds = []
            if boundary_level == "both":
                bounds.extend(_bounds_from_peaks(_get_peaks_by_token(entry, "fine"), num_frames))
                bounds.extend(_bounds_from_peaks(_get_peaks_by_token(entry, "coarse"), num_frames))
            elif "+" in boundary_level:
                tokens = [t.strip() for t in boundary_level.split("+") if t.strip()]
                for token in tokens:
                    token_name, level_num = _parse_level_token(token)
                    if token_name in ("fine", "coarse"):
                        peaks = _get_peaks_by_token(entry, token_name)
                        bounds.extend(_bounds_from_peaks(peaks, num_frames))
                    elif token_name in ("levels", "level"):
                        bounds.extend(_bounds_from_levels(entry, level_num, num_frames))
            else:
                token_name, level_num = _parse_level_token(boundary_level)
                if token_name in ("levels", "level"):
                    bounds.extend(_bounds_from_levels(entry, level_num, num_frames))
                else:
                    peaks = _get_peaks_by_token(entry, token_name)
                    bounds.extend(_bounds_from_peaks(peaks, num_frames))

            raw_count = len([b for b in bounds if b[1] > b[0]])
            seg_dist[-2][raw_count if raw_count > 0 else 1] += 1
            seg_counts_by_level[-2].append(raw_count if raw_count > 0 else 1)

            uniq_bounds = sorted({(s, e) for s, e in bounds if e > s})
            deduped_count = len(uniq_bounds) if uniq_bounds else 1
            seg_dist[-1][deduped_count] += 1
            seg_counts_by_level[-1].append(deduped_count)

    return len(data), seg_dist, max_level_dist, seg_counts_by_level


def plot_max_levels(ax, max_level_dist, title):
    levels = sorted(max_level_dist.keys())
    counts = [max_level_dist[lvl] for lvl in levels]
    labels = [f"level{lvl}" for lvl in levels]
    ax.bar(labels, counts, color="#4E79A7")
    ax.set_title(title)
    ax.set_ylabel("videos")
    ax.set_xlabel("max level (fine=level0)")


def plot_segment_counts(ax, seg_dist, title, logy=False):
    ax.set_title(title)
    ax.set_xlabel("segments per video")
    ax.set_ylabel("videos")
    for lvl in sorted(seg_dist.keys()):
        if lvl == -2:
            label = "combined"
        elif lvl == -1:
            label = "deduped"
        else:
            label = "fine" if lvl == 0 else f"level-{lvl}"
        items = sorted(seg_dist[lvl].items())
        xs = [k for k, _ in items]
        ys = [v for _, v in items]
        line, = ax.plot(xs, ys, marker="o", linewidth=1, markersize=3, label=label)
        total = sum(ys)
        if total > 0:
            mean = sum(x * y for x, y in items) / total
            ax.axvline(mean, color=line.get_color(), linestyle="--", linewidth=1, alpha=0.7)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}" if y >= 1 else f"{y:g}"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.legend()


def plot_segment_histograms(fig, axes, seg_dist, title):
    levels = sorted(seg_dist.keys())
    for ax, lvl in zip(axes, levels):
        if lvl == -2:
            label = "combined"
        elif lvl == -1:
            label = "deduped"
        else:
            label = "fine" if lvl == 0 else f"level-{lvl}"
        items = sorted(seg_dist[lvl].items())
        xs = [k for k, _ in items]
        ys = [v for _, v in items]
        ax.bar(xs, ys, width=0.8, color="#4E79A7")
        ax.set_ylabel(f"{label}")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], fontsize=8, rotation=0)
        ax.tick_params(axis="x", labelbottom=True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
        ax.set_xlabel("segments per video")
    fig.suptitle(title)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="Plot boundary segment statistics.")
    parser.add_argument(
        "--dataset",
        choices=("act_frames", "tvr_frames"),
        default=None,
        help="Dataset config name (maps to boundaries_{act,tvr}_{train,val}.json).",
    )
    parser.add_argument(
        "--boundaries_dir",
        default=os.path.join(repo_root, "boundary_detection", "output"),
        help="Directory containing boundaries_{train,val}.json",
    )
    parser.add_argument(
        "--train",
        default=None,
        help="Path to boundaries_train.json",
    )
    parser.add_argument(
        "--val",
        default=None,
        help="Path to boundaries_val.json",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(repo_root, "boundary_detection", "plots"),
        help="Directory to write plots",
    )
    parser.add_argument("--logy", action="store_true", help="Use log scale for y-axis.")
    parser.add_argument("--hist", action="store_true", help="Also save histogram per level.")
    parser.add_argument(
        "--boundary_level",
        default="fine+levels",
        help="Boundary level definition used for dedupe stats.",
    )
    parser.add_argument("--dedupe", action="store_true", help="Include deduped segment stats.")
    args = parser.parse_args()

    if args.dataset:
        dataset_key = "act" if args.dataset == "act_frames" else "tvr"
        if args.train is None:
            args.train = os.path.join(
                args.boundaries_dir, f"boundaries_{dataset_key}_train.json"
            )
        if args.val is None:
            args.val = os.path.join(
                args.boundaries_dir, f"boundaries_{dataset_key}_val.json"
            )
    else:
        if args.train is None:
            args.train = os.path.join(
                repo_root, "boundary_detection", "output", "boundaries_train.json"
            )
        if args.val is None:
            args.val = os.path.join(
                repo_root, "boundary_detection", "output", "boundaries_val.json"
            )

    os.makedirs(args.out_dir, exist_ok=True)

    for split_name, path in [("train", args.train), ("val", args.val)]:
        total, seg_dist, max_level_dist, seg_counts_by_level = compute_stats(
            path,
            boundary_level=args.boundary_level,
            dedupe=args.dedupe,
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        plot_max_levels(ax, max_level_dist, f"{split_name}: max level distribution (n={total})")
        fig.tight_layout()
        max_level_path = os.path.join(args.out_dir, f"{split_name}_max_level.png")
        fig.savefig(max_level_path, dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        plot_segment_counts(
            ax,
            seg_dist,
            f"{split_name}: segment count distribution (fine=level0)",
            logy=args.logy,
        )
        fig.tight_layout()
        segment_path = os.path.join(args.out_dir, f"{split_name}_segment_counts.png")
        fig.savefig(segment_path, dpi=200)
        plt.close(fig)
        print(f"[{split_name}] saved: {max_level_path}")
        print(f"[{split_name}] saved: {segment_path}")
        print(f"[{split_name}] segment count stats:")
        for lvl in sorted(seg_counts_by_level.keys()):
            counts = seg_counts_by_level[lvl]
            if not counts:
                continue
            mean = sum(counts) / len(counts)
            var = sum((c - mean) ** 2 for c in counts) / len(counts)
            std = math.sqrt(var)
            if lvl == -2:
                label = "combined"
            elif lvl == -1:
                label = "deduped"
            else:
                label = "fine" if lvl == 0 else f"level-{lvl}"
            print(
                f"  {label}: mean={mean:.2f}, std={std:.2f}, "
                f"min={min(counts)}, max={max(counts)}, n={len(counts)}"
            )

        if args.hist:
            levels = sorted(seg_dist.keys())
            fig, axes = plt.subplots(
                nrows=len(levels),
                ncols=1,
                figsize=(9, max(2.2 * len(levels), 3)),
                sharex=True,
            )
            if len(levels) == 1:
                axes = [axes]
            plot_segment_histograms(
                fig,
                axes,
                seg_dist,
                f"{split_name}: segment count histograms (fine=level0)",
            )
            fig.tight_layout(rect=[0, 0.02, 1, 0.95])
            hist_path = os.path.join(args.out_dir, f"{split_name}_segment_hist.png")
            fig.savefig(hist_path, dpi=200)
            plt.close(fig)
            print(f"[{split_name}] saved: {hist_path}")


if __name__ == "__main__":
    main()
