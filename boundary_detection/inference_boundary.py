import argparse
import json
import os

import h5py
import numpy as np
import torch

from dataset import TVRDataset
from model import GlobalPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Boundary detection with temporal feature difference.")
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument(
        "--output_format",
        type=str,
        choices=("json", "jsonl", "auto"),
        default="auto",
        help="auto: infer from output_json suffix (.jsonl => jsonl, else json).",
    )
    parser.add_argument(
        "--segment_h5_out",
        type=str,
        default="",
        help="If set, writes per-video segment embeddings (pooled from fine latents) to this H5.",
    )
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--threshold_std", type=float, default=1.0)
    parser.add_argument(
        "--segment_pooling",
        type=str,
        choices=("mean", "self_attn"),
        default="mean",
        help="How to pool fine latents within a segment before coarse processing.",
    )
    parser.add_argument(
        "--recursive_levels",
        type=int,
        default=0,
        help="Number of recursive segment-level passes to run after fine detection.",
    )
    parser.add_argument(
        "--recursive_seq_len",
        type=int,
        default=0,
        help="Window length for recursive segment encoding (0 => use seq_len).",
    )
    parser.add_argument(
        "--recursive_seq_len_list",
        type=str,
        default="",
        help="Comma-separated seq_len per recursive level (overrides recursive_seq_len when provided).",
    )
    parser.add_argument(
        "--recursive_alpha",
        type=int,
        default=0,
        help="Peak suppression window for recursive levels (0 => use alpha).",
    )
    parser.add_argument(
        "--recursive_alpha_list",
        type=str,
        default="",
        help="Comma-separated alpha per recursive level (overrides recursive_alpha when provided).",
    )
    parser.add_argument(
        "--recursive_a",
        type=float,
        default=2.0,
        help="MAD threshold multiplier for recursive levels (median + a * mad).",
    )
    parser.add_argument(
        "--recursive_a_list",
        type=str,
        default="",
        help="Comma-separated a per recursive level (overrides recursive_a when provided).",
    )
    parser.add_argument(
        "--recursive_checkpoint",
        type=str,
        default="",
        help="If set, uses this checkpoint for recursive segment-level passes.",
    )
    recursive_local_max_group = parser.add_mutually_exclusive_group()
    recursive_local_max_group.add_argument(
        "--recursive_use_local_max", dest="recursive_use_local_max", action="store_true"
    )
    recursive_local_max_group.add_argument(
        "--recursive_no_local_max", dest="recursive_use_local_max", action="store_false"
    )
    parser.set_defaults(recursive_use_local_max=True)
    parser.add_argument(
        "--recursive_until_one",
        action="store_true",
        help="If set, keep recursing until a single segment remains (or no peaks).",
    )
    parser.add_argument(
        "--coarse_checkpoint",
        type=str,
        default="",
        help="If set, runs a second boundary detection pass over pooled segment embeddings.",
    )
    parser.add_argument("--coarse_seq_len", type=int, default=3)
    parser.add_argument("--coarse_alpha", type=int, default=3)
    parser.add_argument("--coarse_threshold_std", type=float, default=1.0)
    parser.add_argument(
        "--coarse_threshold_mode",
        type=str,
        choices=("mean_std", "mad"),
        default="mean_std",
        help="Thresholding mode for coarse peak detection.",
    )
    parser.add_argument(
        "--coarse_mode",
        type=str,
        choices=("peaks", "merge"),
        default="peaks",
        help="peaks: detect coarse boundaries via temporal difference; merge: group adjacent segments by similarity.",
    )
    parser.add_argument(
        "--coarse_sim_threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for coarse merge mode (higher means stricter).",
    )
    parser.add_argument("--coarse_num_layers", type=int, default=2)
    parser.add_argument("--coarse_num_heads", type=int, default=8)
    parser.add_argument("--coarse_ff_dim", type=int, default=1024)
    parser.add_argument("--coarse_dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    local_max_group = parser.add_mutually_exclusive_group()
    local_max_group.add_argument("--use_local_max", dest="use_local_max", action="store_true")
    local_max_group.add_argument("--no_local_max", dest="use_local_max", action="store_false")
    parser.set_defaults(use_local_max=True)
    coarse_local_max_group = parser.add_mutually_exclusive_group()
    coarse_local_max_group.add_argument(
        "--coarse_use_local_max", dest="coarse_use_local_max", action="store_true"
    )
    coarse_local_max_group.add_argument(
        "--coarse_no_local_max", dest="coarse_use_local_max", action="store_false"
    )
    parser.set_defaults(coarse_use_local_max=True)
    return parser.parse_args()


def build_windows(feats, seq_len):
    feat_dim = feats.shape[1]
    pad = torch.zeros(seq_len - 1, feat_dim, dtype=feats.dtype)
    padded = torch.cat([pad, feats], dim=0)
    windows = padded.unfold(0, seq_len, 1)
    if windows.shape[1] == feat_dim and windows.shape[2] == seq_len:
        windows = windows.permute(0, 2, 1)
    return windows


def temporal_difference(latent, window):
    length = latent.shape[0]
    diffs = np.zeros(length, dtype=np.float32)
    for i in range(length):
        past_start = max(0, i - window + 1)
        past = latent[past_start : i + 1]
        future_end = min(length, i + window + 1)
        future = latent[i + 1 : future_end]
        if past.size == 0 or future.size == 0:
            continue
        past_mean = past.mean(axis=0)
        future_mean = future.mean(axis=0)
        diffs[i] = float(np.sum((past_mean - future_mean) ** 2))
    return diffs


def detect_peaks(scores, alpha, threshold_std, use_local_max=True, threshold_mode="mean_std"):
    if len(scores) == 0:
        return []
    if threshold_mode == "mad":
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        thresh = median + threshold_std * mad
    else:
        mean = float(scores.mean())
        std = float(scores.std())
        thresh = mean + threshold_std * std
    peaks = []
    for i in range(len(scores)):
        if scores[i] < thresh:
            continue
        if use_local_max:
            left = max(0, i - alpha)
            right = min(len(scores), i + alpha + 1)
            window = scores[left:right]
            if scores[i] == window.max():
                peaks.append(i)
        else:
            peaks.append(i)

    if not peaks:
        return []
    peaks = sorted(peaks, key=lambda i: scores[i], reverse=True)
    selected = []
    for p in peaks:
        if all(abs(p - s) > alpha for s in selected):
            selected.append(p)
    return sorted(selected)


def _infer_output_format(output_path, output_format):
    if output_format != "auto":
        return output_format
    if output_path.lower().endswith(".jsonl"):
        return "jsonl"
    return "json"


def _parse_level_list(raw, cast_fn):
    raw = (raw or "").strip()
    if not raw:
        return []
    items = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            items.append(cast_fn(part))
        except Exception:
            continue
    return items


def _unique_sorted_ints(values):
    out = []
    seen = set()
    for v in values:
        try:
            iv = int(v)
        except Exception:
            continue
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    out.sort()
    return out


def _segment_edges_from_peaks(peaks, length):
    if length <= 0:
        return [0]
    peaks = _unique_sorted_ints(peaks)
    edges = [0]
    for p in peaks:
        edge = p + 1
        if edge <= 0 or edge >= length:
            continue
        if edge != edges[-1]:
            edges.append(edge)
    if edges[-1] != length:
        edges.append(length)
    return edges


def _pool_segment_self_attn(segment_feats):
    if segment_feats.size == 0:
        return np.zeros((segment_feats.shape[1],), dtype=np.float32)
    feat_dim = segment_feats.shape[1]
    scale = np.sqrt(float(feat_dim))
    scores = (segment_feats @ segment_feats.T) / max(scale, 1e-6)
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    attended = weights @ segment_feats
    return attended.mean(axis=0)


def _pool_segments(latent_feats, edges, mode="mean"):
    if latent_feats.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    feat_dim = latent_feats.shape[1]
    seg_embeds = np.zeros((max(0, len(edges) - 1), feat_dim), dtype=np.float32)
    for i in range(len(edges) - 1):
        start = edges[i]
        end = edges[i + 1]
        if start >= end:
            continue
        segment_feats = latent_feats[start:end]
        if mode == "self_attn":
            seg_embeds[i] = _pool_segment_self_attn(segment_feats)
        else:
            seg_embeds[i] = segment_feats.mean(axis=0)
    return seg_embeds


def _coarse_peaks_to_frame_peaks(peaks_segments, segment_edges):
    frame_peaks = []
    for s in _unique_sorted_ints(peaks_segments):
        edge_idx = s + 1
        if edge_idx <= 0 or edge_idx >= len(segment_edges):
            continue
        edge = int(segment_edges[edge_idx])
        frame_peak = edge - 1
        if frame_peak >= 0:
            frame_peaks.append(frame_peak)
    return _unique_sorted_ints(frame_peaks)


def _segment_edges_from_segment_peaks(peaks_segments, segment_edges, length):
    if length <= 0:
        return [0]
    peaks_segments = _unique_sorted_ints(peaks_segments)
    edges = [0]
    for s in peaks_segments:
        edge_idx = s + 1
        if edge_idx <= 0 or edge_idx >= len(segment_edges):
            continue
        edge = int(segment_edges[edge_idx])
        if edge <= 0 or edge >= length:
            continue
        if edge != edges[-1]:
            edges.append(edge)
    if edges[-1] != length:
        edges.append(length)
    return edges


def _adjacent_cosine_similarity(embeds):
    if embeds.size == 0 or embeds.shape[0] < 2:
        return []
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normalized = embeds / norms
    sims = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return sims.tolist()


def _encode_segment_last_latent(feats, start, end, model, device):
    if start >= end:
        return np.zeros((feats.shape[1],), dtype=np.float32)
    segment = feats[start:end]
    if segment.numel() == 0:
        return np.zeros((feats.shape[1],), dtype=np.float32)
    if segment.dim() == 1:
        segment = segment.unsqueeze(0)
    with torch.no_grad():
        batch = segment.unsqueeze(0).to(device)
        latent, _ = model(batch)
    return latent[0, -1, :].cpu().numpy()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = GlobalPredictor(
        feature_dim=args.feature_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    coarse_model = None
    recursive_model = None
    if args.coarse_checkpoint:
        coarse_model = GlobalPredictor(
            feature_dim=args.feature_dim,
            num_layers=args.coarse_num_layers,
            num_heads=args.coarse_num_heads,
            ff_dim=args.coarse_ff_dim,
            dropout=args.coarse_dropout,
        ).to(device)
        coarse_ckpt = torch.load(args.coarse_checkpoint, map_location=device)
        coarse_model.load_state_dict(coarse_ckpt["model_state"])
        coarse_model.eval()

    if args.recursive_checkpoint:
        recursive_model = GlobalPredictor(
            feature_dim=args.feature_dim,
            num_layers=args.coarse_num_layers,
            num_heads=args.coarse_num_heads,
            ff_dim=args.coarse_ff_dim,
            dropout=args.coarse_dropout,
        ).to(device)
        recursive_ckpt = torch.load(args.recursive_checkpoint, map_location=device)
        recursive_model.load_state_dict(recursive_ckpt["model_state"])
        recursive_model.eval()
    elif coarse_model is not None:
        recursive_model = coarse_model
    else:
        recursive_model = model

    dataset = TVRDataset(
        jsonl_path=args.jsonl_path,
        h5_path=args.h5_path,
        seq_len=args.seq_len,
        split=args.split,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )
    video_ids = dataset.get_video_ids(unique=True)

    results = {}
    segment_h5 = None
    if args.segment_h5_out:
        os.makedirs(os.path.dirname(args.segment_h5_out) or ".", exist_ok=True)
        segment_h5 = h5py.File(args.segment_h5_out, "w")
    with h5py.File(args.h5_path, "r") as h5f:
        for vid in video_ids:
            empty_coarse = {"peaks_segments": [], "peaks": [], "diffs": []}
            if vid not in h5f:
                results[vid] = {
                    "fine": {"peaks": [], "diffs": []},
                    "coarse": empty_coarse,
                }
                continue
            feats = torch.from_numpy(np.asarray(h5f[vid])).float()
            if feats.numel() == 0:
                results[vid] = {
                    "fine": {"peaks": [], "diffs": []},
                    "coarse": empty_coarse,
                }
                continue
            windows = build_windows(feats, args.seq_len)
            latent_list = []
            with torch.no_grad():
                for i in range(0, windows.shape[0], args.batch_size):
                    batch = windows[i : i + args.batch_size].to(device)
                    latent, _ = model(batch)
                    latent_list.append(latent[:, -1, :].cpu())
            latent_feats = torch.cat(latent_list, dim=0).numpy()

            diffs = temporal_difference(latent_feats, args.seq_len)
            peaks = detect_peaks(diffs, args.alpha, args.threshold_std, use_local_max=args.use_local_max)
            segment_edges = _segment_edges_from_peaks(peaks, latent_feats.shape[0])
            segment_embeds = _pool_segments(latent_feats, segment_edges, mode=args.segment_pooling)

            entry = {
                "fine": {"peaks": [int(p) for p in peaks], "diffs": [float(v) for v in diffs]},
                "coarse": empty_coarse,
            }

            if segment_h5 is not None:
                segment_h5.create_dataset(vid, data=segment_embeds, compression="gzip")

            if coarse_model is not None:
                seg_feats = torch.from_numpy(segment_embeds).float()
                if seg_feats.numel() == 0:
                    entry["coarse"] = empty_coarse
                elif args.coarse_mode == "merge":
                    sims = _adjacent_cosine_similarity(segment_embeds)
                    coarse_peaks_segments = [
                        int(i) for i, s in enumerate(sims) if s < args.coarse_sim_threshold
                    ]
                    coarse_peaks_frames = _coarse_peaks_to_frame_peaks(
                        coarse_peaks_segments, segment_edges
                    )
                    entry["coarse"] = {
                        "peaks_segments": coarse_peaks_segments,
                        "peaks": [int(p) for p in coarse_peaks_frames],
                        "diffs": [float(v) for v in sims],
                    }
                else:
                    coarse_windows = build_windows(seg_feats, args.coarse_seq_len)
                    coarse_latent_list = []
                    with torch.no_grad():
                        for i in range(0, coarse_windows.shape[0], args.batch_size):
                            batch = coarse_windows[i : i + args.batch_size].to(device)
                            coarse_latent, _ = coarse_model(batch)
                            coarse_latent_list.append(coarse_latent[:, -1, :].cpu())
                    coarse_latent_feats = torch.cat(coarse_latent_list, dim=0).numpy()
                    coarse_diffs = temporal_difference(coarse_latent_feats, args.coarse_seq_len)
                    coarse_peaks_segments = detect_peaks(
                        coarse_diffs,
                        args.coarse_alpha,
                        args.coarse_threshold_std,
                        use_local_max=args.coarse_use_local_max,
                        threshold_mode=args.coarse_threshold_mode,
                    )
                    coarse_peaks_frames = _coarse_peaks_to_frame_peaks(
                        coarse_peaks_segments, segment_edges
                    )
                    entry["coarse"] = {
                        "peaks_segments": [int(p) for p in coarse_peaks_segments],
                        "peaks": [int(p) for p in coarse_peaks_frames],
                        "diffs": [float(v) for v in coarse_diffs],
                    }

            if args.recursive_levels > 0 or args.recursive_until_one:
                recursive_levels = []
                current_edges = segment_edges
                recursive_seq_len_list = _parse_level_list(args.recursive_seq_len_list, int)
                recursive_alpha_list = _parse_level_list(args.recursive_alpha_list, int)
                recursive_a_list = _parse_level_list(args.recursive_a_list, float)
                recursive_seq_len = args.recursive_seq_len or args.seq_len
                recursive_alpha = args.recursive_alpha or args.alpha
                level_idx = 0
                while True:
                    if len(current_edges) <= 2:
                        break
                    level_idx += 1
                    if args.recursive_levels > 0 and level_idx > args.recursive_levels:
                        break
                    segment_embeds = _pool_segments(
                        latent_feats, current_edges, mode=args.segment_pooling
                    )
                    if segment_embeds.size == 0:
                        break
                    if recursive_seq_len_list and level_idx - 1 < len(recursive_seq_len_list):
                        recursive_seq_len = recursive_seq_len_list[level_idx - 1]
                    if recursive_alpha_list and level_idx - 1 < len(recursive_alpha_list):
                        recursive_alpha = recursive_alpha_list[level_idx - 1]
                    recursive_a = args.recursive_a
                    if recursive_a_list and level_idx - 1 < len(recursive_a_list):
                        recursive_a = recursive_a_list[level_idx - 1]

                    seg_feats = torch.from_numpy(segment_embeds).float()
                    recursive_windows = build_windows(seg_feats, recursive_seq_len)
                    recursive_latent_list = []
                    with torch.no_grad():
                        for i in range(0, recursive_windows.shape[0], args.batch_size):
                            batch = recursive_windows[i : i + args.batch_size].to(device)
                            recursive_latent, _ = recursive_model(batch)
                            recursive_latent_list.append(recursive_latent[:, -1, :].cpu())
                    recursive_latent_feats = torch.cat(recursive_latent_list, dim=0).numpy()

                    recursive_diffs = temporal_difference(recursive_latent_feats, recursive_seq_len)
                    recursive_peaks_segments = detect_peaks(
                        recursive_diffs,
                        recursive_alpha,
                        recursive_a,
                        use_local_max=args.recursive_use_local_max,
                        threshold_mode="mad",
                    )
                    recursive_peaks_frames = _coarse_peaks_to_frame_peaks(
                        recursive_peaks_segments, current_edges
                    )
                    next_edges = _segment_edges_from_segment_peaks(
                        recursive_peaks_segments, current_edges, latent_feats.shape[0]
                    )
                    if not recursive_peaks_segments or next_edges == current_edges:
                        break
                    recursive_levels.append(
                        {
                            "level": level_idx,
                            "peaks_segments": [int(p) for p in recursive_peaks_segments],
                            "peaks": [int(p) for p in recursive_peaks_frames],
                            "diffs": [float(v) for v in recursive_diffs],
                            "edges": [int(e) for e in next_edges],
                        }
                    )
                    current_edges = next_edges
                if recursive_levels:
                    entry["levels"] = recursive_levels

            results[vid] = entry

    if segment_h5 is not None:
        segment_h5.close()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    fmt = _infer_output_format(args.output_json, args.output_format)
    if fmt == "jsonl":
        with open(args.output_json, "w") as f:
            for vid in sorted(results.keys()):
                payload = {"video_id": vid}
                payload.update(results[vid])
                f.write(json.dumps(payload) + "\n")
    else:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
