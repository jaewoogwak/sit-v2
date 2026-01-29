import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

import ipdb

import numpy as np
import logging
import torch.backends.cudnn as cudnn
import os
import pickle
import time

from tqdm import tqdm
import torch

from Utils.utils import gpu
import json
import re
from typing import List, Tuple

def _resolve_video_dir(frames_root, video_id):
    if not frames_root:
        return None
    direct = os.path.join(frames_root, video_id)
    if os.path.isdir(direct):
        return direct
    show = video_id.split("_")[0]
    folder = f"{show}_frames"
    primary = os.path.join(frames_root, folder, video_id)
    if os.path.isdir(primary):
        return primary
    try:
        for name in os.listdir(frames_root):
            if not name.endswith("_frames"):
                continue
            candidate = os.path.join(frames_root, name, video_id)
            if os.path.isdir(candidate):
                return candidate
    except Exception:
        return primary
    return primary


def _parse_frame_number(name):
    stem = os.path.splitext(name)[0]
    try:
        return int(stem)
    except ValueError:
        return None


def _list_frames(video_dir):
    frames = []
    try:
        names = os.listdir(video_dir)
    except Exception:
        return frames
    for name in names:
        if not name.lower().endswith('.jpg'):
            continue
        frame_number = _parse_frame_number(name)
        if frame_number is None:
            continue
        frames.append((frame_number, name))
    frames.sort(key=lambda x: x[0])
    return frames


def _sample_indices(start, end, count):
    if count <= 1:
        return [start]
    span = max(1, end - start)
    return [int(start + (span - 1) * i / float(count - 1)) for i in range(count)]


def _render_segment_strip(video_dir, frames, start_idx, end_idx, count, tile_w, tile_h, title):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    if not frames:
        return None
    start_idx = max(0, min(len(frames), int(start_idx)))
    end_idx = max(start_idx + 1, min(len(frames), int(end_idx)))
    indices = _sample_indices(start_idx, end_idx, count)

    imgs = []
    for idx in indices:
        if idx < 0 or idx >= len(frames):
            continue
        _, name = frames[idx]
        path = os.path.join(video_dir, name)
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            continue
        imgs.append(img.resize((tile_w, tile_h)))

    if not imgs:
        return None

    grid = Image.new('RGB', (tile_w * len(imgs), tile_h + 20), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid)
    for i, img in enumerate(imgs):
        grid.paste(img, (i * tile_w, 20))
    if title:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((4, 2), title, fill=(255, 255, 255), font=font)
    return grid


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def log_cuda(tag: str):
    """Log CUDA memory (allocated, reserved, peak) in GiB with a tag."""
    if not torch.cuda.is_available():
        return
    try:
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[MEM] {tag} | alloc={alloc:.2f}G reserved={reserved:.2f}G peak_alloc={peak:.2f}G")
    except Exception:
        pass


def _get_model_attr(model, name, default=None):
    if hasattr(model, "module"):
        return getattr(model.module, name, default)
    return getattr(model, name, default)


def _extract_boundary_peaks(entry):
    fine_peaks = []
    coarse_peaks = []
    has_fine = False
    has_coarse = False
    if isinstance(entry, dict):
        if "fine" in entry or "coarse" in entry:
            has_fine = "fine" in entry
            has_coarse = "coarse" in entry
            fine_peaks = entry.get("fine", {}).get("peaks", []) or []
            coarse_peaks = entry.get("coarse", {}).get("peaks", []) or []
        elif "levels" in entry:
            has_fine = True
            levels = entry.get("levels") or []
            if isinstance(levels, list):
                for level_entry in levels:
                    if not isinstance(level_entry, dict):
                        continue
                    edges = level_entry.get("edges") or []
                    if edges:
                        for edge in edges[1:-1]:
                            try:
                                peak = int(edge) - 1
                            except Exception:
                                continue
                            if peak >= 0:
                                fine_peaks.append(peak)
                    else:
                        fine_peaks.extend(level_entry.get("peaks", []) or [])
        else:
            has_fine = True
            fine_peaks = entry.get("peaks", []) or []
    elif isinstance(entry, list):
        has_fine = True
        fine_peaks = entry
    return fine_peaks, coarse_peaks, has_fine, has_coarse


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


def _peaks_from_level_entry(level_entry, num_frames):
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


def _split_bounds_by_peaks(num_frames, peaks):
    if num_frames <= 0:
        return [(0, 0)]
    if not peaks:
        return [(0, num_frames)]
    clean_peaks = sorted({int(p) for p in peaks if p is not None})
    clean_peaks = [p for p in clean_peaks if 0 <= p < num_frames]
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
    if not bounds:
        bounds = [(0, num_frames)]
    return bounds


def _unique_bounds(bounds):
    unique = {(int(s), int(e)) for s, e in bounds if s is not None and e is not None and e > s}
    return sorted(unique, key=lambda x: (x[0], x[1]))


def _bounds_from_levels(num_frames, levels, level_num=None):
    bounds = []
    if not isinstance(levels, list):
        return bounds
    for level_entry in levels:
        if not isinstance(level_entry, dict):
            continue
        if level_num is not None and level_entry.get("level") != level_num:
            continue
        peaks = _peaks_from_level_entry(level_entry, num_frames)
        bounds.extend(_split_bounds_by_peaks(num_frames, peaks))
    return bounds


def _compute_segment_bounds(entry, num_frames, boundary_level, dedupe_segments):
    bounds = []
    boundary_level = boundary_level or "fine"
    if dedupe_segments:
        if boundary_level == "both":
            if isinstance(entry, dict) and ("fine" in entry or "coarse" in entry):
                fine_peaks = entry.get("fine", {}).get("peaks", [])
                coarse_peaks = entry.get("coarse", {}).get("peaks", [])
            elif isinstance(entry, dict):
                fine_peaks = entry.get("peaks", [])
                coarse_peaks = []
            elif isinstance(entry, list):
                fine_peaks = entry
                coarse_peaks = []
            else:
                fine_peaks = []
                coarse_peaks = []
            bounds.extend(_split_bounds_by_peaks(num_frames, fine_peaks))
            bounds.extend(_split_bounds_by_peaks(num_frames, coarse_peaks))
        elif "+" in boundary_level:
            tokens = [t.strip() for t in boundary_level.split("+") if t.strip()]
            for token in tokens:
                token_name, level_num = _parse_level_token(token)
                if token_name in ("fine", "coarse"):
                    if isinstance(entry, dict) and ("fine" in entry or "coarse" in entry):
                        peaks = entry.get(token_name, {}).get("peaks", [])
                    elif isinstance(entry, dict) and token_name == "fine":
                        peaks = entry.get("peaks", [])
                    elif isinstance(entry, list) and token_name == "fine":
                        peaks = entry
                    else:
                        peaks = []
                    bounds.extend(_split_bounds_by_peaks(num_frames, peaks))
                elif token_name in ("levels", "level"):
                    levels = entry.get("levels") if isinstance(entry, dict) else []
                    bounds.extend(_bounds_from_levels(num_frames, levels, level_num))
            if not bounds:
                bounds = _split_bounds_by_peaks(num_frames, [])
        else:
            token_name, level_num = _parse_level_token(boundary_level)
            if token_name in ("levels", "level"):
                levels = entry.get("levels") if isinstance(entry, dict) else []
                bounds = _bounds_from_levels(num_frames, levels, level_num)
            else:
                if isinstance(entry, list):
                    peaks = entry
                elif isinstance(entry, dict):
                    if "fine" in entry or "coarse" in entry:
                        entry = entry.get(boundary_level, {}) or {}
                    peaks = entry.get("peaks", [])
                else:
                    peaks = []
                bounds = _split_bounds_by_peaks(num_frames, peaks)
        bounds = _unique_bounds(bounds)
        if not bounds:
            bounds = [(0, num_frames)]
    else:
        if boundary_level == "both":
            if isinstance(entry, dict) and ("fine" in entry or "coarse" in entry):
                fine_peaks = entry.get("fine", {}).get("peaks", [])
                coarse_peaks = entry.get("coarse", {}).get("peaks", [])
            elif isinstance(entry, dict):
                fine_peaks = entry.get("peaks", [])
                coarse_peaks = []
            elif isinstance(entry, list):
                fine_peaks = entry
                coarse_peaks = []
            else:
                fine_peaks = []
                coarse_peaks = []
            bounds.extend(_split_bounds_by_peaks(num_frames, fine_peaks))
            bounds.extend(_split_bounds_by_peaks(num_frames, coarse_peaks))
        elif "+" in boundary_level:
            tokens = [t.strip() for t in boundary_level.split("+") if t.strip()]
            for token in tokens:
                token_name, level_num = _parse_level_token(token)
                if token_name in ("fine", "coarse"):
                    if isinstance(entry, dict) and ("fine" in entry or "coarse" in entry):
                        peaks = entry.get(token_name, {}).get("peaks", [])
                    elif isinstance(entry, dict) and token_name == "fine":
                        peaks = entry.get("peaks", [])
                    elif isinstance(entry, list) and token_name == "fine":
                        peaks = entry
                    else:
                        peaks = []
                    bounds.extend(_split_bounds_by_peaks(num_frames, peaks))
                elif token_name in ("levels", "level"):
                    levels = entry.get("levels") if isinstance(entry, dict) else []
                    bounds.extend(_bounds_from_levels(num_frames, levels, level_num))
            if not bounds:
                bounds = _split_bounds_by_peaks(num_frames, [])
        else:
            token_name, level_num = _parse_level_token(boundary_level)
            if token_name in ("levels", "level"):
                levels = entry.get("levels") if isinstance(entry, dict) else []
                bounds = _bounds_from_levels(num_frames, levels, level_num)
            else:
                if isinstance(entry, list):
                    peaks = entry
                elif isinstance(entry, dict):
                    if "fine" in entry or "coarse" in entry:
                        entry = entry.get(boundary_level, {}) or {}
                    peaks = entry.get("peaks", [])
                else:
                    peaks = []
                bounds = _split_bounds_by_peaks(num_frames, peaks)
    return bounds


def _normalize_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _load_release_map(paths):
    release_map = {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                vid = entry.get("vid_name")
                desc = entry.get("desc")
                ts = entry.get("ts")
                duration = entry.get("duration")
                if vid is None or desc is None or ts is None or duration is None:
                    continue
                release_map[(vid, desc)] = (float(ts[0]), float(ts[1]), float(duration))
    return release_map


def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:

                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt


def eval_q2m(scores, q2m_gts):

    n_q, n_m = scores.shape
    print(f"n_q {n_q} n_m {n_m}")
    gt_ranks = torch.zeros((n_q), dtype=torch.int32).cuda()
    aps = torch.zeros(n_q).cuda()
    # Initialize per-query end times list for end-to-end timing (model start2 -> argsort end)
    eval_q2m._per_query_end_times = []
    for i in range(n_q): # n_q 10해놓기
        s = scores[i]
        sorted_idxs = torch.argsort(s) # top10만
        # end 2
        if hasattr(eval_q2m, '_timing_end2'):
            _sync(); eval_q2m._timing_end2 = time.perf_counter()
        # record per-query end time (after argsort)
        _sync(); eval_q2m._per_query_end_times.append(time.perf_counter())
        rank = n_m + 1
        tmp_set = []
        # guard: some queries may have no GT (not present in mapping)
        keys = q2m_gts.get(i, []) if isinstance(q2m_gts, dict) else q2m_gts[i]
        for k in keys:
            tmp = torch.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank
    
    
    
    
    # compute metrics
    r1 = 100.0 * len(torch.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(torch.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(torch.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(torch.where(gt_ranks <= 100)[0]) / n_q

    return (r1, r5, r10, r100)


def cal_perf(t2v_all_errors, t2v_gt):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100) = eval_q2m(t2v_all_errors, t2v_gt)

    return (t2v_r1, t2v_r5, t2v_r10, t2v_r100)


class validations(nn.Module):
    def __init__(self, cfg):
        super(validations, self).__init__()

        self.cfg = cfg
        self._release_map = None
        self._release_map_norm = None

    def _debug_video_eval(self, model, context_dataloader, context_info, query_eval_loader, score_sum, query_metas):
        video_id = str(self.cfg.get("eval_debug_vid", "")).strip()
        if not video_id:
            return

        video_metas = context_info.get("video_metas", [])
        if video_id not in video_metas:
            print(f"[Eval Debug] video_id not found in context: {video_id}")
            return

        vid_idx = video_metas.index(video_id)
        clip_mask = context_info.get("clip_mask", None)
        if clip_mask is not None:
            seg_mask = clip_mask[vid_idx]
            segment_count = int(seg_mask.sum().item())
        else:
            seg_mask = None
            segment_count = int(context_info["video_proposal_feat"][vid_idx].shape[0])

        boundaries = getattr(context_dataloader.dataset, "boundaries", None)
        boundary_level = getattr(context_dataloader.dataset, "boundary_level", None)
        dedupe_segments = bool(getattr(context_dataloader.dataset, "dedupe_segments", False))
        fine_peaks = []
        coarse_peaks = []
        has_fine = False
        has_coarse = False
        if isinstance(boundaries, dict):
            entry = boundaries.get(video_id, {})
            fine_peaks, coarse_peaks, has_fine, has_coarse = _extract_boundary_peaks(entry)

        fine_segments = (len(fine_peaks) + 1) if has_fine else 0
        coarse_segments = (len(coarse_peaks) + 1) if has_coarse else 0

        gt_cap_ids = [cap_id for cap_id in query_metas if cap_id.split("#", 1)[0] == video_id]
        if not gt_cap_ids:
            print(f"[Eval Debug] no GT queries found for video_id={video_id}")
            return

        topk = int(self.cfg.get("eval_debug_topk", 10) or 10)
        seg_topk = int(self.cfg.get("eval_debug_segment_topk", 5) or 5)

        cap_id_to_idx = {cap_id: i for i, cap_id in enumerate(query_metas)}
        captions = getattr(query_eval_loader.dataset, "captions", {})
        if self._release_map is None:
            paths = []
            for key in ("tvr_release_test_path", "tvr_release_val_path", "tvr_release_train_path"):
                if key in self.cfg:
                    paths.append(self.cfg.get(key))
            self._release_map = _load_release_map(paths)
            self._release_map_norm = {}
            for (vid, desc), ts in self._release_map.items():
                self._release_map_norm[(vid, _normalize_text(desc))] = ts

        num_frames = None
        loader = getattr(context_dataloader.dataset, "_load_frames", None)
        if callable(loader):
            try:
                frames = loader(video_id)
                num_frames = int(frames.shape[0])
            except Exception:
                num_frames = None

        segment_bounds = []
        if isinstance(boundaries, dict) and num_frames is not None:
            entry = boundaries.get(video_id, {})
            segment_bounds = _compute_segment_bounds(entry, num_frames, boundary_level, dedupe_segments)
        else:
            entry = {}

        segment_details = []
        fine_bounds = []
        levels_bounds = []
        levels_labeled_bounds = []
        levels_peaks_info = []
        if isinstance(entry, dict) and num_frames is not None:
            levels = entry.get("levels") if isinstance(entry, dict) else []
            if isinstance(levels, list):
                for level_entry in levels:
                    if not isinstance(level_entry, dict):
                        continue
                    level_id = level_entry.get("level")
                    peaks = _peaks_from_level_entry(level_entry, num_frames)
                    levels_peaks_info.append((level_id, peaks))
            tokens = []
            if boundary_level:
                if "+" in boundary_level:
                    tokens = [t.strip() for t in boundary_level.split("+") if t.strip()]
                else:
                    tokens = [boundary_level]
            if tokens:
                labeled_bounds = []
                for token in tokens:
                    token_name, level_num = _parse_level_token(token)
                    if token_name == "fine":
                        peaks = fine_peaks if has_fine else []
                        bounds = _split_bounds_by_peaks(num_frames, peaks)
                        fine_bounds = bounds
                        labeled_bounds.extend([(b, "fine") for b in bounds])
                    elif token_name in ("levels", "level"):
                        if isinstance(levels, list):
                            for level_entry in levels:
                                if not isinstance(level_entry, dict):
                                    continue
                                if level_num is not None and level_entry.get("level") != level_num:
                                    continue
                                level_id = level_entry.get("level")
                                peaks = _peaks_from_level_entry(level_entry, num_frames)
                                bounds = _split_bounds_by_peaks(num_frames, peaks)
                                label = f"level{level_id}" if level_id is not None else "level?"
                                levels_labeled_bounds.extend([(b, label) for b in bounds])
                        bounds = _bounds_from_levels(num_frames, levels, level_num)
                        levels_bounds = bounds
                        labeled_bounds.extend(levels_labeled_bounds)
                    elif token_name == "coarse":
                        peaks = coarse_peaks if has_coarse else []
                        bounds = _split_bounds_by_peaks(num_frames, peaks)
                        labeled_bounds.extend([(b, "coarse") for b in bounds])
                if labeled_bounds:
                    if dedupe_segments:
                        bounds = _unique_bounds([b for b, _ in labeled_bounds])
                        fine_set = set(b for b, label in labeled_bounds if label == "fine")
                        coarse_set = set(b for b, label in labeled_bounds if label == "coarse")
                        level_map = {}
                        for b, label in labeled_bounds:
                            if not label.startswith("level"):
                                continue
                            level_map.setdefault(b, set()).add(label)
                        for b in bounds:
                            labels = []
                            if b in fine_set:
                                labels.append("fine")
                            if b in level_map:
                                labels.extend(sorted(level_map[b]))
                            if b in coarse_set:
                                labels.append("coarse")
                            segment_details.append((b, "+".join(labels) if labels else "unknown"))
                    else:
                        segment_details = labeled_bounds

        dataset_name = str(self.cfg.get("dataset_name", "dataset")).strip() or "dataset"
        out_root = self.cfg.get(
            "eval_debug_out_dir",
            os.path.join(self.cfg.get("root", "."), "results", "debug_eval", dataset_name),
        )
        video_out_dir = os.path.join(out_root, video_id)
        os.makedirs(video_out_dir, exist_ok=True)
        frames_root = str(self.cfg.get('eval_debug_frames_root', '')).strip()
        segment_frames = int(self.cfg.get('eval_debug_segment_frames', 0) or 0)

        needed = set(gt_cap_ids)
        query_cache = {}
        with torch.no_grad():
            for batch in query_eval_loader:
                batch = gpu(batch)
                query_feat, query_mask, _, cap_ids = batch
                for i, cap_id in enumerate(cap_ids):
                    if cap_id in needed and cap_id not in query_cache:
                        query_cache[cap_id] = (query_feat[i:i + 1], query_mask[i:i + 1])
                if len(query_cache) >= len(needed):
                    break

        print("\n[Eval Debug]")
        print(f"video_id: {video_id}")
        if boundary_level:
            print(f"boundary_level: {boundary_level}")
        if has_fine:
            print(f"fine_peaks: {fine_peaks}")
        if levels_peaks_info:
            for level_id, peaks in levels_peaks_info:
                print(f"levels_peaks: level={level_id} peaks={peaks}")
        if has_fine or has_coarse:
            if "levels" in str(boundary_level):
                print(f"segment_count: total={segment_count} fine={fine_segments} levels={len(levels_bounds)}")
            else:
                print(f"segment_count: total={segment_count} fine={fine_segments} coarse={coarse_segments}")
        else:
            print(f"segment_count: total={segment_count}")
        if segment_details:
            for idx, (bounds, label) in enumerate(segment_details):
                s, e = bounds
                ts_start = float(s) / 3.0
                ts_end = float(e) / 3.0
                print(
                    f"seg{idx}: [{int(s)},{int(e)}] level={label} "
                    f"ts_3fps={ts_start:.3f}-{ts_end:.3f}"
                )

        debug_json = {
            "video_id": video_id,
            "dataset": dataset_name,
            "boundary_level": boundary_level,
            "dedupe_segments": dedupe_segments,
            "num_frames": num_frames,
            "segment_bounds": [[int(s), int(e)] for s, e in segment_bounds],
            "segment_count": int(segment_count),
            "gt_queries": [],
        }

        for cap_id in gt_cap_ids:
            q_idx = cap_id_to_idx.get(cap_id, None)
            if q_idx is None:
                continue

            clip_s_row = None
            frame_s_row = None
            video_query = None
            if cap_id in query_cache:
                query_feat, query_mask = query_cache[cap_id]
                get_pred = _get_model_attr(model, "get_pred_from_raw_query")
                if callable(get_pred):
                    with torch.no_grad():
                        clip_s, frame_s = get_pred(
                            query_feat, query_mask, None,
                            context_info["video_proposal_feat"], context_info["video_feat"],
                            clip_mask=context_info.get("clip_mask"),
                            return_query_feats=False, return_timing=False
                        )
                    clip_s_row = clip_s.detach().cpu().squeeze(0)
                    frame_s_row = frame_s.detach().cpu().squeeze(0)
                encode_query = _get_model_attr(model, "encode_query")
                if callable(encode_query):
                    with torch.no_grad():
                        video_query = encode_query(query_feat, query_mask)
                    if video_query.dim() == 1:
                        video_query = video_query.unsqueeze(0)
                    video_query = F.normalize(video_query, dim=-1)

            scores_row = score_sum[q_idx]
            k = min(topk, scores_row.numel())
            top_vals, top_idxs = torch.topk(scores_row, k=k, largest=True)

            sorted_idxs = torch.argsort(scores_row, descending=True)
            gt_rank = int(torch.where(sorted_idxs == vid_idx)[0][0].item()) + 1

            cap_text = captions.get(cap_id, "")
            gt_ts = None
            gt_ts_3fps = None
            gt_span_frames = None
            if cap_text and self._release_map:
                ts = self._release_map.get((video_id, cap_text))
                if ts is None:
                    ts = self._release_map_norm.get((video_id, _normalize_text(cap_text)))
                if ts is not None:
                    ts_start, ts_end, ts_dur = ts
                    ts_start_3fps = ts_start * 3.0
                    ts_end_3fps = ts_end * 3.0
                    gt_ts = [float(ts_start), float(ts_end), float(ts_dur)]
                    gt_ts_3fps = [float(ts_start_3fps), float(ts_end_3fps)]
                    if num_frames is not None and ts_dur and ts_dur > 0:
                        scale = float(num_frames) / float(ts_dur)
                        gt_span_frames = [
                            max(0, min(num_frames, int(ts_start * scale))),
                            max(0, min(num_frames, int(ts_end * scale))),
                        ]
            query_line = f"\nquery: {cap_id}"
            if cap_text:
                query_line += f" | {cap_text}"
            if gt_ts is not None:
                query_line += (
                    f" | ts={ts_start:.3f}-{ts_end:.3f} (dur={ts_dur:.3f})"
                    f" | ts_x3={ts_start_3fps:.3f}-{ts_end_3fps:.3f}"
                )
            print(query_line)
            if gt_ts is not None:
                print(f"gt_ts: {ts_start:.3f}-{ts_end:.3f} / dur={ts_dur:.3f}")
                print(f"gt_ts_3fps: {ts_start_3fps:.3f}-{ts_end_3fps:.3f}")
            print(f"video_rank: {gt_rank} / {scores_row.numel()}")
            print("top_videos:")
            seg_bounds_cache = {}

            def _get_seg_bounds_for_vid(vid):
                if vid in seg_bounds_cache:
                    return seg_bounds_cache[vid]
                if not isinstance(boundaries, dict):
                    seg_bounds_cache[vid] = []
                    return seg_bounds_cache[vid]
                if not callable(loader):
                    seg_bounds_cache[vid] = []
                    return seg_bounds_cache[vid]
                try:
                    frames = loader(vid)
                    n_frames = int(frames.shape[0])
                except Exception:
                    seg_bounds_cache[vid] = []
                    return seg_bounds_cache[vid]
                entry = boundaries.get(vid, {})
                bounds = _compute_segment_bounds(entry, n_frames, boundary_level, dedupe_segments)
                seg_bounds_cache[vid] = bounds
                return bounds

            for r, (v_idx, v_score) in enumerate(zip(top_idxs.tolist(), top_vals.tolist()), start=1):
                vid_name = video_metas[v_idx] if v_idx < len(video_metas) else str(v_idx)
                seg_idx = None
                if video_query is not None:
                    ctx_feat = context_info["video_proposal_feat"][v_idx]
                    if not bool(self.cfg.get("pre_normalize_context", False)):
                        ctx_feat = F.normalize(ctx_feat, dim=-1)
                    seg_scores = torch.matmul(ctx_feat, video_query.t()).squeeze(1)
                    if clip_mask is not None:
                        seg_scores = seg_scores.masked_fill(clip_mask[v_idx] == 0, -1e10)
                    if seg_scores.numel():
                        seg_idx = int(torch.argmax(seg_scores).item())
                seg_bounds_info = ""
                if seg_idx is not None:
                    bounds = _get_seg_bounds_for_vid(vid_name)
                    if bounds and 0 <= seg_idx < len(bounds):
                        s, e = bounds[seg_idx]
                        ts_start = float(s) / 3.0
                        ts_end = float(e) / 3.0
                        seg_bounds_info = (
                            f" [{int(s)},{int(e)}] ts_3fps={ts_start:.3f}-{ts_end:.3f}"
                        )
                if clip_s_row is not None and frame_s_row is not None:
                    clip_w = float(self.cfg.get("clip_scale_w", 1.0))
                    frame_w = float(self.cfg.get("frame_scale_w", 1.0))
                    clip_part = float(clip_s_row[v_idx]) * clip_w
                    frame_part = float(frame_s_row[v_idx]) * frame_w
                    seg_info = f" segment={seg_idx}{seg_bounds_info}" if seg_idx is not None else ""
                    print(
                        f"  {r:2d}. {vid_name} score={v_score:.4f}{seg_info} "
                        f"(clip_s*clip_w={clip_s_row[v_idx]:.4f}*{clip_w:.3f}={clip_part:.4f} "
                        f"+ frame_s*frame_w={frame_s_row[v_idx]:.4f}*{frame_w:.3f}={frame_part:.4f})"
                    )
                else:
                    seg_info = f" segment={seg_idx}{seg_bounds_info}" if seg_idx is not None else ""
                    print(f"  {r:2d}. {vid_name} score={v_score:.4f}{seg_info}")

            seg_scores_list = []
            if cap_id in query_cache:
                if video_query is None:
                    print("top_segments: (model has no encode_query)")
                    seg_scores_list = []

                if video_query is not None:
                    ctx_feat = context_info["video_proposal_feat"][vid_idx]
                    if not bool(self.cfg.get("pre_normalize_context", False)):
                        ctx_feat = F.normalize(ctx_feat, dim=-1)
                    seg_scores = torch.matmul(ctx_feat, video_query.t()).squeeze(1)
                    if seg_mask is not None:
                        seg_scores = seg_scores.masked_fill(seg_mask == 0, -1e10)
                    sk = min(seg_topk, seg_scores.numel())
                    seg_vals, seg_idxs = torch.topk(seg_scores, k=sk, largest=True)
                    print("top_segments:")
                    for r, (s_idx, s_score) in enumerate(zip(seg_idxs.tolist(), seg_vals.tolist()), start=1):
                        print(f"  {r:2d}. segment={s_idx} score={s_score:.4f}")
                    seg_scores_list = seg_scores.detach().cpu().tolist()

                    if frames_root and segment_frames > 0 and segment_bounds:
                        video_dir = _resolve_video_dir(frames_root, video_id)
                        frames = _list_frames(video_dir) if video_dir else []
                        if frames:
                            strips = []
                            for r, (s_idx, s_score) in enumerate(zip(seg_idxs.tolist(), seg_vals.tolist()), start=1):
                                if s_idx >= len(segment_bounds):
                                    continue
                                s, e = segment_bounds[s_idx]
                                title = f"seg {s_idx} score={s_score:.3f} [{int(s)},{int(e)}]"
                                strip = _render_segment_strip(
                                    video_dir, frames, s, e, segment_frames, 160, 90, title
                                )
                                if strip is not None:
                                    strips.append(strip)
                            if strips:
                                try:
                                    from PIL import Image
                                    total_h = sum(img.size[1] for img in strips)
                                    total_w = max(img.size[0] for img in strips)
                                    canvas = Image.new('RGB', (total_w, total_h), color=(20, 20, 20))
                                    y = 0
                                    for img in strips:
                                        canvas.paste(img, (0, y))
                                        y += img.size[1]
                                    out_img = os.path.join(video_out_dir, f"{cap_id}_top_segments.png")
                                    canvas.save(out_img)
                                except Exception:
                                    pass
            else:
                print("top_segments: (query features not found)")

            seg_scores_count = len(seg_scores_list)
            bounds_count = len(segment_bounds)
            valid_seg_count = None
            if seg_mask is not None:
                valid_seg_count = int(seg_mask.sum().item())
            if valid_seg_count is not None:
                seg_scores_list = seg_scores_list[:valid_seg_count]

            query_json = {
                "cap_id": cap_id,
                "caption": cap_text,
                "gt_ts": gt_ts,
                "gt_ts_3fps": gt_ts_3fps,
                "gt_span_frames": gt_span_frames,
                "segment_scores": seg_scores_list,
                "segment_scores_count": seg_scores_count,
                "segment_bounds_count": bounds_count,
                "segment_bounds_mismatch": bool(bounds_count and seg_scores_list and bounds_count != len(seg_scores_list)),
            }
            debug_json["gt_queries"].append(query_json)

            if num_frames is not None and segment_bounds and seg_scores_list:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                scores_by_frame = np.full((num_frames,), np.nan, dtype=np.float32)
                limit = min(len(segment_bounds), len(seg_scores_list))
                for (s, e), score in zip(segment_bounds[:limit], seg_scores_list[:limit]):
                    s_idx = max(0, min(num_frames, int(s)))
                    e_idx = max(0, min(num_frames, int(e)))
                    if e_idx > s_idx:
                        scores_by_frame[s_idx:e_idx] = float(score)
                fig = plt.figure(figsize=(10, 3))
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(np.arange(num_frames), scores_by_frame, linewidth=1.0)
                if gt_span_frames:
                    ax.axvspan(gt_span_frames[0], gt_span_frames[1], color="orange", alpha=0.2, label="gt")
                    ax.legend(loc="upper right")
                ax.set_xlabel("frame index")
                ax.set_ylabel("segment score")
                fig.tight_layout()
                out_png = os.path.join(video_out_dir, f"{cap_id}_segment_scores.png")
                fig.savefig(out_png, dpi=200)
                plt.close(fig)

        out_json = os.path.join(video_out_dir, "debug_segments.json")
        with open(out_json, "w") as f:
            json.dump(debug_json, f, ensure_ascii=True, indent=2)
    def forward(self, model, context_dataloader, query_eval_loader):

        model.eval()
        
        # Enable timing measurement for detailed analysis
        measure_timing = bool(self.cfg.get("measure_search", False))
        
        context_info = self.compute_context_info(model, context_dataloader)
        
        # Get video count for timing analysis
        num_videos = len(context_info['video_metas'])
        
        score_sum, query_metas, timing_report = self.compute_query2ctx_info(model,
                                                            query_eval_loader,
                                                            context_info)
               
        video_metas = context_info['video_metas']

        self._debug_video_eval(model, context_dataloader, context_info, query_eval_loader, score_sum, query_metas)

        # Measure get_gt time to exclude it later
        if measure_timing:
            _sync(); get_gt_start = time.perf_counter()
        v2t_gt, t2v_gt = get_gt(video_metas, query_metas) # 시간 잴때 빼기
        if measure_timing:
            _sync(); get_gt_end = time.perf_counter()
            get_gt_time = (get_gt_end - get_gt_start) * 1000.0

        # Setup timing for cal_perf -> eval_q2m -> torch.argsort
        if measure_timing:
            eval_q2m._timing_end2 = None

        t2v_r1, t2v_r5, t2v_r10, t2v_r100 = cal_perf(-1 * score_sum, t2v_gt)
        
        # Calculate similarity check time per query (start -> argsort end)
        similarity_check_time = None
        if measure_timing and timing_report and 'similarity_start_times' in timing_report:
            starts = timing_report['similarity_start_times']
            ends = timing_report.get('similarity_end_times', None)
            if ends and len(ends) == len(starts):
                # Per-query durations in ms (excludes GT, measured inside scoring loop)
                durs_ms = [ (ends[i] - starts[i]) * 1000.0 for i in range(len(starts)) ]
                if len(durs_ms) > 0:
                    similarity_check_time = float(np.mean(durs_ms))
            else:
                # Fallback: average from eval_q2m end timestamp (subtract GT time)
                if hasattr(eval_q2m, '_timing_end2') and eval_q2m._timing_end2:
                    avg_start_time = sum(starts) / len(starts)
                    similarity_check_time = (eval_q2m._timing_end2 - avg_start_time) * 1000.0 - get_gt_time
        
        if timing_report:
            # Add our custom timing measurements
            if measure_timing and 'encoding_normalize_avg_ms' in timing_report and similarity_check_time:
                encoding_time = timing_report['encoding_normalize_avg_ms']
                total_search_time = encoding_time + similarity_check_time
                
                print("\n[Performance Timing Analysis]")
                print(f"Total search time: {total_search_time:.2f} ms")
                print(f"Encoding & normalize time: {encoding_time:.2f} ms")
                print(f"Similarity check time: {similarity_check_time:.2f} ms")
                print(f"Videos compared per query: {num_videos}")
                # If top-k timing mode is used, show it
                if int(timing_report.get('timing_topk', 0)) > 0:
                    print(f"Timing mode: top-k (k={int(timing_report.get('timing_topk', 0))})")
                dev = timing_report.get('timing_device', None)
                if dev:
                    print(f"Timing device: {dev.upper()}")
                # Inform that GT time was excluded from similarity check time
                if measure_timing:
                    print(f"(GT mapping time excluded: {get_gt_time:.2f} ms)")
                print()

            # Per-query breakdown (avg over queries)
            if 'per_query_ms' in timing_report:
                pq_stats = timing_report['per_query_ms']
                pq_search_avg = pq_stats.get('avg', None)
                enc_avg = timing_report.get('encoding_normalize_avg_ms', None)
                print("[Per-Query Timing]")
                # Also print how many videos each query is compared against
                print(f"Videos compared per query: {num_videos}")
                if pq_search_avg is not None:
                    print(f"Similarity (all videos) avg: {pq_search_avg:.2f} ms")
                if enc_avg is not None:
                    print(f"Encoding + normalize avg: {enc_avg:.2f} ms")
                if pq_search_avg is not None and enc_avg is not None:
                    print(f"Total per-query avg (enc + sim): {enc_avg + pq_search_avg:.2f} ms")
                # Optional distribution percentiles if available
                if 'p50' in pq_stats and 'p90' in pq_stats and 'p99' in pq_stats:
                    print(f"Similarity per-query p50/p90/p99: {pq_stats['p50']:.2f} / {pq_stats['p90']:.2f} / {pq_stats['p99']:.2f} ms")
                # Argsort/top-k sorting cost
                sort_stats = timing_report.get('argsort_ms', None)
                if sort_stats and isinstance(sort_stats, dict):
                    label = 'top-k' if int(timing_report.get('timing_topk', 0)) > 0 else 'argsort'
                    print(f"{label.capitalize()} avg: {sort_stats.get('avg', 0.0):.2f} ms")
                    if 'p50' in sort_stats and 'p90' in sort_stats and 'p99' in sort_stats:
                        print(f"{label.capitalize()} p50/p90/p99: {sort_stats['p50']:.2f} / {sort_stats['p90']:.2f} / {sort_stats['p99']:.2f} ms")
                if int(timing_report.get('timing_topk', 0)) > 0:
                    print(f"(Per-query timing uses top-k k={int(timing_report.get('timing_topk', 0))})")
                dev = timing_report.get('timing_device', None)
                if dev:
                    print(f"Timing device: {dev.upper()}")
                # End-to-end per-query similarity (model start2 -> argsort end)
                try:
                    starts = timing_report.get('similarity_start_times', None)
                    ends = timing_report.get('similarity_end_times', None)
                    if ends and starts and len(ends) > 0 and len(starts) > 0 and len(ends) == len(starts):
                        import numpy as _np
                        e2e_arr = _np.array([(ends[i] - starts[i]) * 1000.0 for i in range(len(starts))], dtype=_np.float64)
                        print(f"Similarity (end-to-end, incl argsort) avg: {e2e_arr.mean():.2f} ms")
                        print(f"Similarity (end-to-end) p50/p90/p99: { _np.percentile(e2e_arr,50):.2f} / { _np.percentile(e2e_arr,90):.2f} / { _np.percentile(e2e_arr,99):.2f} ms")
                    else:
                        # Fallback to eval_q2m timestamps if end times not available
                        end_times = getattr(eval_q2m, '_per_query_end_times', None)
                        if end_times and starts and len(end_times) > 0 and len(starts) > 0:
                            n = min(len(end_times), len(starts))
                            ends_aligned = end_times[-n:]
                            starts_aligned = starts[-n:]
                            per_gt_ms = (get_gt_time / n) if measure_timing else 0.0
                            e2e = [(ends_aligned[i] - starts_aligned[i]) * 1000.0 - per_gt_ms for i in range(n)]
                            import numpy as _np
                            e2e_arr = _np.array(e2e, dtype=_np.float64)
                            print(f"Similarity (end-to-end, incl argsort) avg: {e2e_arr.mean():.2f} ms")
                            print(f"Similarity (end-to-end) p50/p90/p99: { _np.percentile(e2e_arr,50):.2f} / { _np.percentile(e2e_arr,90):.2f} / { _np.percentile(e2e_arr,99):.2f} ms")
                except Exception:
                    pass

                print()
            
            print("[Detailed Search Timing]")
            # print(json.dumps(timing_report, indent=2))
            

        t2v_rsum = 0
        t2v_rsum += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)

        return [t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_rsum]

    def compute_query2ctx_info(self, model, query_eval_loader, ctx_info):
        measure = bool(self.cfg.get("measure_search", False))
        timing_topk = int(self.cfg.get("timing_topk", 0))
        timing_on_gpu = bool(self.cfg.get("timing_on_gpu", False))
        warmup_batches = int(self.cfg.get("timing_warmup_batches", 1))  # exclude first N batches from timing
        per_query_ms = []   # 전체 쿼리 단위 레이턴시(ms)
        per_clip_ms = []    # 배치 단위 구간 시간(ms)
        per_frame_ms = []
        argsort_ms = []        # per-query argsort/top-k 정렬 시간(ms)
        encoding_times = []  # encoding & normalize times
        similarity_start_times = []  # similarity check start times
        similarity_end_times = []    # per-query end time after argsort

        query_metas, score_sum_list = [], []
        for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):
            
            batch = gpu(batch)
            query_metas.extend(batch[-1])
            query_feat = batch[0] # cfg['eval_query_bsz'] = 50 크기만큼 배치 크기
            query_mask = batch[1]
            
            # ctx_info에는 video_proposal_feat (32)랑 video_feat(1) 존재. 전체 비디오 들고있음 
            # video proposal feat (total_video_num, 32, 384)
            # video feat (total_video_num, 384)
            # print(f"##### [Test] Video proposal feat:{ctx_info['video_proposal_feat'].shape} Video feat:{ ctx_info['video_feat'].shape}")
            # breakpoint()
            if measure:
                # 평가에서만 측정 켜기
                clip_s, frame_s, timing = model.get_pred_from_raw_query(
                    query_feat, query_mask, None,
                    ctx_info["video_proposal_feat"], ctx_info["video_feat"],
                    clip_mask=ctx_info.get("clip_mask"),
                    return_query_feats=False, return_timing=True
                )
                bs = query_feat.size(0)
                
                collect = (idx >= warmup_batches)
                if collect:
                    per_query_ms.extend([timing["search_total_ms"]/bs]*bs)
                    per_clip_ms.append(timing["clip_ms"])
                    per_frame_ms.append(timing["frame_ms"])
                    # encoding normalize per-query 평균도 동일 기준으로 수집
                    encoding_times.extend([timing["encoding_normalize_ms"]/bs]*bs)
                    # Collect similarity start times for cross-function timing
                    if "similarity_start_time" in timing:
                        similarity_start_times.extend([timing["similarity_start_time"]]*bs)
                
            else:
                clip_s, frame_s = model.get_pred_from_raw_query(
                    query_feat, query_mask, None,
                    ctx_info["video_proposal_feat"], ctx_info["video_feat"],
                    clip_mask=ctx_info.get("clip_mask"),
                    return_query_feats=False, return_timing=False
                )
                        
            # In-place weighted sum on GPU, then offload to CPU to reduce VRAM
            clip_s.mul_(self.cfg['clip_scale_w'])
            frame_s.mul_(self.cfg['frame_scale_w'])
            clip_s.add_(frame_s)
            
            # If measuring, capture per-query end time after argsort/top-k for this batch
            if measure:
                try:
                    _sync()
                except Exception:
                    pass
                if timing_on_gpu:
                    # GPU sorting path for timing
                    try:
                        _sync()
                    except Exception:
                        pass
                    t_sort0 = time.perf_counter()
                    if timing_topk and timing_topk > 0 and timing_topk < clip_s.size(1):
                        _ = torch.topk(clip_s, k=timing_topk, dim=1, largest=True, sorted=True)
                    else:
                        _ = torch.argsort(clip_s, dim=1)
                    try:
                        _sync()
                    except Exception:
                        pass
                    t_sort1 = time.perf_counter()
                    t_end = t_sort1
                    if 'collect' in locals() and collect:
                        bs_local = clip_s.size(0)
                        sort_ms = (t_sort1 - t_sort0) * 1000.0
                        per_q_sort = sort_ms / max(1, bs_local)
                        argsort_ms.extend([per_q_sort] * bs_local)
                        similarity_end_times.extend([t_end] * bs_local)
                    # For metrics, still move to CPU once
                    scores_cpu = clip_s.detach().cpu()
                else:
                    # CPU sorting path for timing
                    scores_cpu = clip_s.detach().cpu()
                    t_sort0 = time.perf_counter()
                    if timing_topk and timing_topk > 0 and timing_topk < scores_cpu.size(1):
                        _ = torch.topk(scores_cpu, k=timing_topk, dim=1, largest=True, sorted=True)
                    else:
                        _ = torch.argsort(scores_cpu, dim=1)
                    try:
                        _sync()
                    except Exception:
                        pass
                    t_sort1 = time.perf_counter()
                    t_end = t_sort1
                    if 'collect' in locals() and collect:
                        bs_local = scores_cpu.size(0)
                        sort_ms = (t_sort1 - t_sort0) * 1000.0
                        per_q_sort = sort_ms / max(1, bs_local)
                        argsort_ms.extend([per_q_sort] * bs_local)
                        similarity_end_times.extend([t_end] * bs_local)
            # Reuse scores_cpu if available to avoid duplicate CPU copy
            if measure and 'scores_cpu' in locals():
                score_sum_list.append(scores_cpu)
            else:
                score_sum_list.append(clip_s.detach().cpu())
            del clip_s, frame_s
            
            # validation 루프에 추가
            if idx % 64 == 0:
                torch.cuda.empty_cache()

        score_sum = torch.cat(score_sum_list, dim=0)

        # 집계 리포트
        report = None
        if measure and per_query_ms:
            arr = np.array(per_query_ms, dtype=np.float64)
            report = {
                "per_query_ms": {
                    "avg": float(arr.mean()),
                    "p50": float(np.percentile(arr, 50)),
                    "p90": float(np.percentile(arr, 90)),
                    "p99": float(np.percentile(arr, 99)),
                    "count": int(arr.size),
                },
                "batch_sections_ms": {
                    "clip_avg": float(np.mean(per_clip_ms)),
                    "frame_avg": float(np.mean(per_frame_ms)),
                }
            }
            
            # Add encoding time average (excluding first query)
            if encoding_times:
                enc_arr = np.array(encoding_times, dtype=np.float64)
                report["encoding_normalize_avg_ms"] = float(enc_arr.mean())
            
            # Add similarity start/end times for cross-function timing
            if similarity_start_times:
                report["similarity_start_times"] = similarity_start_times
            if similarity_end_times:
                report["similarity_end_times"] = similarity_end_times
            report["timing_topk"] = int(timing_topk)
            report["timing_device"] = "gpu" if timing_on_gpu else "cpu"
            # Add argsort/top-k per-query timing stats
            if argsort_ms:
                s_arr = np.array(argsort_ms, dtype=np.float64)
                report["argsort_ms"] = {
                    "avg": float(s_arr.mean()),
                    "p50": float(np.percentile(s_arr, 50)),
                    "p90": float(np.percentile(s_arr, 90)),
                    "p99": float(np.percentile(s_arr, 99)),
                    "count": int(s_arr.size),
                }

        return score_sum, query_metas, report

    # def compute_context_info(self, model, context_dataloader):
        
    #     n_total_vid = len(context_dataloader.dataset)
    #     bsz = self.cfg['eval_context_bsz']
    #     metas = []  # list(dicts)
    #     vid_proposal_feat = []
    #     frame_feat, frame_mask = [], []
    #     for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
    #                            total=len(context_dataloader)):
            
    #         batch = gpu(batch) # transfer query batch to GPU
    #         metas.extend(batch[-1])
    #         clip_video_feat_ = batch[0] # (batch, 32, 3072)
    #         frame_video_feat_ = batch[1] # (batch, 111, 3072)
    #         # print(f"clip_video_feat shape: {clip_video_feat_.shape}")
    #         # print(f"frame_video_feat shape: {frame_video_feat_.shape}")
    #         frame_mask_ = batch[2]
    #         _frame_feat, _video_proposal_feat = model.encode_context(
    #             clip_video_feat_, frame_video_feat_, frame_mask_)

    #         frame_feat.append(_frame_feat)
    #         frame_mask.append(frame_mask_)

    #         vid_proposal_feat.append(_video_proposal_feat)

    #     log_cuda("context: before cat")
    #     vid_proposal_feat = torch.cat(vid_proposal_feat, dim=0)

    #     def cat_tensor(tensor_list):
    #         if len(tensor_list) == 0:
    #             return None
    #         else:
    #             seq_l = [e.shape[1] for e in tensor_list]
    #             b_sizes = [e.shape[0] for e in tensor_list]
    #             b_sizes_cumsum = np.cumsum([0] + b_sizes)
    #             if len(tensor_list[0].shape) == 3:
    #                 hsz = tensor_list[0].shape[2]
    #                 res_tensor = tensor_list[0].new_zeros(
    #                     sum(b_sizes), max(seq_l), hsz)
    #             elif len(tensor_list[0].shape) == 2:
    #                 res_tensor = tensor_list[0].new_zeros(
    #                     sum(b_sizes), max(seq_l))
    #             else:
    #                 raise ValueError("Only support 2/3 dimensional tensors")
    #             for i, e in enumerate(tensor_list):
    #                 res_tensor[b_sizes_cumsum[i]                               :b_sizes_cumsum[i+1], :seq_l[i]] = e
    #             return res_tensor

    #     # Concatenate once (frame features and masks)
    #     vid_feat_cat = cat_tensor(frame_feat)
    #     video_mask_cat = cat_tensor(frame_mask)
    #     log_cuda("context: after cat")

    #     # Optional: pre-normalize context once to reduce per-query work/VRAM
    #     pre_norm = bool(self.cfg.get('pre_normalize_context', False))
    #     if pre_norm:
    #         # In-place L2 normalization along the last dim to avoid extra allocations
    #         eps = 1e-12
    #         denom = torch.linalg.norm(vid_proposal_feat, dim=-1, keepdim=True).clamp_min_(eps)
    #         vid_proposal_feat = vid_proposal_feat.div(denom)
    #         denom2 = torch.linalg.norm(vid_feat_cat, dim=-1, keepdim=True).clamp_min_(eps)
    #         vid_feat_cat = vid_feat_cat.div(denom2)
    #         try:
    #             if hasattr(model, 'config'):
    #                 setattr(model.config, 'context_already_normalized', True)
    #         except Exception:
    #             pass
    #         log_cuda("context: after normalize")

    #     # Release references to shorten peak window
    #     try:
    #         del frame_feat
    #         del frame_mask
    #     except Exception:
    #         pass
    #     torch.cuda.empty_cache()
    #     log_cuda("context: after cleanup")

    #     return dict(
    #         video_metas=metas,  # list(dict) (N_videos)
    #         video_proposal_feat=vid_proposal_feat,
    #         video_feat=vid_feat_cat,
    #         video_mask=video_mask_cat
    #     )

    def compute_context_info(self, model, context_dataloader):
        # Streamed preallocation to avoid cat() peak memory
        n_total_vid = len(context_dataloader.dataset)
        metas = []
        start = 0
        vid_prop_bank = None  # (N, K, H)
        vid_frame_bank = None  # (N, H)
        clip_mask_bank = None  # (N, K)
        video_mask_cat = None  # unused downstream; keep minimal placeholder

        log_cuda("context: before cat")
        for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
                            total=len(context_dataloader)):
            batch = gpu(batch)
            metas.extend(batch[-1])
            clip_video_feat_ = batch[0]
            frame_video_feat_ = batch[1]
            frame_mask_ = batch[2]
            last_level_segments = batch[3] if len(batch) >= 6 else None

            _frame_feat, _video_proposal_feat, _clip_mask = model.encode_context(
                clip_video_feat_, frame_video_feat_, frame_mask_, last_level_segments=last_level_segments
            )
            # _frame_feat: (bs, H), _video_proposal_feat: (bs, K, H)

            bsz = _frame_feat.size(0)
            K = _video_proposal_feat.size(1)
            H = _video_proposal_feat.size(2)
            device = _video_proposal_feat.device
            dtype = _video_proposal_feat.dtype

            if vid_prop_bank is None:
                vid_prop_bank = torch.empty((n_total_vid, K, H), device=device, dtype=dtype)
                vid_frame_bank = torch.empty((n_total_vid, H), device=device, dtype=dtype)
                clip_mask_bank = torch.empty((n_total_vid, K), device=device, dtype=torch.float32)

            end = start + bsz
            vid_prop_bank[start:end, :, :] = _video_proposal_feat
            vid_frame_bank[start:end, :] = _frame_feat
            if _clip_mask is None:
                clip_mask_bank[start:end, :K] = 1.0
            else:
                clip_mask_bank[start:end, :K] = _clip_mask
            start = end

        log_cuda("context: after cat")

        # Optional: pre-normalize context once (in-place)
        pre_norm = bool(self.cfg.get('pre_normalize_context', False))
        if pre_norm:
            eps = 1e-12
            denom = torch.linalg.norm(vid_prop_bank, dim=-1, keepdim=True).clamp_min_(eps)
            vid_prop_bank.div_(denom)
            denom2 = torch.linalg.norm(vid_frame_bank, dim=-1, keepdim=True).clamp_min_(eps)
            vid_frame_bank.div_(denom2)
            try:
                if hasattr(model, 'config'):
                    setattr(model.config, 'context_already_normalized', True)
            except Exception:
                pass
            log_cuda("context: after normalize")

        torch.cuda.empty_cache()
        log_cuda("context: after cleanup")

        return dict(
            video_metas=metas,
            video_proposal_feat=vid_prop_bank,
            video_feat=vid_frame_bank,
            video_mask=video_mask_cat,
            clip_mask=clip_mask_bank
        )
