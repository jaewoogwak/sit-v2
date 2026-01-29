import json
import torch
import torch.utils.data as data
import numpy as np
import re
import h5py
import os
import pickle
import logging

_DEBUG_SHAPES_PRINTED = False


def _debug_shapes_enabled():
    return os.environ.get('GMMFORMER_DEBUG_SHAPES', '').strip() not in ('', '0', 'false', 'False')


def _shape_str(x):
    if torch.is_tensor(x):
        return f"torch {tuple(x.shape)} dtype={x.dtype} device={x.device}"
    if isinstance(x, np.ndarray):
        return f"np {x.shape} dtype={x.dtype}"
    if hasattr(x, 'shape'):
        return f"{type(x).__name__} shape={getattr(x, 'shape', None)}"
    return f"{type(x).__name__}"


def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def read_video_ids_limited(cap_file, limit):
    """Read up to `limit` unique video IDs from a caption file.
    Stops early once `limit` is reached to avoid scanning huge files.
    If limit <= 0, falls back to reading all IDs (same as read_video_ids).
    """
    if limit is None or int(limit) <= 0:
        return read_video_ids(cap_file)
    n = int(limit)
    video_ids_list = []
    seen = set()
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader:
            if len(video_ids_list) >= n:
                break
            parts = line.strip().split(' ', 1)
            if not parts:
                continue
            cap_id = parts[0]
            video_id = getVideoId(cap_id)
            if video_id not in seen:
                seen.add(video_id)
                video_ids_list.append(video_id)
    return video_ids_list

def average_to_fixed_length(visual_input, map_size):
    visual_input = torch.from_numpy(visual_input)
    num_sample_clips = map_size
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0).numpy()


    return new_visual_input

def uniform_feature_sampling(features, max_len):
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_features = []
    for i in range(max_len):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_features.append(np.mean(features[s_idx:e_idx], axis=0))
        else:
            new_features.append(features[s_idx])
    new_features = np.asarray(new_features)
    return new_features


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def split_frames_by_peaks(frames, peaks, return_bounds=False):
    """Split frames into segments using inclusive peak indices."""
    num_frames = frames.shape[0]
    if num_frames == 0:
        return [(0, 0)] if return_bounds else [frames]
    if not peaks:
        return [(0, num_frames)] if return_bounds else [frames]
    clean_peaks = sorted({int(p) for p in peaks if p is not None})
    clean_peaks = [p for p in clean_peaks if 0 <= p < num_frames]
    segments = []
    bounds = []
    start = 0
    for peak in clean_peaks:
        end = peak + 1
        if end <= start:
            continue
        segments.append(frames[start:end])
        bounds.append((start, end))
        start = end
    if start < num_frames:
        segments.append(frames[start:num_frames])
        bounds.append((start, num_frames))
    if not segments:
        segments = [frames]
        bounds = [(0, num_frames)]
    return bounds if return_bounds else segments


def _unique_bounds(bounds):
    unique = {(int(s), int(e)) for s, e in bounds if s is not None and e is not None and e > s}
    return sorted(unique, key=lambda x: (x[0], x[1]))


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


def _peaks_from_levels(levels, num_frames, level_num=None):
    peaks = []
    if not isinstance(levels, list):
        return peaks
    for level_entry in levels:
        if not isinstance(level_entry, dict):
            continue
        if level_num is not None and level_entry.get("level") != level_num:
            continue
        peaks.extend(_peaks_from_level_entry(level_entry, num_frames))
    return peaks


def _segments_from_levels(frames, levels, level_num=None):
    segments = []
    if not isinstance(levels, list):
        return segments
    num_frames = frames.shape[0]
    for level_entry in levels:
        if not isinstance(level_entry, dict):
            continue
        if level_num is not None and level_entry.get("level") != level_num:
            continue
        peaks = _peaks_from_level_entry(level_entry, num_frames)
        segments.extend(split_frames_by_peaks(frames, peaks))
    return segments


def _bounds_from_levels(frames, levels, level_num=None):
    bounds = []
    if not isinstance(levels, list):
        return bounds
    num_frames = frames.shape[0]
    for level_entry in levels:
        if not isinstance(level_entry, dict):
            continue
        if level_num is not None and level_entry.get("level") != level_num:
            continue
        peaks = _peaks_from_level_entry(level_entry, num_frames)
        bounds.extend(split_frames_by_peaks(frames, peaks, return_bounds=True))
    return bounds


def _segments_from_last_level(frames, levels):
    if not isinstance(levels, list) or not levels:
        return []
    level_nums = [
        level_entry.get("level")
        for level_entry in levels
        if isinstance(level_entry, dict) and isinstance(level_entry.get("level"), (int, float))
    ]
    if level_nums:
        max_level = max(level_nums)
        target_entries = [
            level_entry
            for level_entry in levels
            if isinstance(level_entry, dict) and level_entry.get("level") == max_level
        ]
    else:
        target_entries = [levels[-1]] if levels else []
    segments = []
    num_frames = frames.shape[0]
    for level_entry in target_entries:
        peaks = _peaks_from_level_entry(level_entry, num_frames)
        segments.extend(split_frames_by_peaks(frames, peaks))
    return segments


    """Split frames into segments using inclusive peak indices."""
    num_frames = frames.shape[0]
    if num_frames == 0:
        return [frames]
    if not peaks:
        return [frames]
    clean_peaks = sorted({int(p) for p in peaks if p is not None})
    clean_peaks = [p for p in clean_peaks if 0 <= p < num_frames]
    segments = []
    start = 0
    for peak in clean_peaks:
        end = peak + 1
        if end <= start:
            continue
        segments.append(frames[start:end])
        start = end
    if start < num_frames:
        segments.append(frames[start:num_frames])
    if not segments:
        segments = [frames]
    return segments


def collate_train(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    # print(f"data shape {data[0][0].shape}")
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    last_level_segments = None
    if len(data[0]) >= 9:
        clip_video_features, frame_video_features, captions, idxs, cap_ids, segment_bounds, cap_timestamps, last_level_segments, video_ids = zip(*data)
    elif len(data[0]) == 8:
        clip_video_features, frame_video_features, captions, idxs, cap_ids, video_ids, segment_bounds, cap_timestamps = zip(*data)
    elif len(data[0]) == 7:
        clip_video_features, frame_video_features, captions, idxs, cap_ids, last_level_segments, video_ids = zip(*data)
        segment_bounds, cap_timestamps = None, None
    else:
        clip_video_features, frame_video_features, captions, idxs, cap_ids, video_ids = zip(*data)
        segment_bounds, cap_timestamps = None, None

    # videos
    if isinstance(clip_video_features[0], (list, tuple)):
        clip_videos = list(clip_video_features)
    else:
        clip_videos = torch.cat(clip_video_features, dim=0).float()
    # print(f"clip videos: {clip_videos.shape}")

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = frame_video_features[0].shape[-1]  # 올바른 feature 차원 (512)
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        # print(f"frame_videos shape: {frame_videos.shape}")
        # print(f"frame shape: {frames.shape}")
        frame_videos[i, :end, :] = frames[:end, :]
        
        videos_mask[i, :end] = 1.0

    #captions
    feat_dim = captions[0][0].shape[-1]

    merge_captions = []
    all_lengths = []
    labels = []
    cap_ts = []
    cap_ts_mask = []

    for index, caps in enumerate(captions):
        labels.extend(index for i in range(len(caps)))
        all_lengths.extend(len(cap) for cap in caps)
        merge_captions.extend(cap for cap in caps)
        if cap_timestamps is not None:
            for ts in cap_timestamps[index]:
                if ts is None:
                    cap_ts.append((0.0, 0.0))
                    cap_ts_mask.append(0.0)
                else:
                    cap_ts.append((float(ts[0]), float(ts[1])))
                    cap_ts_mask.append(1.0)

    target = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
    words_mask = torch.zeros(len(all_lengths), max(all_lengths))

    for index, cap in enumerate(merge_captions):
        end = all_lengths[index]
        target[index, :end, :] = cap[:end, :]
        words_mask[index, :end] = 1.0



    batch = dict(
        clip_video_features=clip_videos,
        frame_video_features=frame_videos,
        videos_mask=videos_mask,
        text_feat=target,
        text_mask=words_mask,
        text_labels=labels,
    )
    if last_level_segments is not None:
        batch['last_level_segments'] = list(last_level_segments)

    if segment_bounds is not None:
        if isinstance(clip_video_features[0], (list, tuple)):
            max_segments = max(len(segs) for segs in clip_video_features)
        else:
            max_segments = max(len(b) for b in segment_bounds) if segment_bounds else 0
        bounds_tensor = torch.zeros(len(segment_bounds), max_segments, 2, dtype=torch.float32)
        bounds_mask = torch.zeros(len(segment_bounds), max_segments, dtype=torch.float32)
        for i, bounds in enumerate(segment_bounds):
            if not bounds:
                continue
            for j, (s, e) in enumerate(bounds):
                if j >= max_segments:
                    break
                bounds_tensor[i, j, 0] = float(s)
                bounds_tensor[i, j, 1] = float(e)
                bounds_mask[i, j] = 1.0
        batch['segment_bounds'] = bounds_tensor
        batch['segment_bounds_mask'] = bounds_mask

    if cap_timestamps is not None:
        batch['text_ts'] = torch.tensor(cap_ts, dtype=torch.float32) if cap_ts else torch.zeros((0, 2), dtype=torch.float32)
        batch['text_ts_mask'] = torch.tensor(cap_ts_mask, dtype=torch.float32) if cap_ts_mask else torch.zeros((0,), dtype=torch.float32)

    return batch


def collate_frame_val(data):
    last_level_segments = None
    if len(data[0]) == 5:
        clip_video_features, frame_video_features, last_level_segments, idxs, video_ids = zip(*data)
    else:
        clip_video_features, frame_video_features, idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    # videos
    if isinstance(clip_video_features[0], (list, tuple)):
        clip_videos = list(clip_video_features)
    else:
        clip_videos = torch.cat(clip_video_features, dim=0).float()

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = frame_video_features[0].shape[-1]  # 올바른 feature 차원 (512)
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    if last_level_segments is not None:
        return clip_videos, frame_videos, videos_mask, list(last_level_segments), idxs, video_ids
    return clip_videos, frame_videos, videos_mask, idxs, video_ids


def collate_text_val(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions,idxs, cap_ids = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths), captions[0].shape[-1])
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None


    return target, words_mask, idxs, cap_ids


class Dataset4PRVR(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat, text_feat_path, cfg, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        self.video2frames = video2frames

        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                if video_id in self.vid_caps:
                    self.vid_caps[video_id].append(cap_id)
                else:
                    self.vid_caps[video_id] = []
                    self.vid_caps[video_id].append(cap_id)
        self.visual_feat = visual_feat
        self.text_feat_path = text_feat_path

        self.map_size = cfg['map_size']
        self.max_ctx_len = cfg['max_ctx_l']
        self.max_desc_len = cfg['max_desc_l']

        self.open_file = False
        self.length = len(self.vid_caps)


    def __getitem__(self, index):

        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')

            self.open_file = True
        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        # video
        frame_list = self.video2frames[video_id]


        frame_vecs = []
        missing = 0
        for frame_id in frame_list:
            try:
                frame_vecs.append(self.visual_feat.read_one(frame_id))
            except Exception:
                missing += 1
                continue
        if len(frame_vecs) == 0:
            # 모든 프레임을 찾지 못한 경우 안전 가드로 영벡터 1개 사용
            try:
                dim = int(getattr(self.visual_feat, 'ndims', 512))
            except Exception:
                dim = 512
            frame_vecs = [np.zeros((dim,), dtype=np.float32)]

        global _DEBUG_SHAPES_PRINTED
        if (not _DEBUG_SHAPES_PRINTED) and _debug_shapes_enabled() and int(index) == 0:
            logging.getLogger().info(
                f"DEBUG[dataset raw] video_id={video_id} first_frame_feat={_shape_str(np.asarray(frame_vecs[0]))} "
                f"(frames_read={len(frame_vecs)} missing={missing})"
            )

        clip_video_feature = average_to_fixed_length(np.array(frame_vecs), self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)
        frame_video_feature = uniform_feature_sampling(np.array(frame_vecs), self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        # text
        cap_tensors = []
        for cap_id in cap_ids:

            cap_feat = self.text_feat[cap_id][...]

            if (not _DEBUG_SHAPES_PRINTED) and _debug_shapes_enabled() and int(index) == 0 and len(cap_tensors) == 0:
                logging.getLogger().info(
                    f"DEBUG[dataset raw] cap_id={cap_id} raw_text_feat={_shape_str(np.asarray(cap_feat))}"
                )
            
            # CLIP 텍스트 피쳐가 1차원인 경우 2차원으로 변환
            if cap_feat.ndim == 1:
                cap_feat = cap_feat.reshape(1, -1)  # (512,) -> (1, 512)

            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
            cap_tensors.append(cap_tensor)

        if (not _DEBUG_SHAPES_PRINTED) and _debug_shapes_enabled() and int(index) == 0:
            logging.getLogger().info(
                "DEBUG[dataset out] "
                f"clip_video_feature={_shape_str(clip_video_feature)} "
                f"frame_video_feature={_shape_str(frame_video_feature)} "
                f"cap_tensor0={_shape_str(cap_tensors[0]) if len(cap_tensors) else 'None'}"
            )
            _DEBUG_SHAPES_PRINTED = True

        return clip_video_feature, frame_video_feature, cap_tensors, index, cap_ids, video_id

    def __len__(self):
        return self.length


class TVRFramesDataset4PRVR(data.Dataset):
    """TVR dataset for per-video frame npy features + boundary-based segments."""

    def __init__(self, cap_file, visual_feat, text_feat_path, cfg, boundaries=None, release_map=None):
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        self.boundaries = boundaries or {}
        self.boundary_level = cfg.get('boundary_level', 'fine')
        self.dedupe_segments = bool(cfg.get('dedupe_segments', False))
        use_last_level_cfg = 'use_last_level_as_frame' in cfg
        self.use_last_level_as_frame = bool(cfg.get('use_last_level_as_frame', False))
        if not use_last_level_cfg:
            boundary_level = str(self.boundary_level).lower()
            if "+" in boundary_level and "fine" in boundary_level and "levels" in boundary_level:
                self.use_last_level_as_frame = True

        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                self.vid_caps.setdefault(video_id, []).append(cap_id)

        self.visual_feat = visual_feat
        self.text_feat_path = text_feat_path
        self.max_ctx_len = cfg['max_ctx_l']
        self.max_desc_len = cfg['max_desc_l']
        self.release_map = release_map or {}
        self.release_map_norm = {
            (vid, self._normalize_text(desc)): val
            for (vid, desc), val in self.release_map.items()
        }
        self.use_soft_mil = bool(cfg.get('use_soft_mil', False))
        self.soft_mil_sanity_max = int(cfg.get('soft_mil_sanity_max', 5000))
        self.open_file = False
        self.length = len(self.vid_caps)
        if self.use_soft_mil:
            self._sanity_check_release_map()

    def _load_frames(self, video_id):
        frames = self.visual_feat.read_video(video_id)
        if frames.ndim == 1:
            frames = frames.reshape(1, -1)
        if frames.shape[0] == 0:
            dim = frames.shape[-1] if frames.ndim > 1 else 512
            frames = np.zeros((1, dim), dtype=np.float32)
        return frames.astype(np.float32, copy=False)

    def _get_peaks_by_level(self, video_id, level, num_frames=None):
        entry = self.boundaries.get(video_id, {})
        if isinstance(entry, list):
            return entry
        if isinstance(entry, dict):
            token, level_num = _parse_level_token(level)
            if token in ("fine", "coarse"):
                if 'fine' in entry or 'coarse' in entry:
                    entry = entry.get(token, {}) or {}
                    return entry.get('peaks', [])
                if token == 'fine':
                    return entry.get('peaks', [])
                return []
            if token in ("levels", "level"):
                levels = entry.get("levels") or []
                num_frames = int(num_frames or 0)
                return _peaks_from_levels(levels, num_frames, level_num)
        return []

    @staticmethod
    def _normalize_text(text):
        if text is None:
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    def _sanity_check_release_map(self):
        if not self.release_map:
            logging.getLogger().warning("[soft_mil] release_map is empty; GT TS matching disabled.")
            return
        total = 0
        matched = 0
        for vid, cap_ids in self.vid_caps.items():
            for cap_id in cap_ids:
                if self.soft_mil_sanity_max > 0 and total >= self.soft_mil_sanity_max:
                    break
                cap_text = self.captions[cap_id]
                key = (vid, cap_text)
                if key in self.release_map:
                    matched += 1
                else:
                    norm_key = (vid, self._normalize_text(cap_text))
                    if norm_key in self.release_map_norm:
                        matched += 1
                total += 1
            if self.soft_mil_sanity_max > 0 and total >= self.soft_mil_sanity_max:
                break
        if total > 0:
            ratio = matched / float(total)
            logging.getLogger().info(
                f"[soft_mil] release_map match ratio: {matched}/{total} ({ratio:.2%})"
            )

    @staticmethod
    def _compute_segment_bounds(num_frames, peaks, duration):
        if duration is None or num_frames <= 0:
            return []
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
        scale = float(duration) / float(num_frames)
        return [(s * scale, e * scale) for s, e in bounds]

    def __getitem__(self, index):
        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')
            self.open_file = True

        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        frames = self._load_frames(video_id)
        frames = l2_normalize_np_array(frames)

        boundary_entry = self.boundaries.get(video_id, {})
        fine_peaks = None
        coarse_peaks = None
        clip_bounds = None
        if self.dedupe_segments:
            bounds = []
            if self.boundary_level == 'both':
                fine_peaks = self._get_peaks_by_level(video_id, 'fine', frames.shape[0])
                coarse_peaks = self._get_peaks_by_level(video_id, 'coarse', frames.shape[0])
                bounds.extend(split_frames_by_peaks(frames, fine_peaks, return_bounds=True))
                bounds.extend(split_frames_by_peaks(frames, coarse_peaks, return_bounds=True))
            elif '+' in self.boundary_level:
                tokens = [t.strip() for t in self.boundary_level.split('+') if t.strip()]
                for token in tokens:
                    token_name, level_num = _parse_level_token(token)
                    if token_name in ('fine', 'coarse'):
                        peaks = self._get_peaks_by_level(video_id, token_name, frames.shape[0])
                        bounds.extend(split_frames_by_peaks(frames, peaks, return_bounds=True))
                    elif token_name in ('levels', 'level'):
                        levels = boundary_entry.get('levels') if isinstance(boundary_entry, dict) else []
                        bounds.extend(_bounds_from_levels(frames, levels, level_num))
                if not bounds:
                    peaks = self._get_peaks_by_level(video_id, 'fine', frames.shape[0])
                    bounds = split_frames_by_peaks(frames, peaks, return_bounds=True)
            else:
                token_name, level_num = _parse_level_token(self.boundary_level)
                if token_name in ('levels', 'level'):
                    levels = boundary_entry.get('levels') if isinstance(boundary_entry, dict) else []
                    bounds = _bounds_from_levels(frames, levels, level_num)
                else:
                    peaks = self._get_peaks_by_level(video_id, self.boundary_level, frames.shape[0])
                    bounds = split_frames_by_peaks(frames, peaks, return_bounds=True)
            clip_bounds = _unique_bounds(bounds)
            if not clip_bounds:
                clip_bounds = [(0, frames.shape[0])]
            segments = [frames[s:e] for s, e in clip_bounds]
        else:
            if self.boundary_level == 'both':
                fine_peaks = self._get_peaks_by_level(video_id, 'fine', frames.shape[0])
                coarse_peaks = self._get_peaks_by_level(video_id, 'coarse', frames.shape[0])
                fine_segments = split_frames_by_peaks(frames, fine_peaks)
                coarse_segments = split_frames_by_peaks(frames, coarse_peaks)
                segments = fine_segments + coarse_segments
            elif '+' in self.boundary_level:
                tokens = [t.strip() for t in self.boundary_level.split('+') if t.strip()]
                segments = []
                for token in tokens:
                    token_name, level_num = _parse_level_token(token)
                    if token_name in ('fine', 'coarse'):
                        peaks = self._get_peaks_by_level(video_id, token_name, frames.shape[0])
                        segments.extend(split_frames_by_peaks(frames, peaks))
                    elif token_name in ('levels', 'level'):
                        levels = boundary_entry.get('levels') if isinstance(boundary_entry, dict) else []
                        segments.extend(_segments_from_levels(frames, levels, level_num))
                if not segments:
                    peaks = self._get_peaks_by_level(video_id, 'fine', frames.shape[0])
                    segments = split_frames_by_peaks(frames, peaks)
            else:
                token_name, level_num = _parse_level_token(self.boundary_level)
                if token_name in ('levels', 'level'):
                    levels = boundary_entry.get('levels') if isinstance(boundary_entry, dict) else []
                    segments = _segments_from_levels(frames, levels, level_num)
                else:
                    peaks = self._get_peaks_by_level(video_id, self.boundary_level, frames.shape[0])
                    segments = split_frames_by_peaks(frames, peaks)
        segment_tensors = [torch.from_numpy(seg) for seg in segments]

        last_level_segments = None
        if self.use_last_level_as_frame:
            levels = boundary_entry.get('levels') if isinstance(boundary_entry, dict) else []
            last_segments = _segments_from_last_level(frames, levels)
            if not last_segments:
                fine_peaks = self._get_peaks_by_level(video_id, 'fine', frames.shape[0])
                last_segments = split_frames_by_peaks(frames, fine_peaks)
            if not last_segments:
                last_segments = [frames]
            last_level_segments = [torch.from_numpy(seg) for seg in last_segments]

        cap_timestamps = None
        segment_bounds = None
        if self.use_soft_mil:
            cap_timestamps = []
            duration = None
            for cap_id in cap_ids:
                cap_text = self.captions[cap_id]
                release_entry = self.release_map.get((video_id, cap_text))
                if release_entry is None:
                    release_entry = self.release_map_norm.get((video_id, self._normalize_text(cap_text)))
                if release_entry is None:
                    cap_timestamps.append(None)
                    continue
                ts_start, ts_end, cap_duration = release_entry
                cap_timestamps.append((ts_start, ts_end))
                if duration is None:
                    duration = cap_duration

            if self.dedupe_segments and clip_bounds is not None:
                if duration is None or frames.shape[0] <= 0:
                    segment_bounds = []
                else:
                    scale = float(duration) / float(frames.shape[0])
                    segment_bounds = [(s * scale, e * scale) for s, e in clip_bounds]
            else:
                if self.boundary_level == 'both':
                    fine_bounds = self._compute_segment_bounds(frames.shape[0], fine_peaks, duration)
                    coarse_bounds = self._compute_segment_bounds(frames.shape[0], coarse_peaks, duration)
                    segment_bounds = fine_bounds + coarse_bounds
                elif '+' in self.boundary_level:
                    tokens = [t.strip() for t in self.boundary_level.split('+') if t.strip()]
                    segment_bounds = []
                    for token in tokens:
                        token_name, level_num = _parse_level_token(token)
                        if token_name in ('fine', 'coarse'):
                            peaks = self._get_peaks_by_level(video_id, token_name, frames.shape[0])
                            segment_bounds.extend(
                                self._compute_segment_bounds(frames.shape[0], peaks, duration)
                            )
                        elif token_name in ('levels', 'level'):
                            levels = boundary_entry.get('levels') if isinstance(boundary_entry, dict) else []
                            if isinstance(levels, list):
                                for level_entry in levels:
                                    if not isinstance(level_entry, dict):
                                        continue
                                    if level_num is not None and level_entry.get('level') != level_num:
                                        continue
                                    peaks = _peaks_from_level_entry(level_entry, frames.shape[0])
                                    segment_bounds.extend(
                                        self._compute_segment_bounds(frames.shape[0], peaks, duration)
                                    )
                else:
                    token_name, level_num = _parse_level_token(self.boundary_level)
                    if token_name in ('levels', 'level'):
                        segment_bounds = []
                        levels = boundary_entry.get('levels') if isinstance(boundary_entry, dict) else []
                        if isinstance(levels, list):
                            for level_entry in levels:
                                if not isinstance(level_entry, dict):
                                    continue
                                if level_num is not None and level_entry.get('level') != level_num:
                                    continue
                                peaks = _peaks_from_level_entry(level_entry, frames.shape[0])
                                segment_bounds.extend(
                                    self._compute_segment_bounds(frames.shape[0], peaks, duration)
                                )
                    else:
                        segment_bounds = self._compute_segment_bounds(frames.shape[0], peaks, duration)

        frame_video_feature = uniform_feature_sampling(frames, self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        cap_tensors = []
        for cap_id in cap_ids:
            cap_feat = self.text_feat[cap_id][...]
            if cap_feat.ndim == 1:
                cap_feat = cap_feat.reshape(1, -1)
            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
            cap_tensors.append(cap_tensor)

        if self.use_soft_mil:
            if self.use_last_level_as_frame:
                return segment_tensors, frame_video_feature, cap_tensors, index, cap_ids, segment_bounds, cap_timestamps, last_level_segments, video_id
            return segment_tensors, frame_video_feature, cap_tensors, index, cap_ids, video_id, segment_bounds, cap_timestamps
        if self.use_last_level_as_frame:
            return segment_tensors, frame_video_feature, cap_tensors, index, cap_ids, last_level_segments, video_id
        return segment_tensors, frame_video_feature, cap_tensors, index, cap_ids, video_id

    def __len__(self):
        return self.length


class Dataset4PRVRWithReader(data.Dataset):
    """Dataset4PRVR 변형: text_reader.get(cap_id) 방식 지원."""

    def __init__(self, cap_file, visual_feat, text_reader, cfg, video2frames=None):
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        self.video2frames = video2frames
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                self.vid_caps.setdefault(video_id, []).append(cap_id)

        self.visual_feat = visual_feat
        self.text_reader = text_reader
        self.map_size = cfg['map_size']
        self.max_ctx_len = cfg['max_ctx_l']
        self.max_desc_len = cfg['max_desc_l']
        self.length = len(self.vid_caps)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        frame_list = self.video2frames[video_id]
        frame_vecs = []
        missing = 0
        for frame_id in frame_list:
            try:
                frame_vecs.append(self.visual_feat.read_one(frame_id))
            except Exception:
                missing += 1
        if len(frame_vecs) == 0:
            try:
                dim = int(getattr(self.visual_feat, 'ndims', 512))
            except Exception:
                dim = 512
            frame_vecs = [np.zeros((dim,), dtype=np.float32)]

        global _DEBUG_SHAPES_PRINTED
        if (not _DEBUG_SHAPES_PRINTED) and _debug_shapes_enabled() and int(index) == 0:
            logging.getLogger().info(
                f"DEBUG[dataset raw] video_id={video_id} first_frame_feat={_shape_str(np.asarray(frame_vecs[0]))} "
                f"(frames_read={len(frame_vecs)} missing={missing})"
            )

        clip_video_feature = average_to_fixed_length(np.array(frame_vecs), self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)

        frame_video_feature = uniform_feature_sampling(np.array(frame_vecs), self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        cap_tensors = []
        for cap_id in cap_ids:
            cap_feat = self.text_reader.get(cap_id)
            if (not _DEBUG_SHAPES_PRINTED) and _debug_shapes_enabled() and int(index) == 0 and len(cap_tensors) == 0:
                logging.getLogger().info(
                    f"DEBUG[dataset raw] cap_id={cap_id} raw_text_feat={_shape_str(np.asarray(cap_feat))}"
                )
            if cap_feat.ndim == 1:
                cap_feat = cap_feat.reshape(1, -1)
            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
            cap_tensors.append(cap_tensor)

        if (not _DEBUG_SHAPES_PRINTED) and _debug_shapes_enabled() and int(index) == 0:
            logging.getLogger().info(
                "DEBUG[dataset out] "
                f"clip_video_feature={_shape_str(clip_video_feature)} "
                f"frame_video_feature={_shape_str(frame_video_feature)} "
                f"cap_tensor0={_shape_str(cap_tensors[0]) if len(cap_tensors) else 'None'}"
            )
            _DEBUG_SHAPES_PRINTED = True

        return clip_video_feature, frame_video_feature, cap_tensors, index, cap_ids, video_id


class VisDataSet4PRVR(data.Dataset):

    def __init__(self, visual_feat, video2frames, cfg, video_ids=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)
        self.map_size = cfg['map_size']
        self.max_ctx_len = cfg['max_ctx_l']
        # Pre-index required frame ids lazily to avoid loading huge id maps
        try:
            if hasattr(self.visual_feat, 'prepare_index') and self.video2frames is not None:
                target_names = []
                for vid in self.video_ids:
                    if vid in self.video2frames:
                        target_names.extend(self.video2frames[vid])
                # Deduplicate to reduce scan time
                if target_names:
                    self.visual_feat.prepare_index(set(target_names))
        except Exception:
            pass
    def __getitem__(self, index):
        video_id = self.video_ids[index]
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        missing = 0
        for frame_id in frame_list:
            try:
                frame_vecs.append(self.visual_feat.read_one(frame_id))
            except Exception:
                missing += 1
                continue
        if len(frame_vecs) == 0:
            try:
                dim = int(getattr(self.visual_feat, 'ndims', 512))
            except Exception:
                dim = 512
            frame_vecs = [np.zeros((dim,), dtype=np.float32)]
        clip_video_feature = average_to_fixed_length(np.array(frame_vecs), self.map_size)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)

        frame_video_feature = uniform_feature_sampling(np.array(frame_vecs), self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        return clip_video_feature, frame_video_feature, index, video_id

    def __len__(self):
        return self.length


class TVRFramesVisDataSet(data.Dataset):
    """TVR evaluation dataset for per-video frame npy features + boundary segments."""

    def __init__(self, visual_feat, cfg, video_ids=None, boundaries=None):
        self.visual_feat = visual_feat
        self.boundaries = boundaries or {}
        self.boundary_level = cfg.get('boundary_level', 'fine')
        self.dedupe_segments = bool(cfg.get('dedupe_segments', False))
        use_last_level_cfg = 'use_last_level_as_frame' in cfg
        self.use_last_level_as_frame = bool(cfg.get('use_last_level_as_frame', False))
        if not use_last_level_cfg:
            boundary_level = str(self.boundary_level).lower()
            if "+" in boundary_level and "fine" in boundary_level and "levels" in boundary_level:
                self.use_last_level_as_frame = True
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = list(self.boundaries.keys())
        self.length = len(self.video_ids)
        self.max_ctx_len = cfg['max_ctx_l']

    def _load_frames(self, video_id):
        frames = self.visual_feat.read_video(video_id)
        if frames.ndim == 1:
            frames = frames.reshape(1, -1)
        if frames.shape[0] == 0:
            dim = frames.shape[-1] if frames.ndim > 1 else 512
            frames = np.zeros((1, dim), dtype=np.float32)
        return frames.astype(np.float32, copy=False)

    def _get_peaks_by_level(self, video_id, level, num_frames=None):
        entry = self.boundaries.get(video_id, {})
        if isinstance(entry, list):
            return entry
        if isinstance(entry, dict):
            token, level_num = _parse_level_token(level)
            if token in ("fine", "coarse"):
                if 'fine' in entry or 'coarse' in entry:
                    entry = entry.get(token, {}) or {}
                    return entry.get('peaks', [])
                if token == 'fine':
                    return entry.get('peaks', [])
                return []
            if token in ("levels", "level"):
                levels = entry.get("levels") or []
                num_frames = int(num_frames or 0)
                return _peaks_from_levels(levels, num_frames, level_num)
        return []

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        frames = self._load_frames(video_id)
        frames = l2_normalize_np_array(frames)
        entry = self.boundaries.get(video_id, {})
        if self.dedupe_segments:
            bounds = []
            if self.boundary_level == 'both':
                if isinstance(entry, dict) and ('fine' in entry or 'coarse' in entry):
                    fine_peaks = entry.get('fine', {}).get('peaks', [])
                    coarse_peaks = entry.get('coarse', {}).get('peaks', [])
                elif isinstance(entry, dict):
                    fine_peaks = entry.get('peaks', [])
                    coarse_peaks = []
                elif isinstance(entry, list):
                    fine_peaks = entry
                    coarse_peaks = []
                else:
                    fine_peaks = []
                    coarse_peaks = []
                bounds.extend(split_frames_by_peaks(frames, fine_peaks, return_bounds=True))
                bounds.extend(split_frames_by_peaks(frames, coarse_peaks, return_bounds=True))
            elif '+' in self.boundary_level:
                tokens = [t.strip() for t in self.boundary_level.split('+') if t.strip()]
                for token in tokens:
                    token_name, level_num = _parse_level_token(token)
                    if token_name in ('fine', 'coarse'):
                        if isinstance(entry, dict) and ('fine' in entry or 'coarse' in entry):
                            peaks = entry.get(token_name, {}).get('peaks', [])
                        elif isinstance(entry, dict) and token_name == 'fine':
                            peaks = entry.get('peaks', [])
                        elif isinstance(entry, list) and token_name == 'fine':
                            peaks = entry
                        else:
                            peaks = []
                        bounds.extend(split_frames_by_peaks(frames, peaks, return_bounds=True))
                    elif token_name in ('levels', 'level'):
                        levels = entry.get('levels') if isinstance(entry, dict) else []
                        bounds.extend(_bounds_from_levels(frames, levels, level_num))
                if not bounds:
                    bounds = split_frames_by_peaks(frames, [], return_bounds=True)
            else:
                token_name, level_num = _parse_level_token(self.boundary_level)
                if token_name in ('levels', 'level'):
                    levels = entry.get('levels') if isinstance(entry, dict) else []
                    bounds = _bounds_from_levels(frames, levels, level_num)
                else:
                    if isinstance(entry, list):
                        peaks = entry
                    elif isinstance(entry, dict):
                        if 'fine' in entry or 'coarse' in entry:
                            entry = entry.get(self.boundary_level, {}) or {}
                        peaks = entry.get('peaks', [])
                    else:
                        peaks = []
                    bounds = split_frames_by_peaks(frames, peaks, return_bounds=True)
            bounds = _unique_bounds(bounds)
            if not bounds:
                bounds = [(0, frames.shape[0])]
            segments = [frames[s:e] for s, e in bounds]
        else:
            if self.boundary_level == 'both':
                if isinstance(entry, dict) and ('fine' in entry or 'coarse' in entry):
                    fine_peaks = entry.get('fine', {}).get('peaks', [])
                    coarse_peaks = entry.get('coarse', {}).get('peaks', [])
                elif isinstance(entry, dict):
                    fine_peaks = entry.get('peaks', [])
                    coarse_peaks = []
                elif isinstance(entry, list):
                    fine_peaks = entry
                    coarse_peaks = []
                else:
                    fine_peaks = []
                    coarse_peaks = []
                fine_segments = split_frames_by_peaks(frames, fine_peaks)
                coarse_segments = split_frames_by_peaks(frames, coarse_peaks)
                segments = fine_segments + coarse_segments
            elif '+' in self.boundary_level:
                tokens = [t.strip() for t in self.boundary_level.split('+') if t.strip()]
                segments = []
                for token in tokens:
                    token_name, level_num = _parse_level_token(token)
                    if token_name in ('fine', 'coarse'):
                        if isinstance(entry, dict) and ('fine' in entry or 'coarse' in entry):
                            peaks = entry.get(token_name, {}).get('peaks', [])
                        elif isinstance(entry, dict) and token_name == 'fine':
                            peaks = entry.get('peaks', [])
                        elif isinstance(entry, list) and token_name == 'fine':
                            peaks = entry
                        else:
                            peaks = []
                        segments.extend(split_frames_by_peaks(frames, peaks))
                    elif token_name in ('levels', 'level'):
                        levels = entry.get('levels') if isinstance(entry, dict) else []
                        segments.extend(_segments_from_levels(frames, levels, level_num))
                if not segments:
                    segments = split_frames_by_peaks(frames, [])
            else:
                token_name, level_num = _parse_level_token(self.boundary_level)
                if token_name in ('levels', 'level'):
                    levels = entry.get('levels') if isinstance(entry, dict) else []
                    segments = _segments_from_levels(frames, levels, level_num)
                else:
                    if isinstance(entry, list):
                        peaks = entry
                    elif isinstance(entry, dict):
                        if 'fine' in entry or 'coarse' in entry:
                            entry = entry.get(self.boundary_level, {}) or {}
                        peaks = entry.get('peaks', [])
                    else:
                        peaks = []
                    segments = split_frames_by_peaks(frames, peaks)
        segment_tensors = [torch.from_numpy(seg) for seg in segments]

        last_level_segments = None
        if self.use_last_level_as_frame:
            levels = entry.get('levels') if isinstance(entry, dict) else []
            last_segments = _segments_from_last_level(frames, levels)
            if not last_segments:
                fine_peaks = self._get_peaks_by_level(video_id, 'fine', frames.shape[0])
                last_segments = split_frames_by_peaks(frames, fine_peaks)
            if not last_segments:
                last_segments = [frames]
            last_level_segments = [torch.from_numpy(seg) for seg in last_segments]

        frame_video_feature = uniform_feature_sampling(frames, self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        if self.use_last_level_as_frame:
            return segment_tensors, frame_video_feature, last_level_segments, index, video_id
        return segment_tensors, frame_video_feature, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4PRVR(data.Dataset):
    """
    Load captions
    """

    def __init__(self, cap_file, text_feat_path, cfg):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.text_feat_path = text_feat_path
        self.max_desc_len = cfg['max_desc_l']
        self.open_file = False
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')

            self.open_file = True


        cap_feat = self.text_feat[cap_id][...]

        global _DEBUG_SHAPES_PRINTED
        if (not _DEBUG_SHAPES_PRINTED) and _debug_shapes_enabled() and int(index) == 0:
            logging.getLogger().info(
                f"DEBUG[text-only raw] cap_id={cap_id} raw_text_feat={_shape_str(np.asarray(cap_feat))}"
            )
        
        # CLIP 텍스트 피쳐가 1차원인 경우 2차원으로 변환
        if cap_feat.ndim == 1:
            cap_feat = cap_feat.reshape(1, -1)  # (512,) -> (1, 512)

        cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]

        if (not _DEBUG_SHAPES_PRINTED) and _debug_shapes_enabled() and int(index) == 0:
            logging.getLogger().info(f"DEBUG[text-only out] cap_tensor={_shape_str(cap_tensor)}")
            _DEBUG_SHAPES_PRINTED = True

        return cap_tensor, index, cap_id

    def __len__(self):
        return self.length


class TxtDataSet4PRVRWithReader(data.Dataset):
    """TxtDataSet4PRVR 변형: text_reader.get(cap_id) 방식 지원."""

    def __init__(self, cap_file, text_reader, cfg, video_ids=None, limit=0):
        self.captions = {}
        self.cap_ids = []
        target_videos = set(video_ids) if video_ids is not None else None
        cap_limit = max(0, int(limit or 0))

        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader:
                parts = line.strip().split(' ', 1)
                if len(parts) < 2:
                    continue
                cap_id, caption = parts
                if target_videos is not None:
                    vid = getVideoId(cap_id)
                    if vid not in target_videos:
                        continue
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                if cap_limit and len(self.cap_ids) >= cap_limit:
                    break
        self.text_reader = text_reader
        self.max_desc_len = cfg['max_desc_l']
        self.length = len(self.cap_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        try:
            cap_feat = self.text_reader.get(cap_id)
        except KeyError as e:
            print(f"[TxtDataSet4PRVRWithReader] missing text feature for {cap_id}", flush=True)
            raise KeyError(f"Text feature not found for {cap_id}") from e
        global _DEBUG_SHAPES_PRINTED
        if (not _DEBUG_SHAPES_PRINTED) and _debug_shapes_enabled() and int(index) == 0:
            logging.getLogger().info(
                f"DEBUG[text-only raw] cap_id={cap_id} raw_text_feat={_shape_str(np.asarray(cap_feat))}"
            )
        if cap_feat.ndim == 1:
            cap_feat = cap_feat.reshape(1, -1)
        cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
        if (not _DEBUG_SHAPES_PRINTED) and _debug_shapes_enabled() and int(index) == 0:
            logging.getLogger().info(f"DEBUG[text-only out] cap_tensor={_shape_str(cap_tensor)}")
            _DEBUG_SHAPES_PRINTED = True
        return cap_tensor, index, cap_id


class WebVidTxtDatasetForEval(data.Dataset):
    """WebVid text dataset supporting sharded HDF5 files (DL-DKD logic)."""

    def __init__(self, cap_file, text_feat_dir, cfg, video_ids=None, limit=None):
        self.captions = {}
        self.cap_ids = []

        cfg_limit = max(0, int(cfg.get('eval_query_limit', 0) or 0))
        if limit is None or int(limit) <= 0:
            limit = cfg_limit
        limit = max(0, int(limit or 0))
        vids = set(video_ids) if video_ids is not None else None
        total_lines = 0
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader:
                total_lines += 1
                parts = line.strip().split(' ', 1)
                if len(parts) < 2:
                    continue
                cap_id, caption = parts
                if vids is not None:
                    vid = cap_id.split('#', 1)[0]
                    if vid not in vids:
                        continue
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                if limit and len(self.cap_ids) >= limit:
                    break

        self.text_feat_dir = text_feat_dir
        self.max_desc_len = cfg['max_desc_l']
        self.length = len(self.cap_ids)

        # Lazy-opened shard files cache
        self._shard_files = {}

        # Build or load a cap_id -> (shard_name, chunk_name, key) index
        self.index_path = os.path.join(self.text_feat_dir, 'webvid_text_index.json')
        self._cap_index = None
        try:
            if os.path.exists(self.index_path):
                with open(self.index_path, 'r') as f:
                    self._cap_index = json.load(f)
            else:
                self._cap_index = self._build_cap_index()
        except Exception as e:
            print(f"[WebVidTxtDatasetForEval] Index unavailable ({e}); will fall back to shard scanning.")
            self._cap_index = None

    def _get_shard_file(self, shard_name):
        if shard_name not in self._shard_files:
            shard_path = os.path.join(self.text_feat_dir, shard_name)
            if os.path.exists(shard_path):
                self._shard_files[shard_name] = h5py.File(shard_path, 'r')
            else:
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
        return self._shard_files[shard_name]

    def _build_cap_index(self):
        index = {}
        shard_files = [fn for fn in os.listdir(self.text_feat_dir)
                       if fn.endswith('.hdf5') and 'shard' in fn]
        for shard_name in sorted(shard_files):
            shard_path = os.path.join(self.text_feat_dir, shard_name)
            try:
                f = h5py.File(shard_path, 'r')
            except Exception:
                continue
            try:
                for chunk_name in f.keys():
                    if not chunk_name.startswith('chunk_'):
                        continue
                    grp = f[chunk_name]
                    for ds_key in grp.keys():
                        index[ds_key] = (shard_name, chunk_name, ds_key)
            finally:
                try:
                    f.close()
                except Exception:
                    pass
        return index

    def _find_feature_in_shards(self, cap_id):
        if '#' in cap_id:
            vid_part, cap_part = cap_id.split('#', 1)
            search_key = f"{vid_part}_{cap_part}"
        else:
            search_key = cap_id

        if self._cap_index is not None and search_key in self._cap_index:
            shard_name, chunk_name, ds_key = self._cap_index[search_key]
            shard_file = self._get_shard_file(shard_name)
            return shard_file[chunk_name][ds_key][...]

        shard_files = [fn for fn in os.listdir(self.text_feat_dir)
                       if fn.endswith('.hdf5') and 'shard' in fn]
        for shard_name in sorted(shard_files):
            shard_file = self._get_shard_file(shard_name)
            for chunk_name in shard_file.keys():
                if chunk_name.startswith('chunk_'):
                    chunk = shard_file[chunk_name]
                    if search_key in chunk:
                        return chunk[search_key][...]
        raise KeyError(f"Feature not found for {cap_id} (searched as {search_key})")

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]

        cap_feat = self._find_feature_in_shards(cap_id)
        cap_feat_normalized = l2_normalize_np_array(cap_feat)
        cap_tensor = torch.from_numpy(cap_feat_normalized).squeeze()

        if len(cap_tensor.shape) == 1:
            cap_tensor = cap_tensor.unsqueeze(0)

        cap_tensor = cap_tensor[:self.max_desc_len]

        return cap_tensor, index, cap_id

    def __len__(self):
        return self.length

    def __del__(self):
        for shard_file in self._shard_files.values():
            try:
                shard_file.close()
            except Exception:
                pass
if __name__ == '__main__':
    pass
