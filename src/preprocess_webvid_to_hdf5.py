#!/usr/bin/env python3
"""
Convert WebVid PT shards (video/text/index) into GMMFormer-compatible HDF5 and caption files.

Target layout under <out_root>/webvid:
  FeatureData/new_clip_vit_32_webvid_vid_features.hdf5
  TextData/clip_ViT_B_32_webvid_query_feat.hdf5
  TextData/webvidtrain.caption.txt
  TextData/webvidtest.caption.txt

Notes:
- This script expects PyTorch to be installed in your runtime.
- PT shards are large; consider using --max_videos / --max_caps to create a subset for initial experiments.
- Index shards are used to map caption IDs to video IDs. If your index format differs, adapt `build_cap2vid_from_index`.
"""

import os
import re
import gc
import h5py
import glob
import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple, List, Optional

import torch


def list_shards(base: Path, split: str, kind: str) -> List[Path]:
    pat = str(base / f"{split}_{kind}_*.pt")
    files = sorted(Path(p) for p in glob.glob(pat))
    return files


def build_cap2vid_from_index(index_shards: List[Path]) -> Dict[str, str]:
    """Build mapping cap_id -> video_id by reading index shards.

    Supported index formats per shard:
      1) dict with keys 'cap_to_vid' or 'cap_to_video'
      2) dict mapping cap_id -> video_id directly
      3) list/tuple of dicts with keys {'cap_id','video_id'}

    Returns: dict {cap_id: video_id}
    """
    cap2vid: Dict[str, str] = {}
    for p in index_shards:
        obj = torch.load(p, map_location='cpu')

        if isinstance(obj, dict):
            # Case 1: nested mapping
            for key in ['cap_to_vid', 'cap_to_video', 'cap2vid', 'cap_vid']:
                if key in obj and isinstance(obj[key], dict):
                    cap2vid.update(obj[key])
                    break
            else:
                # Case 2: assume direct mapping cap_id -> video_id
                # Heuristic: value must be str-like
                sample_val = next(iter(obj.values())) if obj else None
                if isinstance(sample_val, str):
                    cap2vid.update({str(k): str(v) for k, v in obj.items()})
                else:
                    raise ValueError(f"Unrecognized index shard format: {p}")
        elif isinstance(obj, (list, tuple)):
            # Case 3: list of dict entries
            for it in obj:
                if isinstance(it, dict):
                    cid = it.get('cap_id') or it.get('caption_id') or it.get('cid')
                    vid = it.get('video_id') or it.get('vid')
                    if cid is not None and vid is not None:
                        cap2vid[str(cid)] = str(vid)
                else:
                    raise ValueError(f"Index list element is not dict in {p}")
        else:
            # Support tensor index format downstream in numeric pipeline
            raise ValueError(f"Unsupported index shard type in {p}: {type(obj)}")

    return cap2vid


def ensure_dirs(base_out: Path):
    (base_out / 'FeatureData').mkdir(parents=True, exist_ok=True)
    (base_out / 'TextData').mkdir(parents=True, exist_ok=True)


def write_caption_file(path: Path, cap_ids: Iterable[str]):
    with open(path, 'w') as f:
        for cid in cap_ids:
            # The caption string is unused by the loader; write a placeholder
            f.write(f"{cid} dummy\n")


def normalize_id(s: str) -> str:
    """Ensure HDF5 path-safe key (avoid slashes)."""
    return s.replace('/', '_').replace('\\', '_')


def merge_text_shards_to_hdf5(text_shards: List[Path], out_h5: Path,
                               cap_whitelist: set = None, max_caps: int = None,
                               rename_map: Dict[str, str] = None) -> int:
    """Merge text feature shards into a single HDF5 with one dataset per cap_id.

    If rename_map is provided, keys will be remapped (e.g., to `video_id#cap_id`).
    """
    n_written = 0
    with h5py.File(out_h5, 'w') as hf:
        for shard in text_shards:
            data = torch.load(shard, map_location='cpu')
            if not isinstance(data, dict):
                raise ValueError(f"Unexpected text shard format: {shard}")
            for cid, feat in data.items():
                if cap_whitelist is not None and cid not in cap_whitelist:
                    continue
                cid_out = rename_map.get(cid, cid) if rename_map else cid
                cid_norm = normalize_id(str(cid_out))
                arr = feat.detach().cpu().numpy()
                # Accept (512,) or (1,512) or (T,512). We store as-is.
                hf.create_dataset(cid_norm, data=arr, compression="gzip")
                n_written += 1
                if max_caps is not None and n_written >= max_caps:
                    return n_written
            # free memory
            del data
            gc.collect()
    return n_written


def merge_video_shards_to_hdf5(video_shards: List[Path], out_h5: Path,
                                vid_whitelist: set = None, max_videos: int = None) -> int:
    """Merge video feature shards into a single HDF5 with one dataset per video_id."""
    n_written = 0
    with h5py.File(out_h5, 'w') as hf:
        for shard in video_shards:
            data = torch.load(shard, map_location='cpu')
            if not isinstance(data, dict):
                raise ValueError(f"Unexpected video shard format: {shard}")
            for vid, feat in data.items():
                if vid_whitelist is not None and vid not in vid_whitelist:
                    continue
                vid_norm = normalize_id(str(vid))
                arr = feat.detach().cpu().numpy()
                # Expect (T,512) but store as-is
                hf.create_dataset(vid_norm, data=arr, compression="gzip")
                n_written += 1
                if max_videos is not None and n_written >= max_videos:
                    return n_written
            del data
            gc.collect()
    return n_written


# ===== Numeric pipeline for Tensor-based shards =====

def load_tensor(path: Path) -> torch.Tensor:
    t = torch.load(path, map_location='cpu')
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"Expected tensor at {path}, got {type(t)}")
    return t


def get_tensor_shard_sizes(paths: List[Path]) -> List[int]:
    sizes = []
    for idx, p in enumerate(paths):
        t = load_tensor(p)
        if t.ndim == 1:
            sizes.append(int(t.shape[0]))
        else:
            sizes.append(int(t.shape[0]))
        if paths:
            if (idx + 1) % 10 == 0 or (idx + 1) == len(paths):
                print(f"    processed {idx + 1}/{len(paths)} tensor shards (size={sizes[-1]})")
    return sizes


def build_offsets(sizes: List[int]) -> List[Tuple[int, int]]:
    offsets = []
    cur = 0
    for n in sizes:
        offsets.append((cur, cur + n))
        cur += n
    return offsets


def locate_index(offsets: List[Tuple[int, int]], idx: int) -> Tuple[int, int]:
    for shard_idx, (s, e) in enumerate(offsets):
        if s <= idx < e:
            return shard_idx, idx - s
    raise IndexError(f"Global index {idx} out of range")


def write_text_by_indices_tensor(text_shards: List[Path], text_offsets: List[Tuple[int,int]],
                                 out_h5: Path,
                                 cap_indices: List[int],
                                 cap_to_vid: Dict[int, int]) -> int:
    n = 0
    with h5py.File(out_h5, 'w') as hf:
        cache_idx = -1
        cache_t = None
        for cap_idx in cap_indices:
            shard_idx, local = locate_index(text_offsets, cap_idx)
            if shard_idx != cache_idx:
                cache_t = load_tensor(text_shards[shard_idx])
                cache_idx = shard_idx
            feat = cache_t[local]
            vid_idx = cap_to_vid[cap_idx]
            key = f"vid_{vid_idx}#cap_{cap_idx}"
            nkey = normalize_id(key)
            if nkey in hf:
                continue
            hf.create_dataset(nkey, data=feat.detach().cpu().numpy(), compression='gzip')
            n += 1
    return n


def write_video_by_indices_tensor(video_shards: List[Path], video_offsets: List[Tuple[int,int]],
                                  out_h5: Path,
                                  vid_indices: List[int]) -> int:
    n = 0
    with h5py.File(out_h5, 'w') as hf:
        cache_idx = -1
        cache_t = None
        for vid_idx in vid_indices:
            shard_idx, local = locate_index(video_offsets, vid_idx)
            if shard_idx != cache_idx:
                cache_t = load_tensor(video_shards[shard_idx])
                cache_idx = shard_idx
            feat = cache_t[local]
            key = f"vid_{vid_idx}"
            hf.create_dataset(normalize_id(key), data=feat.detach().cpu().numpy(), compression='gzip')
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Convert WebVid PT shards to GMMFormer HDF5")
    ap.add_argument('--input_dir', type=str, default='/home/jaewoo/webvid/output', help='Path to PT shards dir')
    ap.add_argument('--out_root', type=str, default='/home/jaewoo', help='Output root (will create webvid/ subfolders)')
    ap.add_argument('--max_videos', type=int, default=None, help='Limit number of videos (train split) for subset')
    ap.add_argument('--caps_per_video', type=int, default=5, help='Number of captions per selected train video')
    ap.add_argument('--max_caps', type=int, default=None, help='Legacy: limit total captions (train). If set with max_videos, caps_per_video takes precedence')
    ap.add_argument('--test_max_videos', type=int, default=None, help='Limit number of videos (test split) for subset')
    ap.add_argument('--test_caps_per_video', type=int, default=5, help='Number of captions per selected test video')
    ap.add_argument('--split', type=str, default='both', choices=['train','test','both'], help='Which splits to convert')
    ap.add_argument('--videos_per_h5', type=int, default=None, help='If set, shard videos across multiple HDF5 files of this many videos each')
    ap.add_argument('--caps_per_h5', type=int, default=None, help='If set, shard captions across multiple HDF5 files of this many caps each')
    ap.add_argument('--use_shard_local_pairs', action='store_true', help='Treat each shard as locally aligned (text[i] pairs video[i]) and build global IDs across all shards')
    args = ap.parse_args()

    base = Path(args.input_dir)
    out = Path(args.out_root) / 'webvid'
    ensure_dirs(out)

    print(f"[webvid-preprocess] input_dir={base}")
    print(f"[webvid-preprocess] out_root={out}")

    # Gather shards
    train_text = list_shards(base, 'train', 'text')
    train_video = list_shards(base, 'train', 'video')
    train_index = list_shards(base, 'train', 'index')

    test_text = list_shards(base, 'test', 'text')
    test_video = list_shards(base, 'test', 'video')
    test_index = list_shards(base, 'test', 'index')

    print(f"[webvid-preprocess] found train shards - text:{len(train_text)} video:{len(train_video)} index:{len(train_index)}")
    print(f"[webvid-preprocess] found test shards  - text:{len(test_text)} video:{len(test_video)} index:{len(test_index)}")

    # Helper: ensure shard counts match for local-pair mode
    def _check_match(ts, vs, split):
        if len(ts) != len(vs):
            raise RuntimeError(f"text/video shard count mismatch for {split}: {len(ts)} vs {len(vs)}")

    # Streaming manifest writer (avoid large in-memory dicts)
    class StreamingJSONMap:
        def __init__(self, path: Path):
            self.path = path
            path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, 'w')
            self.first = True
            self.f.write('{' + '\n')
        def add(self, key: str, value: str):
            line = f'"{key}": "{value}"'
            if not self.first:
                self.f.write(',\n')
            self.f.write(line)
            self.first = False
        def close(self):
            self.f.write('\n}\n')
            self.f.close()

    # Local-shard aligned processing
    if args.use_shard_local_pairs:
        if not args.videos_per_h5 or not args.caps_per_h5:
            raise ValueError('--use_shard_local_pairs requires --videos_per_h5 and --caps_per_h5 to control shard sizes')
        _check_match(train_text, train_video, 'train')
        _check_match(test_text, test_video, 'test')

        # Prepare writers
        video_manifest_path = out / 'FeatureData' / 'video_manifest.json'
        text_manifest_path = out / 'TextData' / 'text_manifest.json'
        vman = StreamingJSONMap(video_manifest_path)
        tman = StreamingJSONMap(text_manifest_path)

        cap_train_path = out / 'TextData' / 'webvidtrain.caption.txt'
        cap_test_path = out / 'TextData' / 'webvidtest.caption.txt'
        cap_train_f = open(cap_train_path, 'w')
        cap_test_f = open(cap_test_path, 'w')

        # Video/text HDF5 shard state
        cur_vid_count = 0
        cur_vid_shard = 0
        vid_h5 = None

        def _ensure_vid_h5():
            nonlocal vid_h5, cur_vid_count, cur_vid_shard
            if vid_h5 is None or (cur_vid_count % args.videos_per_h5) == 0:
                if vid_h5 is not None:
                    vid_h5.close()
                vfile = out / 'FeatureData' / f"video_shard_{cur_vid_shard:03d}.hdf5"
                vid_h5 = h5py.File(vfile, 'w')
                cur_vid_shard += 1

        cur_txt_count = 0
        cur_txt_shard = 0
        txt_h5 = None

        def _ensure_txt_h5():
            nonlocal txt_h5, cur_txt_count, cur_txt_shard
            if txt_h5 is None or (cur_txt_count % args.caps_per_h5) == 0:
                if txt_h5 is not None:
                    txt_h5.close()
                tfile = out / 'TextData' / f"text_shard_{cur_txt_shard:03d}.hdf5"
                txt_h5 = h5py.File(tfile, 'w')
                cur_txt_shard += 1

        # Global counters
        global_vid = 0
        global_cap = 0

        def process_split(text_shards, video_shards, cap_file, split_name):
            nonlocal cur_vid_count, cur_txt_count, global_vid, global_cap
            for si, (tp, vp) in enumerate(zip(text_shards, video_shards)):
                t = load_tensor(tp)
                v = load_tensor(vp)
                if t.ndim != 2:
                    raise ValueError(f"text shard {tp} expected 2D (N,512), got {tuple(t.shape)}")
                if v.ndim != 3:
                    raise ValueError(f"video shard {vp} expected 3D (N,F,512), got {tuple(v.shape)}")
                n = min(t.shape[0], v.shape[0])
                if n <= 0:
                    continue
                # Per-shard cache
                for i in range(n):
                    _ensure_vid_h5(); _ensure_txt_h5()
                    vid_key = f"vid_{global_vid}"
                    cap_key = f"{vid_key}#cap_{global_cap}"
                    # Write video
                    if vid_key not in vid_h5:
                        vid_h5.create_dataset(normalize_id(vid_key), data=v[i].detach().cpu().numpy(), compression='gzip')
                        rel_v = os.path.relpath(vid_h5.filename, out / 'FeatureData')
                        vman.add(vid_key, rel_v)
                        cur_vid_count += 1
                    # Write text
                    nkey = normalize_id(cap_key)
                    if nkey not in txt_h5:
                        txt_h5.create_dataset(nkey, data=t[i].detach().cpu().numpy(), compression='gzip')
                        rel_t = os.path.relpath(txt_h5.filename, out / 'TextData')
                        tman.add(cap_key, rel_t)
                        cur_txt_count += 1
                    # Caption line
                    cap_file.write(f"{cap_key} dummy\n")
                    # advance
                    global_vid += 1
                    global_cap += 1

        # Process requested splits
        if args.split in ('train','both'):
            print(f"[local-pairs] Processing train shards: {len(train_text)}")
            process_split(train_text, train_video, cap_train_f, 'train')
        if args.split in ('test','both'):
            print(f"[local-pairs] Processing test shards: {len(test_text)}")
            process_split(test_text, test_video, cap_test_f, 'test')

        # Close files
        if vid_h5 is not None: vid_h5.close()
        if txt_h5 is not None: txt_h5.close()
        cap_train_f.close(); cap_test_f.close()
        vman.close(); tman.close()

        # Write compact meta for index-based routing
        import json as _json
        with open(out / 'FeatureData' / 'video_meta.json', 'w') as f:
            _json.dump({
                'kind': 'video',
                'pattern': 'video_shard_%03d.hdf5',
                'per_shard': args.videos_per_h5,
                'total': global_vid
            }, f)
        with open(out / 'TextData' / 'text_meta.json', 'w') as f:
            _json.dump({
                'kind': 'text',
                'pattern': 'text_shard_%03d.hdf5',
                'per_shard': args.caps_per_h5,
                'total': global_cap
            }, f)

        print("Done. Outputs written under:")
        print(f"  {out}")
        return

    # Detect shard formats
    def detect_mode(sample_path: Optional[Path]) -> str:
        if not sample_path:
            return 'none'
        obj = torch.load(sample_path, map_location='cpu')
        if isinstance(obj, dict):
            return 'dict'
        elif isinstance(obj, torch.Tensor):
            return 'tensor'
        elif isinstance(obj, list):
            return 'list'
        else:
            return type(obj).__name__

    text_mode = detect_mode(train_text[0] if train_text else (test_text[0] if test_text else None))
    video_mode = detect_mode(train_video[0] if train_video else (test_video[0] if test_video else None))
    index_mode = detect_mode(train_index[0] if train_index else (test_index[0] if test_index else None))

    # Pipeline A: dict-based mapping via explicit cap_id -> video_id
    cap2vid_train: Dict[str, str] = {}
    cap2vid_test: Dict[str, str] = {}
    numeric_pipeline = False
    if index_mode == 'tensor':
        numeric_pipeline = True
        print('[webvid-preprocess] Detected tensor-based shards; using numeric pipeline')
    else:
        print('[webvid-preprocess] Detected dict-based shards; using manifest pipeline')
        if args.split in ('train','both'):
            if not train_index:
                raise FileNotFoundError("No train_index_*.pt shards found; cannot map captions to videos.")
            cap2vid_train = build_cap2vid_from_index(train_index)
        if args.split in ('test','both'):
            if not test_index:
                raise FileNotFoundError("No test_index_*.pt shards found; cannot map captions to videos.")
            cap2vid_test = build_cap2vid_from_index(test_index)

    # Compute whitelists with strong pairing guarantees
    def build_video_to_caps(cap2vid: Dict[str, str]) -> Dict[str, List[str]]:
        v2c: Dict[str, List[str]] = {}
        for cid, vid in cap2vid.items():
            v2c.setdefault(str(vid), []).append(str(cid))
        return v2c

    train_v2c = build_video_to_caps(cap2vid_train) if cap2vid_train else {}
    test_v2c = build_video_to_caps(cap2vid_test) if cap2vid_test else {}

    # Select train videos
    if args.max_videos and train_v2c:
        selected_train_vids = list(sorted(train_v2c.keys()))[:args.max_videos]
    else:
        selected_train_vids = list(sorted(train_v2c.keys())) if train_v2c else []

    # Select captions per selected train video
    cap_whitelist = set()
    for vid in selected_train_vids:
        caps = train_v2c.get(vid, [])
        if args.caps_per_video and args.caps_per_video > 0:
            caps = caps[:args.caps_per_video]
        cap_whitelist.update(caps)

    # Fallback: legacy max_caps without max_videos
    if not selected_train_vids and args.max_caps and cap2vid_train:
        for cid in list(cap2vid_train.keys())[:args.max_caps]:
            cap_whitelist.add(cid)
        selected_train_vids = sorted(set(cap2vid_train[cid] for cid in cap_whitelist))

    vid_whitelist = set(selected_train_vids)

    # Test selection
    if args.test_max_videos and test_v2c:
        selected_test_vids = list(sorted(test_v2c.keys()))[:args.test_max_videos]
    else:
        selected_test_vids = list(sorted(test_v2c.keys())) if test_v2c else []

    test_cap_whitelist = set()
    for vid in selected_test_vids:
        caps = test_v2c.get(vid, [])
        if args.test_caps_per_video and args.test_caps_per_video > 0:
            caps = caps[:args.test_caps_per_video]
        test_cap_whitelist.update(caps)

    # Prepare cap id rename maps so dataset can split by '#'
    # New cap id format: f"{video_id}#{original_cap_id}"
    rename_train = {cid: f"{normalize_id(str(vid))}#{cid}" for cid, vid in cap2vid_train.items()}
    rename_test = {cid: f"{normalize_id(str(vid))}#{cid}" for cid, vid in cap2vid_test.items()}

    # Merge depending on pipeline
    text_h5 = out / 'TextData' / 'clip_ViT_B_32_webvid_query_feat.hdf5'
    vid_h5 = out / 'FeatureData' / 'new_clip_vit_32_webvid_vid_features.hdf5'

    if numeric_pipeline:
        # Validate text/video shards are tensors
        if text_mode != 'tensor' or video_mode != 'tensor':
            raise RuntimeError(f"Index is tensor but text/video shards are not tensors (text={text_mode}, video={video_mode}). Please share shard schema or adjust loader.")

        print('[webvid-preprocess] computing tensor shard sizes (train) ...')
        text_sizes = get_tensor_shard_sizes(train_text)
        video_sizes = get_tensor_shard_sizes(train_video)
        print('[webvid-preprocess] computing tensor shard sizes (test) ...')
        test_text_sizes = get_tensor_shard_sizes(test_text) if test_text else []
        test_video_sizes = get_tensor_shard_sizes(test_video) if test_video else []
        text_offsets = build_offsets(text_sizes)
        video_offsets = build_offsets(video_sizes)
        test_text_offsets = build_offsets(test_text_sizes) if test_text_sizes else []
        test_video_offsets = build_offsets(test_video_sizes) if test_video_sizes else []

        # Read all index pairs and determine column roles
        def load_pairs(paths: List[Path], split_name: str) -> torch.Tensor:
            arrs = []
            for idx, p in enumerate(paths):
                t = load_tensor(p)
                t = t.long()
                t = t.view(-1, t.shape[-1])
                if t.shape[1] < 2:
                    raise ValueError(f"Index tensor at {p} must have at least 2 columns")
                arrs.append(t[:, :2])
                if (idx + 1) % 10 == 0 or (idx + 1) == len(paths):
                    print(f"    [{split_name}] loaded {idx + 1}/{len(paths)} index shards (rows={t.shape[0]})")
            return torch.cat(arrs, dim=0) if arrs else torch.empty(0, 2, dtype=torch.long)

        train_pairs = load_pairs(train_index, 'train') if train_index else torch.empty(0, 2, dtype=torch.long)
        test_pairs = load_pairs(test_index, 'test') if test_index else torch.empty(0, 2, dtype=torch.long)

        print(f"[webvid-preprocess] total train caps={sum(text_sizes)} videos={sum(video_sizes)}")
        if test_text_sizes or test_video_sizes:
            print(f"[webvid-preprocess] total test caps={sum(test_text_sizes)} videos={sum(test_video_sizes)}")

        # Decide which column is cap vs vid by comparing maxima
        total_caps = sum(text_sizes)
        total_vids = sum(video_sizes)
        max0 = int(train_pairs[:, 0].max().item()) if train_pairs.numel() else -1
        max1 = int(train_pairs[:, 1].max().item()) if train_pairs.numel() else -1

        # Score both assignments
        s01 = abs((max0 + 1) - total_caps) + abs((max1 + 1) - total_vids)
        s10 = abs((max1 + 1) - total_caps) + abs((max0 + 1) - total_vids)
        cap_col, vid_col = (0, 1) if s01 <= s10 else (1, 0)

        # Build mappings and selections
        from collections import defaultdict
        v2caps = defaultdict(list)
        cap2vid_num = {}
        for row in train_pairs.tolist():
            cap_idx = int(row[cap_col])
            vid_idx = int(row[vid_col])
            if args.max_videos and len(v2caps) >= args.max_videos and vid_idx not in v2caps:
                continue
            if args.caps_per_video and len(v2caps[vid_idx]) >= args.caps_per_video:
                continue
            v2caps[vid_idx].append(cap_idx)
            cap2vid_num[cap_idx] = vid_idx

        selected_vids = list(v2caps.keys())
        selected_caps = [c for caps in v2caps.values() for c in caps]
        # Deduplicate caps while preserving order
        selected_caps = list(dict.fromkeys(selected_caps))

        print(f"[webvid-preprocess] numeric pipeline selections: train_vids={len(selected_vids)} train_caps={len(selected_caps)}")

        # Test selections
        v2caps_test = defaultdict(list)
        cap2vid_num_test = {}
        for row in test_pairs.tolist():
            cap_idx = int(row[cap_col])
            vid_idx = int(row[vid_col])
            if args.test_max_videos and len(v2caps_test) >= args.test_max_videos and vid_idx not in v2caps_test:
                continue
            if args.test_caps_per_video and len(v2caps_test[vid_idx]) >= args.test_caps_per_video:
                continue
            v2caps_test[vid_idx].append(cap_idx)
            cap2vid_num_test[cap_idx] = vid_idx

        selected_test_vids = list(v2caps_test.keys())
        selected_test_caps = [c for caps in v2caps_test.values() for c in caps]
        selected_test_caps = list(dict.fromkeys(selected_test_caps))

        print(f"[webvid-preprocess] numeric pipeline selections: test_vids={len(selected_test_vids)} test_caps={len(selected_test_caps)}")

        def chunk_list(lst, size):
            for i in range(0, len(lst), size):
                yield lst[i:i+size]

        # Sharded or single-file writing
        video_manifest = {}
        text_manifest = {}

        if args.videos_per_h5:
            print(f"Writing videos sharded in chunks of {args.videos_per_h5}...")
            shard_id = 0
            for chunk in chunk_list(selected_vids, args.videos_per_h5):
                vfile = out / 'FeatureData' / f"video_shard_{shard_id:03d}.hdf5"
                print(f"  -> train video shard {shard_id:03d} ({len(chunk)} videos)")
                write_video_by_indices_tensor(train_video, video_offsets, vfile, chunk)
                rel = os.path.relpath(vfile, out / 'FeatureData')
                for vid_idx in chunk:
                    video_manifest[f"vid_{vid_idx}"] = rel
                shard_id += 1
            # test videos
            for chunk in chunk_list(selected_test_vids, args.videos_per_h5):
                vfile = out / 'FeatureData' / f"video_shard_{shard_id:03d}.hdf5"
                print(f"  -> test video shard {shard_id:03d} ({len(chunk)} videos)")
                write_video_by_indices_tensor(test_video, test_video_offsets, vfile, chunk)
                rel = os.path.relpath(vfile, out / 'FeatureData')
                for vid_idx in chunk:
                    video_manifest[f"vid_{vid_idx}"] = rel
                shard_id += 1
        else:
            print(f"Writing numeric video HDF5 with {len(selected_vids)} videos: {vid_h5}")
            write_video_by_indices_tensor(train_video, video_offsets, vid_h5, selected_vids)
            if selected_test_vids:
                with h5py.File(vid_h5, 'a') as hf:
                    for vid_idx in selected_test_vids:
                        shard_idx, local = locate_index(test_video_offsets, vid_idx)
                        t = load_tensor(test_video[shard_idx])
                        feat = t[local]
                        key = f"vid_{vid_idx}"
                        if key in hf:
                            continue
                        hf.create_dataset(normalize_id(key), data=feat.detach().cpu().numpy(), compression='gzip')
            # Manifest for single file
            for vid_idx in selected_vids + selected_test_vids:
                video_manifest[f"vid_{vid_idx}"] = os.path.relpath(vid_h5, out / 'FeatureData')

        if args.caps_per_h5:
            print(f"Writing texts sharded in chunks of {args.caps_per_h5}...")
            shard_id = 0
            for chunk in chunk_list(selected_caps, args.caps_per_h5):
                tfile = out / 'TextData' / f"text_shard_{shard_id:03d}.hdf5"
                # Write chunk
                n = 0
                with h5py.File(tfile, 'w') as hf:
                    cache_idx = -1
                    cache_t = None
                    for cap_idx in chunk:
                        shard_idx, local = locate_index(text_offsets, cap_idx)
                        if shard_idx != cache_idx:
                            cache_t = load_tensor(train_text[shard_idx])
                            cache_idx = shard_idx
                        feat = cache_t[local]
                        vid_idx = cap2vid_num[cap_idx]
                        key = f"vid_{vid_idx}#cap_{cap_idx}"
                        nkey = normalize_id(key)
                        if nkey in hf:
                            continue
                        hf.create_dataset(nkey, data=feat.detach().cpu().numpy(), compression='gzip')
                        text_manifest[key] = os.path.relpath(tfile, out / 'TextData')
                        n += 1
                print(f"  -> train text shard {shard_id:03d} ({n} caps written)")
                shard_id += 1
            # test caps
            for chunk in chunk_list(selected_test_caps, args.caps_per_h5):
                tfile = out / 'TextData' / f"text_shard_{shard_id:03d}.hdf5"
                with h5py.File(tfile, 'w') as hf:
                    cache_idx = -1
                    cache_t = None
                    for cap_idx in chunk:
                        shard_idx, local = locate_index(test_text_offsets, cap_idx)
                        if shard_idx != cache_idx:
                            cache_t = load_tensor(test_text[shard_idx])
                            cache_idx = shard_idx
                        feat = cache_t[local]
                        vid_idx = cap2vid_num_test[cap_idx]
                        key = f"vid_{vid_idx}#cap_{cap_idx}"
                        nkey = normalize_id(key)
                        if nkey in hf:
                            continue
                        hf.create_dataset(nkey, data=feat.detach().cpu().numpy(), compression='gzip')
                        text_manifest[key] = os.path.relpath(tfile, out / 'TextData')
                print(f"  -> test text shard {shard_id:03d} ({len(chunk)} caps written)")
                shard_id += 1
        else:
            print(f"Writing numeric text HDF5 with {len(selected_caps)} unique caps: {text_h5}")
            write_text_by_indices_tensor(train_text, text_offsets, text_h5, selected_caps, cap2vid_num)
            if selected_test_caps:
                with h5py.File(text_h5, 'a') as hf:
                    for cap_idx in selected_test_caps:
                        shard_idx, local = locate_index(test_text_offsets, cap_idx)
                        t = load_tensor(test_text[shard_idx])
                        feat = t[local]
                        vid_idx = cap2vid_num_test[cap_idx]
                        key = f"vid_{vid_idx}#cap_{cap_idx}"
                        if normalize_id(key) in hf:
                            continue
                        hf.create_dataset(normalize_id(key), data=feat.detach().cpu().numpy(), compression='gzip')
            # Manifest for single file
            rel = os.path.relpath(text_h5, out / 'TextData')
            for c in selected_caps:
                key = f"vid_{cap2vid_num[c]}#cap_{c}"
                text_manifest[key] = rel
            for c in selected_test_caps:
                key = f"vid_{cap2vid_num_test[c]}#cap_{c}"
                text_manifest[key] = rel

        # Save manifests
        import json
        with open(out / 'FeatureData' / 'video_manifest.json', 'w') as f:
            json.dump(video_manifest, f)
        with open(out / 'TextData' / 'text_manifest.json', 'w') as f:
            json.dump(text_manifest, f)

        # Caption files
        write_caption_file(out / 'TextData' / 'webvidtrain.caption.txt', [f"vid_{cap2vid_num[c]}#cap_{c}" for c in selected_caps])
        write_caption_file(out / 'TextData' / 'webvidtest.caption.txt', [f"vid_{v}#cap_{c}" for v, caps in v2caps_test.items() for c in caps])

    else:
        # dict-based pipeline (original path)
        if args.split in ('train','both'):
            print(f"Writing text HDF5 (train/test merged): {text_h5}")
        n_text_written = 0
        if args.split in ('train','both') and train_text:
            n_text_written += merge_text_shards_to_hdf5(
                train_text, text_h5,
                cap_whitelist=cap_whitelist if cap_whitelist else None,
                max_caps=None,  # already controlled via whitelist
                rename_map=rename_train
            )
        if args.split in ('test','both') and test_text:
            # Append test captions to same file
            with h5py.File(text_h5, 'a') as hf:
                for shard in test_text:
                    data = torch.load(shard, map_location='cpu')
                    for cid, feat in data.items():
                        if test_cap_whitelist and cid not in test_cap_whitelist:
                            continue
                        cid_out = rename_test.get(cid, cid)
                        cid_norm = normalize_id(str(cid_out))
                        if cid_norm in hf:
                            continue
                        arr = feat.detach().cpu().numpy()
                        hf.create_dataset(cid_norm, data=arr, compression="gzip")
                        n_text_written += 1
                    del data
                    gc.collect()

        # Merge videos
        if args.split in ('train','both') and train_video:
            print(f"Writing video HDF5 (train): {vid_h5}")
            merge_video_shards_to_hdf5(train_video, vid_h5,
                                       vid_whitelist=vid_whitelist if vid_whitelist else None,
                                       max_videos=None)
        if args.split in ('test','both') and test_video:
            # Append test videos to same file (avoid collisions)
            with h5py.File(vid_h5, 'a') as hf:
                for shard in test_video:
                    data = torch.load(shard, map_location='cpu')
                    for vid, feat in data.items():
                        if selected_test_vids and vid not in selected_test_vids:
                            continue
                        vid_norm = normalize_id(str(vid))
                        if vid_norm in hf:
                            continue
                        arr = feat.detach().cpu().numpy()
                        hf.create_dataset(vid_norm, data=arr, compression="gzip")
                    del data
                    gc.collect()

    # Write caption id lists (placeholders for content)
    if not numeric_pipeline:
        if args.split in ('train','both') and cap2vid_train:
            write_caption_file(out / 'TextData' / 'webvidtrain.caption.txt', [rename_train[c] for c in cap_whitelist])
        if args.split in ('test','both') and cap2vid_test:
            write_caption_file(out / 'TextData' / 'webvidtest.caption.txt', [rename_test[c] for c in test_cap_whitelist])

    print("Done. Outputs written under:")
    print(f"  {out}")


if __name__ == '__main__':
    main()
