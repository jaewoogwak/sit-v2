import os
import json
import zipfile
import numpy as np
import pickle
import sys, array
import logging
import h5py
import json
from collections import OrderedDict
import re
import bisect

import ipdb

class BigFile:

    def __init__(self, datadir):

        self.nr_of_images, self.ndims = map(int, open(os.path.join(datadir, 'shape.txt')).readline().split())
        self.id_file = os.path.join(datadir, "id.txt")
        # Lazy: avoid loading huge id.txt into RAM; map only requested keys
        self.name2index = {}
        self.binary_file = os.path.join(datadir, "feature.bin")
        print("[%s] %dx%d instances ready from %s (lazy index)" % (self.__class__.__name__, self.nr_of_images, self.ndims, datadir))

    def prepare_index(self, names):
        """Ensure indices for a set of names by scanning id.txt once."""
        targets = set(str(n) for n in names if n is not None)
        targets -= self.name2index.keys()
        if not targets:
            return
        found = 0
        with open(self.id_file, 'r', encoding='ISO-8859-1', errors='ignore') as fr:
            for idx, line in enumerate(fr):
                nm = line.strip().split()[0] if line else None
                if nm in targets:
                    self.name2index[nm] = idx
                    found += 1
                    if found >= len(targets):
                        break
        missing = len(targets) - found
        if missing > 0:
            print(f"[BigFile] Warning: {missing} keys not found during prepare_index")

    def read(self, requested, isname=True):
        requested = list(requested)
        if isname:
            # lazily map missing keys
            need = [x for x in requested if x not in self.name2index]
            if need:
                self.prepare_index(need)
            index_name_array = [(self.name2index[x], x) for x in requested if x in self.name2index]
        else:
            assert (min(requested) >= 0)
            assert (max(requested) < self.nr_of_images)
            index_name_array = [(x, None) for x in requested]
        if len(index_name_array) == 0:
            return [], []

        index_name_array.sort(key=lambda v: v[0])
        sorted_index = [x[0] for x in index_name_array]

        nr_of_images = len(index_name_array)
        offset = np.float32(1).nbytes * self.ndims

        res = array.array('f')
        fr = open(self.binary_file, 'rb')
        fr.seek(sorted_index[0] * offset)
        res.fromfile(fr, self.ndims)
        previous = sorted_index[0]

        for nxt in sorted_index[1:]:
            move = (nxt - 1 - previous) * offset
            fr.seek(move, 1)
            res.fromfile(fr, self.ndims)
            previous = nxt

        fr.close()

        names_out = [x[1] if x[1] is not None else str(x[0]) for x in index_name_array]
        return names_out, [res[i * self.ndims:(i + 1) * self.ndims].tolist() for i in range(nr_of_images)]

    def read_one(self, name):
        renamed, vectors = self.read([name])
        if not vectors:
            raise KeyError(f"Key {name} not found in BigFile index")
        return vectors[0]

    def shape(self):
        return [self.nr_of_images, self.ndims]


class HDF5File:
    """HDF5 파일을 읽기 위한 클래스 (CLIP 피쳐 지원)"""
    
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.h5file = h5py.File(hdf5_path, 'r')
        
        # HDF5 파일에서 첫 번째 키의 feature 차원 확인
        first_key = list(self.h5file.keys())[0]
        first_feature = self.h5file[first_key][...]
        self.ndims = first_feature.shape[-1]
        self.nr_of_images = len(self.h5file.keys())
        
        print(f"[{self.__class__.__name__}] {self.nr_of_images} instances with {self.ndims} dims loaded from {hdf5_path}")
    
    def read_one(self, name):
        """단일 피쳐 읽기"""
        # 프레임 인덱스가 포함된 키인 경우 비디오 ID만 추출
        if '_' in name and name.count('_') >= 2:
            # v_videoID_frameIndex 형태에서 v_videoID만 추출
            parts = name.rsplit('_', 1)
            if parts[1].isdigit():  # 마지막 부분이 숫자인 경우 (프레임 인덱스)
                video_id = parts[0]
                if video_id in self.h5file:
                    # 전체 비디오 피쳐에서 해당 프레임 인덱스 반환
                    frame_idx = int(parts[1])
                    video_features = self.h5file[video_id][...]
                    if frame_idx < len(video_features):
                        return video_features[frame_idx]
                    else:
                        # 프레임 인덱스가 범위를 벗어나면 마지막 프레임 반환
                        return video_features[-1]
        
        # 원래 키 그대로 시도
        if name in self.h5file:
            return self.h5file[name][...]
        else:
            raise KeyError(f"Key {name} not found in HDF5 file")
    
    def read(self, requested, isname=True):
        """여러 피쳐 읽기 (BigFile과 호환성을 위한 인터페이스)"""
        if not isname:
            raise NotImplementedError("Index-based reading not implemented for HDF5File")
        
        names = []
        vectors = []
        for name in requested:
            try:
                feat = self.read_one(name)
                names.append(name)
                vectors.append(feat.tolist())
            except KeyError:
                continue
        
        return names, vectors
    
    def shape(self):
        return [self.nr_of_images, self.ndims]
    
    def __del__(self):
        if hasattr(self, 'h5file'):
            self.h5file.close()


class InternVideoHDF5:
    """Nested HDF5 reader for InternVideo2 features: /videos/<vid>/clip_embeddings."""

    def __init__(self, hdf5_path, video_group='videos', dataset_name='clip_embeddings'):
        self.hdf5_path = hdf5_path
        self.video_group = video_group
        self.dataset_name = dataset_name
        self.h5file = h5py.File(hdf5_path, 'r')

        group = self.h5file[self.video_group]
        first_key = list(group.keys())[0]
        first_feature = group[first_key][self.dataset_name][...]
        self.ndims = first_feature.shape[-1]
        self.nr_of_images = len(group.keys())

        print(f"[{self.__class__.__name__}] {self.nr_of_images} instances with {self.ndims} dims loaded from {hdf5_path}")

    def _get_dataset(self, video_id):
        group = self.h5file[self.video_group]
        if video_id not in group:
            raise KeyError(f"Key {video_id} not found in HDF5 group {self.video_group}")
        return group[video_id][self.dataset_name]

    def read_one(self, name):
        """Read one frame or full video features."""
        try:
            dataset = self._get_dataset(name)
            return dataset[...]
        except KeyError:
            pass

        if '_' in name and name.count('_') >= 2:
            parts = name.rsplit('_', 1)
            if parts[1].isdigit():
                video_id = parts[0]
                frame_idx = int(parts[1])
                try:
                    dataset = self._get_dataset(video_id)
                    if frame_idx < dataset.shape[0]:
                        return dataset[frame_idx]
                    return dataset[-1]
                except KeyError:
                    pass


class TVRFrameNPY:
    """Reader for per-video frame embedding npy files (TVR)."""

    def __init__(self, feature_dir, show_prefixes=None, default_show="bbt", ndims=512):
        self.feature_dir = feature_dir
        self.show_prefixes = show_prefixes or ["bbt", "castle", "friends", "grey", "house", "met"]
        self.default_show = default_show
        self.ndims = ndims
        self.nr_of_images = 0

    def _resolve_path(self, video_id):
        show = None
        base = video_id
        for prefix in self.show_prefixes:
            if video_id.startswith(prefix + "_"):
                show = prefix
                base = video_id[len(prefix) + 1:]
                break

        candidates = []
        if show is None:
            candidates.append(f"{self.default_show}_frames__{video_id}.npy")
            for prefix in self.show_prefixes:
                candidates.append(f"{prefix}_frames__{video_id}.npy")
        else:
            candidates.append(f"{show}_frames__{base}.npy")
            candidates.append(f"{show}_frames__{video_id}.npy")

        for name in candidates:
            path = os.path.join(self.feature_dir, name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Frame npy not found for video_id={video_id}")

    def read_video(self, video_id):
        path = self._resolve_path(video_id)
        return np.load(path, mmap_mode="r")

    def shape(self):
        return [self.nr_of_images, self.ndims]


class SimpleFrameNPY:
    """Reader for per-video frame embedding npy files named as <video_id>.npy."""

    def __init__(self, feature_dir, suffix=".npy", ndims=512):
        self.feature_dir = feature_dir
        self.suffix = suffix
        self.ndims = ndims
        self.nr_of_images = 0

    def _resolve_path(self, video_id):
        filename = video_id if video_id.endswith(self.suffix) else f"{video_id}{self.suffix}"
        path = os.path.join(self.feature_dir, filename)
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Frame npy not found for video_id={video_id}")

    def read_video(self, video_id):
        path = self._resolve_path(video_id)
        return np.load(path, mmap_mode="r")

    def shape(self):
        return [self.nr_of_images, self.ndims]

        raise KeyError(f"Key {name} not found in HDF5 file")

    def read(self, requested, isname=True):
        if not isname:
            raise NotImplementedError("Index-based reading not implemented for InternVideoHDF5")
        names = []
        vectors = []
        for name in requested:
            try:
                feat = self.read_one(name)
                names.append(name)
                vectors.append(feat.tolist())
            except KeyError:
                continue
        return names, vectors

    def shape(self):
        return [self.nr_of_images, self.ndims]

    def __del__(self):
        if hasattr(self, 'h5file'):
            self.h5file.close()


class InternVideoTextH5:
    """Text feature reader for InternVideo2: /queries/<vid>/embeddings."""

    def __init__(self, hdf5_path, query_group='queries', dataset_name='embeddings'):
        self.hdf5_path = hdf5_path
        self.query_group = query_group
        self.dataset_name = dataset_name
        self.h5file = None

    def _ensure(self):
        if self.h5file is None:
            self.h5file = h5py.File(self.hdf5_path, 'r')

    @staticmethod
    def _parse_cap_id(cap_id):
        video_id = cap_id.split('#', 1)[0]
        idx = 0
        if '#' in cap_id:
            tail = cap_id.split('#')[-1]
            if tail.isdigit():
                idx = int(tail)
        return video_id, idx

    def get(self, cap_id):
        self._ensure()
        video_id, idx = self._parse_cap_id(cap_id)
        group = self.h5file[self.query_group]
        if video_id not in group:
            raise KeyError(f"Key {video_id} not found in HDF5 group {self.query_group}")
        dataset = group[video_id][self.dataset_name]
        if idx < 0:
            idx = 0
        if idx >= dataset.shape[0]:
            idx = dataset.shape[0] - 1
        return dataset[idx]

    def __del__(self):
        if hasattr(self, 'h5file') and self.h5file is not None:
            self.h5file.close()


class MultiHDF5File:
    """Multi-file HDF5 reader with manifest mapping name -> hdf5 path.
    Manifest format (JSON): { "name_or_video_id": "relative/or/absolute.hdf5", ... }
    Keeps a small LRU cache of open files to limit file handles.
    """

    def __init__(self, manifest_path, max_open=8):
        self.manifest_path = manifest_path
        with open(manifest_path, 'r') as f:
            self.map = json.load(f)
        # Normalize to strings
        self.map = {str(k): str(v) for k, v in self.map.items()}
        # Base dir to resolve relative paths
        self.base_dir = os.path.dirname(os.path.abspath(manifest_path))
        # LRU cache for h5 files
        self.max_open = max_open
        self._open_files = OrderedDict()
        # Derive dims: sample one key
        sample_key = next(iter(self.map.keys()))
        fpath = self._resolve(self.map[sample_key])
        with h5py.File(fpath, 'r') as hf:
            try:
                if sample_key in hf:
                    arr = hf[sample_key][...]
                else:
                    first = next(iter(hf.keys()))
                    arr = hf[first][...]
                self.ndims = arr.shape[-1] if isinstance(arr, np.ndarray) else int(arr[-1])
            except Exception as e:
                # 안전 폴백: CLIP 텍스트 512차원 가정
                print(f"[MultiHDF5File] warn: probe failed on {fpath} ({e}); default ndims=512")
                self.ndims = 512
        self.nr_of_images = len(self.map)
        
    def _resolve(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)

    def _get_h5(self, path):
        path = self._resolve(path)
        if path in self._open_files:
            self._open_files.move_to_end(path)
            return self._open_files[path]
        # Open new
        hf = h5py.File(path, 'r')
        self._open_files[path] = hf
        # Evict
        while len(self._open_files) > self.max_open:
            old_path, old_hf = self._open_files.popitem(last=False)
            try:
                old_hf.close()
            except Exception:
                pass
        return hf


class ShardedPackedBigFile:
    """BigFile 변형: feature_shard_*.bin으로 분할 저장된 i3d 피쳐 로더."""

    def __init__(self, datadir: str, frames_per_video: int = 12):
        self.datadir = datadir
        self.frames_per_video = int(frames_per_video)
        shape_fp = os.path.join(datadir, 'shape.txt')
        if not os.path.isfile(shape_fp):
            raise FileNotFoundError(f"shape.txt not found in {datadir}")
        with open(shape_fp, 'r') as fr:
            self.nr_of_images, self.ndims = map(int, fr.readline().split())

        shard_files = [f for f in os.listdir(datadir)
                       if f.startswith('feature_shard_') and f.endswith('.bin')]
        if not shard_files:
            raise FileNotFoundError(f"No feature_shard_*.bin under {datadir}")

        def _key(name: str) -> int:
            try:
                return int(name.split('_')[-1].split('.')[0])
            except Exception:
                return 1 << 30

        shard_files.sort(key=_key)
        self.shard_paths = [os.path.join(datadir, f) for f in shard_files]
        row_bytes = np.float32(1).nbytes * self.ndims
        self.rows_per_shard = [int(os.path.getsize(p) // row_bytes) for p in self.shard_paths]
        self.cum_rows = []
        total = 0
        for r in self.rows_per_shard:
            total += r
            self.cum_rows.append(total)
        self.total_rows_computed = self.cum_rows[-1]
        if self.total_rows_computed != self.nr_of_images:
            logging.warning(
                f"[ShardedPackedBigFile] shape rows {self.nr_of_images} != computed {self.total_rows_computed}")
        assert self.nr_of_images % self.frames_per_video == 0, (
            "ShardedPackedBigFile expects rows divisible by frames_per_video")
        self.total_videos = self.nr_of_images // self.frames_per_video
        self._fh = None
        self._fh_path = None
        print("[%s] %dx%d instances across %d shards from %s (packed %d frames/video, ~%d videos)" % (
            self.__class__.__name__, self.nr_of_images, self.ndims, len(self.shard_paths), datadir,
            self.frames_per_video, self.total_videos))

    def _name_to_index(self, name: str) -> int:
        try:
            base, frame = name.rsplit('_', 1)
            vid_idx = int(base.split('_')[1])
            frame_idx = int(frame)
        except Exception as e:
            raise KeyError(f"Unrecognized packed frame name: {name}") from e
        if frame_idx < 0 or frame_idx >= self.frames_per_video:
            raise IndexError(f"Frame index out of range for {name}")
        return vid_idx * self.frames_per_video + frame_idx

    def _index_to_shard(self, idx: int):
        if idx < 0 or idx >= self.nr_of_images:
            raise IndexError(idx)
        shard_pos = bisect.bisect_right(self.cum_rows, idx)
        start_row = 0 if shard_pos == 0 else self.cum_rows[shard_pos - 1]
        offset_row = idx - start_row
        return shard_pos, offset_row

    def _ensure_open(self, path: str):
        if self._fh_path == path and self._fh is not None:
            return
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = open(path, 'rb')
        self._fh_path = path

    def read_one(self, name: str):
        gidx = self._name_to_index(name)
        shard_idx, offset_row = self._index_to_shard(gidx)
        path = self.shard_paths[shard_idx]
        self._ensure_open(path)
        row_bytes = np.float32(1).nbytes * self.ndims
        self._fh.seek(offset_row * row_bytes)
        buf = array.array('f')
        buf.fromfile(self._fh, self.ndims)
        return buf.tolist()

    def read(self, requested, isname=True):
        if not isname:
            raise NotImplementedError("Index-based access not supported")
        names = []
        vectors = []
        for name in requested:
            try:
                vec = self.read_one(name)
            except Exception:
                continue
            names.append(name)
            vectors.append(vec)
        return names, vectors

    def shape(self):
        return [self.nr_of_images, self.ndims]

    def prepare_index(self, names):
        # Sharded 파일은 필요한 프레임만 직접 접근하므로 별도 인덱스 준비 불필요
        return

    def __del__(self):
        if getattr(self, '_fh', None) is not None:
            try:
                self._fh.close()
            except Exception:
                pass


class Video2FramesView:
    """vid_{i} -> [vid_{i}_0, ...] 뷰를 제공하는 경량 매핑."""

    def __init__(self, frames_per_video: int = 12):
        self.frames_per_video = int(frames_per_video)

    def __getitem__(self, video_id: str):
        if not video_id.startswith('vid_'):
            raise KeyError(f"Unexpected video id format: {video_id}")
        try:
            int(video_id.split('_')[1])
        except Exception as e:
            raise KeyError(f"Invalid video id: {video_id}") from e
        return [f"{video_id}_{f}" for f in range(self.frames_per_video)]

    def keys(self):
        # 무한 generator 형태이나, 실제 사용 시에는 video_ids 인자를 통해 범위를 제한한다.
        idx = 0
        while True:
            yield f"vid_{idx}"
            idx += 1


class IndexedTextByJsonIndex:
    """webvid_text_index.json 기반 RoBERTa 텍스트 샤드 라우터."""

    def __init__(self, text_dir: str, index_path: str, max_open: int = 8):
        self.text_dir = text_dir
        self.index_path = index_path
        with open(index_path, 'r') as f:
            self._index = json.load(f)
        self.max_open = max_open
        self._open_files = OrderedDict()

    @staticmethod
    def _normalize_key(cap_id: str) -> str:
        key = cap_id.replace('#', '_') if '#' in cap_id else cap_id
        return key.replace('/', '_')

    def _get_file(self, shard_name: str):
        path = os.path.join(self.text_dir, shard_name)
        if path in self._open_files:
            self._open_files.move_to_end(path)
            return self._open_files[path]
        hf = h5py.File(path, 'r')
        self._open_files[path] = hf
        while len(self._open_files) > self.max_open:
            _, old = self._open_files.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return hf

    def get(self, cap_id: str):
        key = self._normalize_key(cap_id)
        if key not in self._index:
            raise KeyError(f"{cap_id} not found in index {self.index_path}")
        shard_name, chunk_name, ds_key = self._index[key]
        hf = self._get_file(shard_name)
        chunk = hf[chunk_name]
        return chunk[ds_key][...]

    def close(self):
        for hf in list(self._open_files.values()):
            try:
                hf.close()
            except Exception:
                pass
        self._open_files.clear()

    def __del__(self):
        self.close()


class RobertaShardTextReader:
    """webvid_dummy RoBERTa 텍스트 샤드용 경량 라우터 (index JSON 미사용)."""

    def __init__(self, text_dir: str, shards_tsv: str = 'roberta_shards.tsv', max_open: int = 6):
        self.text_dir = text_dir
        self.max_open = max_open
        self._open_files = OrderedDict()
        self.shards = []
        self.index_path = os.path.join(text_dir, 'webvid_text_index.json')
        self._index = None

        tsv_path = shards_tsv
        if not os.path.isabs(tsv_path):
            tsv_path = os.path.join(text_dir, shards_tsv)
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"{tsv_path} not found for RoBERTa shards")

        with open(tsv_path, 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                shard_idx = int(parts[0])
                filename = parts[1]
                start_row = int(parts[2])
                num_rows = int(parts[3])
                self.shards.append({
                    'idx': shard_idx,
                    'filename': filename,
                    'start': start_row,
                    'end': start_row + num_rows,
                    'rows': num_rows,
                    'chunk_ids': None,
                    'chunk_size': None,
                })
        if not self.shards:
            raise RuntimeError(f"No shard entries parsed from {tsv_path}")

        self._starts = [info['start'] for info in self.shards]
        self.total_rows = sum(info['rows'] for info in self.shards)

    @staticmethod
    def _sanitize_key(cap_id: str) -> str:
        key = cap_id.replace('#', '_')
        key = key.replace('/', '_')
        return key

    @staticmethod
    def _parse_cap_index(cap_id: str) -> int:
        if '#cap_' in cap_id:
            try:
                return int(cap_id.split('#cap_')[-1].split('_')[0])
            except ValueError:
                pass
        # Fallback: extract trailing digits
        digits = ''.join(ch if ch.isdigit() else ' ' for ch in cap_id).split()
        if digits:
            return int(digits[-1])
        raise KeyError(f"Cannot parse caption index from {cap_id}")

    def _locate_shard(self, cap_idx: int):
        pos = bisect.bisect_right(self._starts, cap_idx) - 1
        while pos >= 0:
            info = self.shards[pos]
            if info['start'] <= cap_idx < info['end']:
                return pos, info
            pos -= 1
        raise KeyError(f"Caption index {cap_idx} outside shard range")

    def _get_file(self, info):
        path = os.path.join(self.text_dir, info['filename'])
        if path in self._open_files:
            self._open_files.move_to_end(path)
            hf = self._open_files[path]
        else:
            hf = h5py.File(path, 'r')
            self._open_files[path] = hf
            while len(self._open_files) > self.max_open:
                _, old = self._open_files.popitem(last=False)
                try:
                    old.close()
                except Exception:
                    pass

        if info['chunk_ids'] is None:
            chunk_ids = []
            for name in hf.keys():
                if name.startswith('chunk_'):
                    try:
                        chunk_ids.append(int(name.split('_')[1]))
                    except Exception:
                        continue
            chunk_ids.sort()
            info['chunk_ids'] = chunk_ids if chunk_ids else [0]
            import math
            if chunk_ids:
                info['chunk_size'] = max(1, math.ceil(info['rows'] / max(len(chunk_ids), 1)))
            else:
                info['chunk_size'] = info['rows']

        chunk_ids = info['chunk_ids']
        if not chunk_ids:
            info['chunk_size'] = info.get('chunk_size', info['rows'])

        return hf

    def _get_file_by_name(self, shard_name: str):
        path = os.path.join(self.text_dir, shard_name)
        if path in self._open_files:
            self._open_files.move_to_end(path)
            return self._open_files[path]
        hf = h5py.File(path, 'r')
        self._open_files[path] = hf
        while len(self._open_files) > self.max_open:
            _, old = self._open_files.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return hf

    def _ensure_index(self):
        if self._index is None and os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                self._index = json.load(f)

    def get(self, cap_id: str):
        cap_idx = self._parse_cap_index(cap_id)
        shard_pos, info = self._locate_shard(cap_idx)
        hf = self._get_file(info)
        offset = cap_idx - info['start']
        chunk_ids = info['chunk_ids']
        chunk_size = info['chunk_size']
        approx = min(len(chunk_ids) - 1, offset // chunk_size)
        search_order = [approx]
        for delta in (-1, 1, -2, 2):
            cand = approx + delta
            if 0 <= cand < len(chunk_ids):
                search_order.append(cand)

        raw_key = cap_id.replace('/', '_')
        ds_key = self._sanitize_key(cap_id)
        # handle occasional '#enc#' style ids
        alt_key = None
        if '#enc#' in cap_id:
            alt_cap = '#'.join(cap_id.split('#enc#'))
            alt_key = self._sanitize_key(alt_cap)

        self._ensure_index()
        if self._index is not None:
            idx_entry = None
            for candidate in (ds_key, raw_key, alt_key):
                if candidate and candidate in self._index:
                    idx_entry = self._index[candidate]
                    break
            if idx_entry is not None:
                shard_name, chunk_name, real_key = idx_entry
                hf_direct = self._get_file_by_name(shard_name)
                if chunk_name in hf_direct:
                    grp = hf_direct[chunk_name]
                    for candidate_key in (real_key,
                                          real_key.replace('#', '_'),
                                          real_key.replace('#', '_').replace('/', '_')):
                        if candidate_key in grp:
                            return grp[candidate_key][...]
                # If index points to missing entry, fall through to dynamic search

        visited = set()
        for idx in search_order:
            chunk_name = f"chunk_{chunk_ids[idx]:04d}"
            if chunk_name in visited:
                continue
            visited.add(chunk_name)
            if chunk_name not in hf:
                continue
            grp = hf[chunk_name]
            if ds_key in grp:
                return grp[ds_key][...]
            if alt_key and alt_key in grp:
                return grp[alt_key][...]

        # 마지막 수단: shard의 모든 chunk 탐색
        for chunk_idx in chunk_ids:
            chunk_name = f"chunk_{chunk_idx:04d}"
            if chunk_name in visited:
                continue
            if chunk_name not in hf:
                continue
            grp = hf[chunk_name]
            if ds_key in grp:
                return grp[ds_key][...]
            if alt_key and alt_key in grp:
                return grp[alt_key][...]
            for key in grp.keys():
                if key == ds_key or (alt_key and key == alt_key):
                    return grp[key][...]
        raise KeyError(f"Feature not found for {cap_id}")

    def close(self):
        for hf in list(self._open_files.values()):
            try:
                hf.close()
            except Exception:
                pass
        self._open_files.clear()

    def __del__(self):
        self.close()


class MultiTextH5:
    """Text feature multi-HDF5 reader using manifest mapping cap_id -> hdf5 path."""
    def __init__(self, manifest_path, max_open=8):
        self._multi = MultiHDF5File(manifest_path, max_open=max_open)

    def get(self, cap_id):
        return self._multi.read_one(cap_id)

    def __contains__(self, cap_id):
        return str(cap_id) in self._multi.map


class IndexedVideoHDF5:
    """Index-based HDF5 router for videos.
    Expects shards named like video_shard_%03d.hdf5 under FeatureData and JSON meta:
      {"per_shard": 50000, "pattern": "video_shard_%03d.hdf5", "total": N}
    Keys: vid_{index}
    """

    def __init__(self, feature_dir, meta_json, max_open=8):
        self.feature_dir = feature_dir
        with open(meta_json, 'r') as f:
            meta = json.load(f)
        self.per_shard = int(meta.get('per_shard'))
        self.pattern = meta.get('pattern', 'video_shard_%03d.hdf5')
        self.total = int(meta.get('total'))
        self.max_open = max_open
        self._open_files = OrderedDict()

    _vid_re = re.compile(r"vid_(\d+)")

    def _resolve(self, shard_idx: int):
        fname = self.pattern % shard_idx
        return os.path.join(self.feature_dir, fname)

    def _get_h5(self, path):
        if path in self._open_files:
            self._open_files.move_to_end(path)
            return self._open_files[path]
        hf = h5py.File(path, 'r')
        self._open_files[path] = hf
        while len(self._open_files) > self.max_open:
            _, old = self._open_files.popitem(last=False)
            try: old.close()
            except Exception: pass
        return hf

    def read_one(self, name: str):
        m = self._vid_re.match(str(name))
        if not m:
            raise KeyError(f"Invalid video key: {name}")
        idx = int(m.group(1))
        shard = idx // self.per_shard
        path = self._resolve(shard)
        hf = self._get_h5(path)
        key = str(name)
        if key in hf:
            return hf[key][...]
        nkey = key.replace('/', '_').replace('\\', '_')
        if nkey in hf:
            return hf[nkey][...]
        raise KeyError(f"Key {key} not found in {path}")

    def shape(self):
        return [self.total, 512]

    def __del__(self):
        for _, hf in list(self._open_files.items()):
            try: hf.close()
            except Exception: pass


class IndexedTextH5:
    """Index-based HDF5 router for text.
    Expects shards named like text_shard_%03d.hdf5 under TextData and JSON meta:
      {"per_shard": 500000, "pattern": "text_shard_%03d.hdf5", "total": N}
    Keys: vid_{vid_idx}#cap_{cap_idx}
    """

    def __init__(self, text_dir, meta_json, max_open=8):
        self.text_dir = text_dir
        with open(meta_json, 'r') as f:
            meta = json.load(f)
        self.per_shard = int(meta.get('per_shard'))
        self.pattern = meta.get('pattern', 'text_shard_%03d.hdf5')
        self.total = int(meta.get('total'))
        self.max_open = max_open
        self._open_files = OrderedDict()

    _cap_re = re.compile(r"vid_(\d+)#cap_(\d+)")

    def _resolve(self, shard_idx: int):
        fname = self.pattern % shard_idx
        return os.path.join(self.text_dir, fname)

    def _get_h5(self, path):
        if path in self._open_files:
            self._open_files.move_to_end(path)
            return self._open_files[path]
        hf = h5py.File(path, 'r')
        self._open_files[path] = hf
        while len(self._open_files) > self.max_open:
            _, old = self._open_files.popitem(last=False)
            try: old.close()
            except Exception: pass
        return hf

    def get(self, cap_id: str):
        m = self._cap_re.match(str(cap_id))
        if not m:
            raise KeyError(f"Invalid cap key: {cap_id}")
        vid_idx = int(m.group(1))
        cap_idx = int(m.group(2))
        shard = cap_idx // self.per_shard
        path = self._resolve(shard)
        hf = self._get_h5(path)
        key = f"vid_{vid_idx}#cap_{cap_idx}"
        if key in hf:
            return hf[key][...]
        nkey = key.replace('/', '_').replace('\\', '_')
        if nkey in hf:
            return hf[nkey][...]
        raise KeyError(f"Key {key} not found in {path}")


def log_config(log_dir, ca='logging'):
    logger = logging.getLogger()
    filename = log_dir +'/' + ca + '.log'
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    # logger.addHandler(logging.StreamHandler())
    logger.addHandler(handler)


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


def compute_overlap(pred, gt):
    # check format
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    pred = pred if pred_is_list else [pred]
    gt = gt if gt_is_list else [gt]
    # compute overlap
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    overlap = overlap if pred_is_list else overlap[0]
    return overlap


def time_to_index(start_time, end_time, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) / float(num_units) * duration
    e_times = np.arange(1, num_units + 1).astype(np.float32) / float(num_units) * duration
    candidates = np.stack([np.repeat(s_times[:, None], repeats=num_units, axis=1),
                           np.repeat(e_times[None, :], repeats=num_units, axis=0)], axis=2).reshape((-1, 2))
    overlaps = compute_overlap(candidates.tolist(), [start_time, end_time]).reshape(num_units, num_units)
    start_index = np.argmax(overlaps) // num_units
    end_index = np.argmax(overlaps) % num_units
    return start_index, end_index


def read_dict(filepath):
    f = open(filepath,'r')
    a = f.read()
    dict_data = eval(a)
    f.close()
    return dict_data

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def save_lines(list_of_str, filepath):
    with open(filepath, "w") as f:
        f.write("\n".join(list_of_str))


def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def convert_to_seconds(hms_time):
    """ convert '00:01:12' to 72 seconds.
    :hms_time (str): time in comma separated string, e.g. '00:01:12'
    :return (int): time in seconds, e.g. 72
    """
    times = [float(t) for t in hms_time.split(":")]
    return times[0] * 3600 + times[1] * 60 + times[2]


def get_video_name_from_url(url):
    return url.split("/")[-1][:-4]


def merge_dicts(list_dicts):
    merged_dict = list_dicts[0].copy()
    for i in range(1, len(list_dicts)):
        merged_dict.update(list_dicts[i])
    return merged_dict


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def make_zipfile(src_dir, save_path, enclosing_dir="", exclude_dirs=None, exclude_extensions=None,
                 exclude_dirs_substring=None):
    """make a zip file of root_dir, save it to save_path.
    exclude_paths will be excluded if it is a subdir of root_dir.
    An enclosing_dir is added is specified.
    """
    abs_src = os.path.abspath(src_dir)
    with zipfile.ZipFile(save_path, "w") as zf:
        for dirname, subdirs, files in os.walk(src_dir):
            if exclude_dirs is not None:
                for e_p in exclude_dirs:
                    if e_p in subdirs:
                        subdirs.remove(e_p)
            if exclude_dirs_substring is not None:
                to_rm = []
                for d in subdirs:
                    if exclude_dirs_substring in d:
                        to_rm.append(d)
                for e in to_rm:
                    subdirs.remove(e)
            arcname = os.path.join(enclosing_dir, dirname[len(abs_src) + 1:])
            zf.write(dirname, arcname)
            for filename in files:
                if exclude_extensions is not None:
                    if os.path.splitext(filename)[1] in exclude_extensions:
                        continue  # do not zip it
                absname = os.path.join(dirname, filename)
                arcname = os.path.join(enclosing_dir, absname[len(abs_src) + 1:])
                zf.write(absname, arcname)


class AverageMeter(object):
    """Computes and stores the average and current/max/min value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10

    def update(self, val, n=1):
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dissect_by_lengths(np_array, lengths, dim=0, assert_equal=True):
    """Dissect an array (N, D) into a list a sub-array,
    np_array.shape[0] == sum(lengths), Output is a list of nd arrays, singlton dimention is kept"""
    if assert_equal:
        assert len(np_array) == sum(lengths)
    length_indices = [0, ]
    for i in range(len(lengths)):
        length_indices.append(length_indices[i] + lengths[i])
    if dim == 0:
        array_list = [np_array[length_indices[i]:length_indices[i+1]] for i in range(len(lengths))]
    elif dim == 1:
        array_list = [np_array[:, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    elif dim == 2:
        array_list = [np_array[:, :, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    else:
        raise NotImplementedError
    return array_list


def get_ratio_from_counter(counter_obj, threshold=200):
    keys = counter_obj.keys()
    values = counter_obj.values()
    filtered_values = [counter_obj[k] for k in keys if k > threshold]
    return float(sum(filtered_values)) / sum(values)


def get_show_name(vid_name):
    """
    get tvshow name from vid_name
    :param vid_name: video clip name
    :return: tvshow name
    """
    show_list = ["friends", "met", "castle", "house", "grey"]
    vid_name_prefix = vid_name.split("_")[0]
    show_name = vid_name_prefix if vid_name_prefix in show_list else "bbt"
    return show_name
