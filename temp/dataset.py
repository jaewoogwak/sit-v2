import json
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TVRDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        h5_path,
        seq_len=5,
        split="train",
        split_ratio=0.9,
        seed=42,
    ):
        self.jsonl_path = jsonl_path
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self._h5 = None

        self.entries = self._load_entries()
        self.indices = self._make_split_indices()

    def _load_entries(self):
        entries = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        return entries

    def _make_split_indices(self):
        rng = random.Random(self.seed)
        all_indices = list(range(len(self.entries)))
        rng.shuffle(all_indices)
        if self.split in ("all", "full"):
            return all_indices
        cut = int(len(all_indices) * self.split_ratio)
        if self.split == "train":
            return all_indices[:cut]
        if self.split == "val":
            return all_indices[cut:]
        raise ValueError(f"Unknown split: {self.split}")

    def _get_vid(self, entry):
        for key in ("vid_name", "video_id", "vid", "video", "video_name"):
            if key in entry:
                return entry[key]
        raise KeyError("No video id field found in JSONL entry")

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __len__(self):
        return len(self.indices)

    def _sample_start(self, length, idx):
        if length <= 0:
            return 0
        if self.split == "train":
            return random.randint(0, max(0, length - 1))
        rng = random.Random(self.seed + idx)
        return rng.randint(0, max(0, length - 1))

    def _build_sequence(self, feats, start):
        seq_len = self.seq_len
        feat_dim = feats.shape[1]
        seq = np.zeros((seq_len, feat_dim), dtype=feats.dtype)
        target = np.zeros((feat_dim,), dtype=feats.dtype)
        for i in range(seq_len):
            t = start + i
            if t < feats.shape[0]:
                seq[i] = feats[t]
        target_idx = start + seq_len
        if target_idx < feats.shape[0]:
            target = feats[target_idx]
        return seq, target

    def _build_sequence_from_chunk(self, chunk, feat_dim, dtype):
        seq_len = self.seq_len
        seq = np.zeros((seq_len, feat_dim), dtype=dtype)
        target = np.zeros((feat_dim,), dtype=dtype)
        if chunk.size == 0:
            return seq, target
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        for i in range(min(seq_len, chunk.shape[0])):
            seq[i] = chunk[i]
        if chunk.shape[0] > seq_len:
            target = chunk[seq_len]
        return seq, target

    def __getitem__(self, idx):
        entry = self.entries[self.indices[idx]]
        vid = self._get_vid(entry)
        dset = self._get_h5()[vid]
        length = dset.shape[0] if len(dset.shape) > 0 else 0
        start = self._sample_start(length, idx)
        end = min(length, start + self.seq_len + 1)
        chunk = np.asarray(dset[start:end])
        feat_dim = chunk.shape[1] if chunk.ndim > 1 else (dset.shape[1] if len(dset.shape) > 1 else 1)
        seq, target = self._build_sequence_from_chunk(chunk, feat_dim, chunk.dtype if chunk.size else dset.dtype)
        return torch.from_numpy(seq).float(), torch.from_numpy(target).float()

    def get_video_ids(self, unique=False):
        vids = [self._get_vid(self.entries[i]) for i in self.indices]
        if not unique:
            return vids
        seen = {}
        for v in vids:
            seen.setdefault(v, None)
        return list(seen.keys())
