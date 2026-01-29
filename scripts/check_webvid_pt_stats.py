#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path

import torch


def shard_paths(base: Path, split: str, kind: str):
    return sorted(Path(p) for p in glob.glob(str(base / f"{split}_{kind}_*.pt")))


def count_items(obj):
    if isinstance(obj, torch.Tensor):
        return int(obj.shape[0])
    if isinstance(obj, dict):
        return len(obj)
    if isinstance(obj, (list, tuple)):
        return len(obj)
    return 0


def sum_sizes(paths):
    total = 0
    for p in paths:
        obj = torch.load(p, map_location='cpu')
        total += count_items(obj)
    return total


def max_indices(index_paths):
    m0 = m1 = -1
    for p in index_paths:
        t = torch.load(p, map_location='cpu')
        if not isinstance(t, torch.Tensor):
            continue
        t = t.long().view(-1, t.shape[-1])
        if t.shape[1] < 2:
            continue
        m0 = max(m0, int(t[:, 0].max()))
        m1 = max(m1, int(t[:, 1].max()))
    return (m0 + 1 if m0 >= 0 else 0), (m1 + 1 if m1 >= 0 else 0)


def infer_index_roles(idx_paths, text_total, video_total):
    if not idx_paths:
        return None
    c0, c1 = max_indices(idx_paths)
    # choose which column is cap vs vid
    s01 = abs(c0 - text_total) + abs(c1 - video_total)
    s10 = abs(c1 - text_total) + abs(c0 - video_total)
    cap_col, vid_col = (0, 1) if s01 <= s10 else (1, 0)
    cap_count, vid_count = (c0, c1) if cap_col == 0 else (c1, c0)
    return dict(cap_col=cap_col, vid_col=vid_col, cap_count=cap_count, vid_count=vid_count)


def main():
    ap = argparse.ArgumentParser(description='Check WebVid PT shard stats')
    ap.add_argument('--pt_dir', type=str, default='~/webvid/output', help='Directory with PT shards')
    args = ap.parse_args()
    base = Path(os.path.expanduser(args.pt_dir))

    tr_text = shard_paths(base, 'train', 'text')
    tr_vid = shard_paths(base, 'train', 'video')
    tr_idx = shard_paths(base, 'train', 'index')
    te_text = shard_paths(base, 'test', 'text')
    te_vid = shard_paths(base, 'test', 'video')
    te_idx = shard_paths(base, 'test', 'index')

    print(f'train shards: text={len(tr_text)} video={len(tr_vid)} index={len(tr_idx)}')
    print(f'test  shards: text={len(te_text)} video={len(te_vid)} index={len(te_idx)}')

    train_text_total = sum_sizes(tr_text)
    train_video_total = sum_sizes(tr_vid)
    test_text_total = sum_sizes(te_text)
    test_video_total = sum_sizes(te_vid)
    print(f'PT totals → train: text={train_text_total:,} video={train_video_total:,} | '
          f'test: text={test_text_total:,} video={test_video_total:,}')

    print('train index info:', infer_index_roles(tr_idx, train_text_total, train_video_total))
    print('test  index info:', infer_index_roles(te_idx, test_text_total, test_video_total))


if __name__ == '__main__':
    main()

