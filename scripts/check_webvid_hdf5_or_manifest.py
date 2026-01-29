#!/usr/bin/env python3
import os
import json
import random
import argparse
from pathlib import Path

import h5py


def count_video_entries(root: Path) -> int:
    vm = root / 'FeatureData' / 'video_manifest.json'
    if vm.exists():
        return len(json.load(open(vm)))
    vpath = root / 'FeatureData' / 'new_clip_vit_32_webvid_vid_features.hdf5'
    with h5py.File(vpath, 'r') as hf:
        return len(hf.keys())


def count_text_entries(root: Path) -> int:
    tm = root / 'TextData' / 'text_manifest.json'
    if tm.exists():
        return len(json.load(open(tm)))
    tpath = root / 'TextData' / 'clip_ViT_B_32_webvid_query_feat.hdf5'
    with h5py.File(tpath, 'r') as hf:
        return len(hf.keys())


def count_caption_lines(root: Path, split: str) -> int:
    cap = root / 'TextData' / f'webvid{split}.caption.txt'
    if not cap.exists():
        return 0
    with open(cap, 'r') as f:
        return sum(1 for _ in f)


def sample_verify(root: Path, n=50) -> None:
    vm = root / 'FeatureData' / 'video_manifest.json'
    tm = root / 'TextData' / 'text_manifest.json'
    vmap = json.load(open(vm)) if vm.exists() else None

    cap = root / 'TextData' / 'webvidtrain.caption.txt'
    if not cap.exists():
        print('No train caption file to verify samples')
        return
    lines = [l.strip().split(' ', 1)[0] for l in open(cap).read().strip().splitlines()]
    if not lines:
        print('Caption file is empty')
        return
    picks = random.sample(lines, min(n, len(lines)))
    if vmap is not None:
        missing = [cid for cid in picks if cid.split('#')[0] not in vmap]
        if missing:
            print('Missing video keys in manifest:', missing[:5], '... total:', len(missing))
        else:
            print('OK: sampled videos exist in video_manifest.json')
    else:
        vpath = root / 'FeatureData' / 'new_clip_vit_32_webvid_vid_features.hdf5'
        with h5py.File(vpath, 'r') as hf:
            missing = [cid for cid in picks if cid.split('#')[0] not in hf]
            if missing:
                print('Missing video keys in single HDF5:', missing[:5], '... total:', len(missing))
            else:
                print('OK: sampled videos exist in video HDF5')


def main():
    ap = argparse.ArgumentParser(description='Check WebVid HDF5/manifest stats and sanity')
    ap.add_argument('--root', type=str, default='/home/jaewoo/webvid', help='Output root of converted dataset')
    ap.add_argument('--verify', action='store_true', help='Sample-verify that caption video IDs exist')
    ap.add_argument('--samples', type=int, default=50, help='Number of samples to verify')
    args = ap.parse_args()
    root = Path(args.root)

    v = count_video_entries(root)
    t = count_text_entries(root)
    tr_caps = count_caption_lines(root, 'train')
    te_caps = count_caption_lines(root, 'test')

    print('video entries:', v)
    print('text entries :', t)
    print('caption lines: train', tr_caps, '| test', te_caps)

    if args.verify:
        sample_verify(root, n=args.samples)


if __name__ == '__main__':
    main()

