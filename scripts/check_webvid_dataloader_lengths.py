#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path


def add_src_to_path():
    # repo_root/src
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    src_path = repo_root / 'src'
    sys.path.insert(0, str(src_path))


def main():
    ap = argparse.ArgumentParser(description='Check GMMFormer dataloader lengths for webvid config')
    ap.add_argument('--dataset', type=str, default='webvid', help='Dataset name (default webvid)')
    args = ap.parse_args()

    add_src_to_path()
    from Configs.builder import get_configs
    from Datasets.builder import get_datasets

    cfg = get_configs(args.dataset)
    cfg, train_loader, context_dataloader, query_eval_loader, test_context_dataloader, test_query_eval_loader = get_datasets(cfg)

    print('train videos:', len(train_loader.dataset))
    print('val   videos:', len(context_dataloader.dataset))
    print('val   caps  :', len(query_eval_loader.dataset))
    print('test  videos:', len(test_context_dataloader.dataset))
    print('test  caps  :', len(test_query_eval_loader.dataset))


if __name__ == '__main__':
    main()

