import os
from torch.utils.data import DataLoader

from Utils.basic_utils import (BigFile, HDF5File, MultiHDF5File, IndexedVideoHDF5, read_dict,
                               ShardedPackedBigFile, Video2FramesView, IndexedTextByJsonIndex,
                               RobertaShardTextReader, InternVideoHDF5, InternVideoTextH5,
                               TVRFrameNPY, SimpleFrameNPY)
from Datasets.samplers import ShardEpochSampler
from Datasets.data_provider import Dataset4PRVR, VisDataSet4PRVR, TxtDataSet4PRVR, \
    Dataset4PRVRWithReader, TxtDataSet4PRVRWithReader, WebVidTxtDatasetForEval, \
    collate_train, collate_frame_val, collate_text_val, read_video_ids, read_video_ids_limited, \
    TVRFramesDataset4PRVR, TVRFramesVisDataSet
from Datasets.msrvtt_dataset import MSRVTTDataset4PRVR, MSRVTTVisDataSet, MSRVTTTxtDataSet, \
    read_video_ids_msrvtt
from Datasets.msrvtt_dataset import read_video_ids_msrvtt, read_video_ids_msrvtt_limited


import json
import h5py
import torch
import time
import sys

def get_datasets(cfg):

    rootpath = cfg['data_root']
    collection = cfg['collection']

    # Debug mode: ensure dataset access happens in main process for "single sample" printing.
    if bool(cfg.get('debug_shapes', False)):
        cfg['num_workers'] = 0
        cfg['pin_memory'] = False

    frames_per_video = int(cfg.get('frames_per_video', 12))

    # split naming
    if collection == 'msrvtt':
        trainCollection = 'msrvtttrain'
        valCollection = 'msrvttval'
        testCollection = 'msrvtttest'
    elif collection in WEBVID_DUMMY_FAMILIES:
        # webvid_dummy*는 train은 기존 webvidtrain, val/test는 dataset명 기반 파일 사용
        trainCollection = 'webvidtrain'
        valCollection = f'{collection}val'
        testCollection = f'{collection}test'
    elif collection == 'webvid':
        trainCollection = 'webvidtrain'
        valCollection = 'webvidtest'
        testCollection = 'webvidtest'
    else:
        trainCollection = f'{collection}train'
        valCollection = f'{collection}val'
        testCollection = f'{collection}test'

    cap_file = {
        'train': f'{trainCollection}.caption.txt',
        'val': f'{valCollection}.caption.txt',
    }

    import json as json_module

    def _load_release_map(path):
        if not path or not os.path.exists(path):
            print(f"[soft_mil] release_map path={path} exists={os.path.exists(path) if path else False}")
            return {}
        release_map = {}
        total = 0
        json_errors = 0
        missing_fields = 0
        first_bad = None
        with open(path, 'r') as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    entry = json_module.loads(line)
                except Exception:
                    json_errors += 1
                    if first_bad is None:
                        first_bad = line[:200]
                    continue
                vid = entry.get('vid_name')
                desc = entry.get('desc')
                ts = entry.get('ts')
                duration = entry.get('duration')
                if vid is None or desc is None or ts is None or duration is None:
                    missing_fields += 1
                    continue
                release_map[(vid, desc)] = (float(ts[0]), float(ts[1]), float(duration))
        print(
            f"[soft_mil] release_map path={path} size={len(release_map)} "
            f"lines={total} json_errors={json_errors} missing_fields={missing_fields}"
        )
        if first_bad is not None:
            print(f"[soft_mil] release_map first_bad={first_bad}")
        return release_map

    def _resolve_caption_path(collection_name, filename):
        path = os.path.join(rootpath, collection_name, 'TextData', filename)
        if os.path.exists(path):
            return path
        alias = WEBVID_DUMMY_TEXT_BASE.get(collection_name)
        if alias and alias != collection_name:
            alias_path = os.path.join(rootpath, alias, 'TextData', filename)
            if os.path.exists(alias_path):
                return alias_path
        return path

    # Resolve full caption paths and detect missing val -> reuse test
    val_cap_full = _resolve_caption_path(collection, cap_file['val'])
    test_cap_full = _resolve_caption_path(collection, f'{testCollection}.caption.txt')
    val_same_as_test = False
    if not os.path.exists(val_cap_full) and os.path.exists(test_cap_full):
        cap_file['val'] = f'{testCollection}.caption.txt'
        val_cap_full = test_cap_full
        val_same_as_test = True

    if collection in WEBVID_DUMMY_FAMILIES and bool(cfg.get('eval', False)):
        val_same_as_test = True

    text_reader = None
    text_feature = cfg.get('text_feature', cfg['visual_feature'])

    # Text feature path (prefer meta; avoid huge JSON manifests in memory)
    if text_feature == 'internvideo2':
        text_feat_path = cfg.get(
            'internvideo2_text_h5',
            os.path.join(rootpath, collection, 'all__samples', 'internvideo2_tvr_all_query_feat.hdf5')
        )
        text_reader = InternVideoTextH5(text_feat_path)
    elif text_feature == 'clip':
        if collection == 'msrvtt':
            text_feat_path = os.path.join(rootpath, collection, 'TextData', 'clip_ViT_B_32_msrvtt_query_feat.hdf5')
        elif collection == 'webvid':
            text_meta = os.path.join(rootpath, collection, 'TextData', 'text_meta.json')
            text_manifest = os.path.join(rootpath, collection, 'TextData', 'text_manifest.json')
            single_text_h5 = os.path.join(rootpath, collection, 'TextData', 'clip_ViT_B_32_webvid_query_feat.hdf5')
            # Prefer meta if available; else manifest (we convert to auto meta later); else single h5
            if os.path.exists(text_meta):
                text_feat_path = text_meta
            elif os.path.exists(text_manifest):
                text_feat_path = text_manifest
            else:
                text_feat_path = single_text_h5
        else:
            text_feat_path = os.path.join(rootpath, collection, 'TextData', f'clip_ViT_B_32_{collection}_query_feat.hdf5')
    else:
        if collection == 'msrvtt':
            text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_msrvtt_query_feat.hdf5')
        elif collection in WEBVID_DUMMY_FAMILIES:
            base_collection = WEBVID_DUMMY_TEXT_BASE.get(collection, collection)
            text_dir = os.path.join(rootpath, base_collection, 'TextData')
            shards_tsv = os.path.join(text_dir, 'roberta_shards.tsv')
            index_path = os.path.join(text_dir, 'webvid_text_index.json')
            if os.path.exists(shards_tsv):
                text_reader = RobertaShardTextReader(text_dir, shards_tsv)
                text_feat_path = shards_tsv
            elif os.path.exists(index_path):
                text_reader = IndexedTextByJsonIndex(text_dir, index_path)
                text_feat_path = index_path
            else:
                raise FileNotFoundError(f"Neither roberta_shards.tsv nor webvid_text_index.json found under {text_dir}")
        else:
            text_feat_path = os.path.join(rootpath, collection, 'TextData', f'roberta_{collection}_query_feat.hdf5')

    caption_files = {x: _resolve_caption_path(collection, cap_file[x]) for x in cap_file}

    # Visual features
    if cfg['visual_feature'] == 'tvr_frames':
        feature_dir = cfg.get('frame_feature_dir', '')
        visual_feat_path = feature_dir
        visual_feats = TVRFrameNPY(feature_dir)
    elif cfg['visual_feature'] == 'act_frames':
        feature_dir = cfg.get('frame_feature_dir', '')
        visual_feat_path = feature_dir
        visual_feats = SimpleFrameNPY(feature_dir)
    elif cfg['visual_feature'] == 'internvideo2':
        visual_feat_path = cfg.get(
            'internvideo2_vid_h5',
            os.path.join(rootpath, collection, 'all__samples', 'internvideo2_tvr_all_vid_features.hdf5')
        )
        visual_feats = InternVideoHDF5(visual_feat_path)
    elif cfg['visual_feature'] == 'clip':
        if collection == 'msrvtt':
            visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', 'new_clip_vit_32_msrvtt_vid_features.hdf5')
            visual_feats = HDF5File(visual_feat_path)
        elif collection == 'webvid':
            # Avoid loading huge manifest into memory; prefer meta or auto-generate a lightweight meta
            feature_dir = os.path.join(rootpath, collection, 'FeatureData')
            video_meta = os.path.join(feature_dir, 'video_meta.json')
            video_manifest = os.path.join(feature_dir, 'video_manifest.json')

            def _auto_build_video_meta():
                import json as _json
                import h5py as _h5py
                # Derive per_shard from first shard
                first_shard = os.path.join(feature_dir, 'video_shard_000.hdf5')
                per_shard = 50000
                if os.path.exists(first_shard):
                    try:
                        with _h5py.File(first_shard, 'r') as hf:
                            per_shard = len(hf.keys())
                    except Exception:
                        per_shard = 50000
                # Derive total from caption file (line count) if available
                train_cap = os.path.join(rootpath, collection, 'TextData', 'webvidtrain.caption.txt')
                total = None
                if os.path.exists(train_cap):
                    try:
                        with open(train_cap, 'r') as fr:
                            total = sum(1 for _ in fr)
                    except Exception:
                        total = None
                # Fallback: estimate from number of shards
                if total is None:
                    shards = [n for n in os.listdir(feature_dir) if n.startswith('video_shard_') and n.endswith('.hdf5')]
                    if len(shards) > 0:
                        # last shard may be partial; optimistic estimate
                        total = (len(shards) - 1) * per_shard + per_shard
                # Write meta next to model (workspace-writable)
                meta = {
                    'per_shard': int(per_shard),
                    'pattern': 'video_shard_%03d.hdf5',
                    'total': int(total) if total is not None else int(per_shard),
                }
                # Save meta JSON to model_root
                save_path = os.path.join(cfg['model_root'], 'video_meta.auto.json')
                with open(save_path, 'w') as f:
                    _json.dump(meta, f)
                return save_path

            if os.path.exists(video_meta):
                visual_feat_path = video_meta
                visual_feats = IndexedVideoHDF5(feature_dir, video_meta)
            elif os.path.exists(video_manifest):
                # Build lightweight meta on the fly to avoid loading massive manifest
                auto_meta = _auto_build_video_meta()
                visual_feat_path = auto_meta
                visual_feats = IndexedVideoHDF5(feature_dir, auto_meta)
            else:
                # Fall back to the single consolidated HDF5 if present
                visual_feat_path = os.path.join(feature_dir, 'new_clip_vit_32_webvid_vid_features.hdf5')
                visual_feats = HDF5File(visual_feat_path)
        else:
            visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', f'new_clip_vit_32_{collection}_vid_features.hdf5')
            visual_feats = HDF5File(visual_feat_path)
    else:
        if collection in WEBVID_DUMMY_FAMILIES:
            feature_dir = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'])
            frames_per_video = int(cfg.get('frames_per_video', 12))
            visual_feat_path = feature_dir
            visual_feats = ShardedPackedBigFile(feature_dir, frames_per_video=frames_per_video)
        else:
            visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'])
            visual_feats = BigFile(visual_feat_path)

    print(f"visual_feats shape {visual_feats.shape()}")
    
    # Load and check text features shape
    if text_reader is not None:
        try:
            if hasattr(text_reader, '_ensure_index'):
                text_reader._ensure_index()
        except Exception:
            pass
        if isinstance(text_reader, InternVideoTextH5):
            try:
                text_reader._ensure()
                qgroup = text_reader.h5file['queries']
                sample_key = list(qgroup.keys())[0]
                sample_shape = qgroup[sample_key]['embeddings'].shape
                print(f"query_feats: {len(qgroup.keys())} videos, each shape {sample_shape}")
            except Exception as e:
                print(f"query_feats: {text_feat_path} (could not read: {e})")
        else:
            entry_cnt = 'unknown'
            if hasattr(text_reader, '_index') and text_reader._index:
                entry_cnt = len(text_reader._index)
            elif hasattr(text_reader, 'total_rows'):
                entry_cnt = text_reader.total_rows
            print(f"query_feats: indexed via {text_feat_path} ({entry_cnt} entries)")
    elif text_feat_path.endswith('.hdf5'):
        try:
            import h5py as h5py_module
            with h5py_module.File(text_feat_path, 'r') as f:
                total_queries = len(f.keys())
                sample_key = list(f.keys())[0]
                query_feat_shape = f[sample_key].shape
                print(f"query_feats: {total_queries} queries, each shape {query_feat_shape}")
        except Exception as e:
            print(f"Could not read query features shape: {e}")
    elif text_feat_path.endswith('.json'):
        try:
            import json as json_module
            with open(text_feat_path, 'r') as f:
                manifest_data = json_module.load(f)
                if isinstance(manifest_data, dict):
                    if 'total' in manifest_data:
                        # meta format
                        total_queries = manifest_data['total']
                        print(f"query_feats loaded from JSON meta: {text_feat_path}, total queries: {total_queries}")
                    else:
                        # manifest format (dict mapping query_id to shard_file)
                        total_queries = len(manifest_data)
                        print(f"query_feats loaded from JSON manifest: {text_feat_path}, total queries: {total_queries}")
                elif isinstance(manifest_data, list):
                    # manifest format (list of files)
                    total_queries = len(manifest_data)
                    print(f"query_feats loaded from JSON manifest: {text_feat_path}, total queries: {total_queries}")
                else:
                    print(f"query_feats loaded from JSON: {text_feat_path}")
        except Exception as e:
            print(f"query_feats loaded from JSON meta: {text_feat_path} (could not read: {e})")
    else:
        print(f"query_feats path: {text_feat_path}")


    try:
        cfg['visual_feat_dim'] = getattr(visual_feats, 'ndims', None) or visual_feats.shape()[-1]
    except Exception:
        cfg['visual_feat_dim'] = int(cfg.get('visual_feat_dim', 512))

    import json as json_module

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

    def _level_segment_count(entry, level_token):
        if not isinstance(entry, dict):
            return 0
        levels = entry.get('levels') or []
        if not isinstance(levels, list):
            return 0
        token, level_num = _parse_level_token(level_token)
        total = 0
        for level_entry in levels:
            if not isinstance(level_entry, dict):
                continue
            if token == 'level' and level_num is not None and level_entry.get('level') != level_num:
                continue
            if token == 'level' and level_num is None:
                continue
            edges = level_entry.get('edges') or []
            if edges:
                total += max(0, len(edges) - 1)
            else:
                peaks = level_entry.get('peaks') or []
                if peaks:
                    total += len(peaks) + 1
        return total

    def _load_boundaries(path):
        if not path or not os.path.exists(path):
            return {}
        try:
            if path.lower().endswith('.jsonl'):
                boundaries = {}
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json_module.loads(line)
                        vid = entry.get('video_id')
                        if vid is None:
                            continue
                        boundaries[vid] = entry
                return boundaries
            with open(path, 'r') as f:
                return json_module.load(f)
        except Exception as e:
            print(f"[builder] Failed to load boundaries from {path}: {e}")
            return {}

    boundary_train = {}
    boundary_val = {}
    release_map_train = {}
    if cfg.get('visual_feature') in {'tvr_frames', 'act_frames'}:
        boundary_level = cfg.get('boundary_level', 'fine')
        boundary_train = _load_boundaries(cfg.get('boundary_train_path'))
        boundary_val = _load_boundaries(cfg.get('boundary_val_path'))
        if not cfg.get('max_segments'):
            def _extract_peaks(entry, level):
                if isinstance(entry, list):
                    return entry
                if isinstance(entry, dict):
                    token, level_num = _parse_level_token(level)
                    if token in ('fine', 'coarse'):
                        if 'fine' in entry or 'coarse' in entry:
                            entry = entry.get(token, {}) or {}
                            return entry.get('peaks', [])
                        if token == 'fine':
                            return entry.get('peaks', [])
                        return []
                    if token in ('levels', 'level'):
                        peaks = []
                        levels = entry.get('levels') or []
                        if isinstance(levels, list):
                            for level_entry in levels:
                                if not isinstance(level_entry, dict):
                                    continue
                                if token == 'level' and level_num is not None and level_entry.get('level') != level_num:
                                    continue
                                if token == 'level' and level_num is None:
                                    continue
                                peaks.extend(level_entry.get('peaks', []) or [])
                        return peaks
                return []

            def _segment_count(entry):
                if boundary_level == 'both':
                    fine = len(_extract_peaks(entry, 'fine')) + 1
                    coarse = len(_extract_peaks(entry, 'coarse')) + 1
                    return fine + coarse
                if '+' in boundary_level:
                    tokens = [t.strip() for t in boundary_level.split('+') if t.strip()]
                    total = 0
                    for token in tokens:
                        token_name, _ = _parse_level_token(token)
                        if token_name in ('fine', 'coarse'):
                            total += len(_extract_peaks(entry, token_name)) + 1
                        elif token_name in ('levels', 'level'):
                            total += _level_segment_count(entry, token)
                    return max(total, 1)
                token_name, _ = _parse_level_token(boundary_level)
                if token_name in ('levels', 'level'):
                    count = _level_segment_count(entry, boundary_level)
                    return count if count > 0 else 1
                return len(_extract_peaks(entry, boundary_level)) + 1

            max_train = max((_segment_count(v) for v in boundary_train.values()), default=1)
            max_val = max((_segment_count(v) for v in boundary_val.values()), default=1)
            cfg['max_segments'] = max(max_train, max_val)
        if bool(cfg.get('use_soft_mil', False)):
            release_map_train = _load_release_map(
                cfg.get('release_train_path') or cfg.get('tvr_release_train_path')
            )

    # video2frames (load once and reuse for test)
    # For non-clip (e.g., i3d BigFile), avoid loading the massive mapping into RAM in eval mode.
    if cfg['visual_feature'] in {'tvr_frames', 'act_frames'}:
        video2frames = None
        if cfg['visual_feature'] == 'act_frames':
            video2frames_path = cfg.get('video2frames_path') or os.path.join(
                rootpath, collection, 'FeatureData', 'i3d', 'video2frames.txt'
            )
            if os.path.exists(video2frames_path):
                video2frames = read_dict(video2frames_path)
    elif cfg['visual_feature'] == 'internvideo2':
        video2frames = {}
        try:
            vgroup = visual_feats.h5file['videos']
            for vid in vgroup.keys():
                dset = vgroup[vid]['clip_embeddings']
                video2frames[vid] = [f"{vid}_{i}" for i in range(dset.shape[0])]
        except Exception as e:
            print(f"[builder] Failed to build internvideo2 video2frames: {e}")
            video2frames = None
    elif cfg['visual_feature'] == 'clip':
        if collection == 'msrvtt':
            video2frames_path = os.path.join(rootpath, collection, 'FeatureData', 'clip', 'video2frames.txt')
        elif collection == 'tvr':
            video2frames_path = os.path.join(rootpath, collection, 'FeatureData', 'i3d_resnet', 'video2frames.txt')
        elif collection == 'webvid':
            video2frames_path = None
        else:
            video2frames_path = os.path.join(rootpath, collection, 'FeatureData', 'i3d', 'video2frames.txt')
        video2frames = read_dict(video2frames_path) if video2frames_path is not None else None
    else:
        # non-clip path (BigFile 등)
        if collection in WEBVID_DUMMY_FAMILIES:
            video2frames = Video2FramesView(frames_per_video=frames_per_video)
        else:
            # Restrict lazy id.txt scanning to WebVid(-10M) only; for act/tvr/others use video2frames.txt
            lazy_eval = bool(cfg.get('eval', False)) and collection in ({'webvid', 'webvid-10m', 'webvid_10m', 'webvid10m'} | WEBVID_DUMMY_FAMILIES)
            if not lazy_eval:
                video2frames_path = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt')
                video2frames = read_dict(video2frames_path)
            else:
                # Build a lightweight mapping only for needed IDs (val/test and optional limited train for search)
                if collection == 'msrvtt':
                    read_ids = read_video_ids_msrvtt
                    read_ids_limited = read_video_ids_msrvtt_limited
                else:
                    read_ids = read_video_ids
                    read_ids_limited = read_video_ids_limited

                val_cap_path = _resolve_caption_path(collection, f'{valCollection}.caption.txt')
                test_cap_path = _resolve_caption_path(collection, f'{testCollection}.caption.txt')
                val_ids = set(read_ids(val_cap_path)) if os.path.exists(val_cap_path) else set()
                test_ids = set(read_ids(test_cap_path)) if os.path.exists(test_cap_path) else set()

                need_train = bool(cfg.get('search', False)) and int(cfg.get('search_limit', 0)) > 0
                train_ids = set()
                if need_train:
                    train_cap_path = _resolve_caption_path(collection, f'{trainCollection}.caption.txt')
                    if os.path.exists(train_cap_path):
                        train_ids = set(read_ids_limited(train_cap_path, int(cfg.get('search_limit', 0))))
                need_train_sampling = need_train and (len(train_ids) == 0)
                train_limit = int(cfg.get('search_limit', 0)) if need_train_sampling else 0

                target_ids = set()
                target_ids.update(val_ids)
                target_ids.update(test_ids)
                target_ids.update(train_ids)

                id_path = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'id.txt')
                video2frames = {vid: [] for vid in target_ids}
                max_per_video = max(int(cfg.get('max_ctx_l', 128)) * 4, 512)
                try:
                    required_vids = set(val_ids) | set(test_ids)
                    done_vids = set(x for x in required_vids if video2frames.get(x) and len(video2frames[x]) >= max_per_video)
                    train_added = len(train_ids)
                    print(f"[builder] Building lazy video2frames from {id_path}: val={len(val_ids)} test={len(test_ids)} train_limit={train_limit} max_per_video={max_per_video}")
                    sys.stdout.flush()

                    lines = 0
                    next_log = time.time() + 2.0

                    with open(id_path, 'r', encoding='ISO-8859-1', errors='ignore') as fr:
                        for line in fr:
                            lines += 1
                            name = line.strip().split()[0] if line else None
                            if not name:
                                continue
                            vid = None
                            if name in target_ids:
                                vid = name
                            else:
                                us = name.rfind('_')
                                if us > 0:
                                    cand = name[:us]
                                    if cand in target_ids:
                                        vid = cand
                                if vid is None:
                                    sl = name.rfind('/')
                                    if sl > 0:
                                        cand = name[:sl]
                                        if cand in target_ids:
                                            vid = cand
                            if vid is None and need_train_sampling and train_added < train_limit:
                                base = None
                                us = name.rfind('_')
                                if us > 0:
                                    base = name[:us]
                                if base is None:
                                    sl = name.rfind('/')
                                    if sl > 0:
                                        base = name[:sl]
                                if base and base not in val_ids and base not in test_ids:
                                    if base not in video2frames:
                                        video2frames[base] = []
                                    if len(video2frames[base]) < max_per_video:
                                        video2frames[base].append(name)
                                        train_ids.add(base)
                                        train_added += 1
                                continue
                            if vid is None:
                                continue
                            lst = video2frames.get(vid)
                            if lst is None:
                                continue
                            if len(lst) < max_per_video:
                                lst.append(name)
                                if vid in required_vids and len(lst) >= max_per_video:
                                    done_vids.add(vid)
                            if time.time() >= next_log:
                                print(f"[builder] id.txt progress: lines={lines:,} done_valtest={len(done_vids)}/{len(required_vids)} train_added={train_added}/{train_limit}")
                                sys.stdout.flush()
                                next_log = time.time() + 2.0
                            if (not required_vids or len(done_vids) == len(required_vids)) and (not need_train_sampling or train_added >= train_limit):
                                break
                    video2frames = {k: v for k, v in video2frames.items() if v}
                except Exception as e:
                    print(f"Failed to build lazy video2frames index from {id_path}: {e}")
                    video2frames = None

    # Datasets
    build_train = not bool(cfg.get('eval', False))
    val_text_dataset = None
    val_video_dataset = None
    if collection == 'msrvtt':
        if build_train:
            train_dataset = MSRVTTDataset4PRVR(caption_files['train'], visual_feats, text_feat_path, cfg, video2frames=video2frames)
        if not val_same_as_test:
            val_text_dataset = MSRVTTTxtDataSet(caption_files['val'], text_feat_path, cfg)
            val_video_ids_list = read_video_ids_msrvtt(caption_files['val'])
            val_video_dataset = MSRVTTVisDataSet(visual_feats, video2frames, cfg, video_ids=val_video_ids_list)
    elif collection in WEBVID_DUMMY_FAMILIES:
        if text_reader is None:
            raise RuntimeError(f"{collection} requires text_reader")
        if build_train:
            train_dataset = Dataset4PRVRWithReader(caption_files['train'], visual_feats, text_reader, cfg, video2frames=video2frames)
        if not val_same_as_test and not bool(cfg.get('eval', False)):
            val_text_dataset = TxtDataSet4PRVRWithReader(caption_files['val'], text_reader, cfg)
            val_video_ids_list = read_video_ids(caption_files['val'])
            val_video_dataset = VisDataSet4PRVR(visual_feats, video2frames, cfg, video_ids=val_video_ids_list)
    elif collection == 'webvid' and (os.path.exists(os.path.join(rootpath, collection, 'FeatureData', 'video_meta.json')) \
                                     or os.path.exists(os.path.join(rootpath, collection, 'FeatureData', 'video_manifest.json')) \
                                     or os.path.exists(os.path.join(rootpath, collection, 'FeatureData', 'video_shard_000.hdf5'))):
        # Use aligned WebVid dataset to avoid loading massive caption text into memory
        import json
        from Utils.basic_utils import IndexedTextH5, MultiTextH5
        from Datasets.webvid_aligned_dataset import WebVidAlignedDataset4PRVR

        video_meta_path = os.path.join(rootpath, collection, 'FeatureData', 'video_meta.json')
        video_manifest_path = os.path.join(rootpath, collection, 'FeatureData', 'video_manifest.json')
    
        if os.path.exists(video_meta_path):
            with open(video_meta_path, 'r') as f:
                vmeta = json.load(f)
            total_train_videos = int(vmeta.get('total', 0))
        else:
            # Fall back to counting lines in caption file (cheap and constant memory)
            train_cap = os.path.join(rootpath, collection, 'TextData', 'webvidtrain.caption.txt')
            try:
                with open(train_cap, 'r') as fr:
                    total_train_videos = sum(1 for _ in fr)
            except Exception:
                # Final fallback to the reader's reported size
                total_train_videos = int(getattr(visual_feats, 'nr_of_images', 0))

        # Build text reader based on available metadata
        if text_feat_path.endswith('text_meta.json'):
            text_reader = IndexedTextH5(os.path.join(rootpath, collection, 'TextData'), text_feat_path)
        elif text_feat_path.endswith('text_manifest.json'):
            # Build lightweight meta on the fly to avoid loading massive manifest into memory
            import json as _json
            import h5py as _h5py
            text_dir = os.path.join(rootpath, collection, 'TextData')
            first_shard = os.path.join(text_dir, 'text_shard_000.hdf5')
            per_shard = 500000
            if os.path.exists(first_shard):
                try:
                    with _h5py.File(first_shard, 'r') as hf:
                        per_shard = len(hf.keys())
                except Exception:
                    per_shard = 500000
            train_cap = os.path.join(rootpath, collection, 'TextData', 'webvidtrain.caption.txt')
            total = None
            try:
                with open(train_cap, 'r') as fr:
                    total = sum(1 for _ in fr)
            except Exception:
                total = None
            if total is None:
                # estimate from number of shards
                shards = [n for n in os.listdir(text_dir) if n.startswith('text_shard_') and n.endswith('.hdf5')]
                if len(shards) > 0:
                    total = (len(shards) - 1) * per_shard + per_shard
            meta = {
                'per_shard': int(per_shard),
                'pattern': 'text_shard_%03d.hdf5',
                'total': int(total) if total is not None else int(per_shard)
            }
            save_path = os.path.join(cfg['model_root'], 'text_meta.auto.json')
            with open(save_path, 'w') as f:
                _json.dump(meta, f)
            text_reader = IndexedTextH5(text_dir, save_path)
        else:
            import h5py
            class SingleTextH5:
                def __init__(self, p):
                    self._p = p
                    self._h5 = None
                def _ensure(self):
                    if self._h5 is None:
                        self._h5 = h5py.File(self._p, 'r')
                def get(self, k):
                    self._ensure()
                    return self._h5[k][...]
            text_reader = SingleTextH5(text_feat_path)

        train_dataset = WebVidAlignedDataset4PRVR(visual_feats, text_reader, cfg, total_train_videos)

        # Validation uses the provided webvidtest captions (small)
        val_cap = os.path.join(rootpath, collection, 'TextData', 'webvidtest.caption.txt')
        val_text_dataset = MSRVTTTxtDataSet(val_cap, text_feat_path, cfg)
        val_video_ids_list = read_video_ids_msrvtt(val_cap)
        val_video_dataset = MSRVTTVisDataSet(visual_feats, video2frames, cfg, video_ids=val_video_ids_list)
    else:
        if build_train:
            if cfg.get('visual_feature') in {'tvr_frames', 'act_frames'}:
                train_dataset = TVRFramesDataset4PRVR(
                    caption_files['train'], visual_feats, text_feat_path, cfg,
                    boundaries=boundary_train, release_map=release_map_train
                )
                print(f"[builder] {cfg.get('visual_feature')} train videos: {len(train_dataset.video_ids)}")
            elif text_reader is not None:
                train_dataset = Dataset4PRVRWithReader(caption_files['train'], visual_feats, text_reader, cfg, video2frames=video2frames)
            else:
                train_dataset = Dataset4PRVR(caption_files['train'], visual_feats, text_feat_path, cfg, video2frames=video2frames)
        if not val_same_as_test:
            if text_reader is not None:
                val_text_dataset = TxtDataSet4PRVRWithReader(caption_files['val'], text_reader, cfg)
            else:
                val_text_dataset = TxtDataSet4PRVR(caption_files['val'], text_feat_path, cfg)
            val_video_ids_list = read_video_ids(caption_files['val'])
            if cfg.get('visual_feature') in {'tvr_frames', 'act_frames'}:
                val_video_dataset = TVRFramesVisDataSet(
                    visual_feats, cfg, video_ids=val_video_ids_list, boundaries=boundary_val
                )
                print(f"[builder] {cfg.get('visual_feature')} val videos: {len(val_video_ids_list)}")
            else:
                val_video_dataset = VisDataSet4PRVR(visual_feats, video2frames, cfg, video_ids=val_video_ids_list)

    search_limit = int(cfg.get('search_limit', 0)) if isinstance(cfg, dict) else 0
    eval_query_limit = int(cfg.get('eval_query_limit', 0)) if isinstance(cfg, dict) else 0

    # Test split
    test_cap_file = {'test': f'{testCollection}.caption.txt'}
    test_caption_files = {x: _resolve_caption_path(collection, test_cap_file[x]) for x in test_cap_file}

    if cfg['visual_feature'] == 'tvr_frames':
        test_visual_feat_path = cfg.get('frame_feature_dir', '')
        test_visual_feats = TVRFrameNPY(test_visual_feat_path)
    elif cfg['visual_feature'] == 'act_frames':
        test_visual_feat_path = cfg.get('frame_feature_dir', '')
        test_visual_feats = SimpleFrameNPY(test_visual_feat_path)
    elif cfg['visual_feature'] == 'internvideo2':
        test_visual_feat_path = cfg.get(
            'internvideo2_vid_h5',
            os.path.join(rootpath, collection, 'all__samples', 'internvideo2_tvr_all_vid_features.hdf5')
        )
        test_visual_feats = InternVideoHDF5(test_visual_feat_path)
    elif cfg['visual_feature'] == 'clip':
        if collection == 'msrvtt':
            test_visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', 'new_clip_vit_32_msrvtt_vid_features.hdf5')
            test_visual_feats = HDF5File(test_visual_feat_path)
        elif collection == 'webvid':
            video_meta = os.path.join(rootpath, collection, 'FeatureData', 'video_meta.json')
            video_manifest = os.path.join(rootpath, collection, 'FeatureData', 'video_manifest.json')
            if os.path.exists(video_meta):
                test_visual_feat_path = video_meta
                test_visual_feats = IndexedVideoHDF5(os.path.join(rootpath, collection, 'FeatureData'), video_meta)
            elif os.path.exists(video_manifest):
                test_visual_feat_path = video_manifest
                test_visual_feats = MultiHDF5File(video_manifest)
            else:
                test_visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', 'new_clip_vit_32_webvid_vid_features.hdf5')
                test_visual_feats = HDF5File(test_visual_feat_path)
        else:
            test_visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', f'new_clip_vit_32_{collection}_vid_features.hdf5')
            test_visual_feats = HDF5File(test_visual_feat_path)
    else:
        if collection in WEBVID_DUMMY_FAMILIES:
            test_visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'])
            test_visual_feats = ShardedPackedBigFile(test_visual_feat_path, frames_per_video=frames_per_video)
        else:
            test_visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'])
            test_visual_feats = BigFile(test_visual_feat_path)

    # Test video2frames (reuse the already loaded video2frames)
    test_video2frames = video2frames

    # Test datasets
    if collection == 'msrvtt':
        test_video_ids_list = read_video_ids_msrvtt(test_caption_files['test'])
        test_vid_dataset = MSRVTTVisDataSet(test_visual_feats, test_video2frames, cfg, video_ids=test_video_ids_list)
        test_text_dataset = MSRVTTTxtDataSet(test_caption_files['test'], text_feat_path, cfg)
    elif collection == 'webvid':
        test_video_ids_list = read_video_ids_msrvtt(test_caption_files['test'])
        test_vid_dataset = MSRVTTVisDataSet(test_visual_feats, test_video2frames, cfg, video_ids=test_video_ids_list)
        test_text_dataset = MSRVTTTxtDataSet(test_caption_files['test'], text_feat_path, cfg)
    elif collection in WEBVID_DUMMY_FAMILIES:
        if bool(cfg.get('eval', False)):
            if search_limit > 0:
                test_video_ids_list = read_video_ids_limited(test_caption_files['test'], search_limit)
            else:
                test_video_ids_list = read_video_ids(test_caption_files['test'])
            # Preserve order but ensure uniqueness
            seen = set()
            filtered_ids = []
            for vid in test_video_ids_list:
                if vid in seen:
                    continue
                seen.add(vid)
                filtered_ids.append(vid)
            test_video_ids_list = filtered_ids
            if not test_video_ids_list:
                raise RuntimeError("No test video ids found for webvid_dummy evaluation")
            test_vid_dataset = VisDataSet4PRVR(test_visual_feats, test_video2frames, cfg, video_ids=test_video_ids_list)
            base_collection = WEBVID_DUMMY_TEXT_BASE.get(collection, collection)
            text_feat_dir = os.path.join(rootpath, base_collection, 'TextData')
            effective_limit = len(test_video_ids_list)
            if eval_query_limit > 0:
                effective_limit = min(effective_limit, eval_query_limit)
            test_text_dataset = WebVidTxtDatasetForEval(
                test_caption_files['test'],
                text_feat_dir,
                cfg,
                video_ids=set(test_video_ids_list),
                limit=effective_limit
            )
        else:
            test_video_ids_list = read_video_ids(test_caption_files['test'])
            if cfg.get('visual_feature') in {'tvr_frames', 'act_frames'}:
                test_vid_dataset = TVRFramesVisDataSet(
                    test_visual_feats, cfg, video_ids=test_video_ids_list, boundaries=boundary_val
                )
            else:
                test_vid_dataset = VisDataSet4PRVR(test_visual_feats, test_video2frames, cfg, video_ids=test_video_ids_list)
            if text_reader is None:
                raise RuntimeError(f"{collection} requires text_reader for test dataset")
            test_text_dataset = TxtDataSet4PRVRWithReader(test_caption_files['test'], text_reader, cfg)
    else:
        test_video_ids_list = read_video_ids(test_caption_files['test'])
        if cfg.get('visual_feature') in {'tvr_frames', 'act_frames'}:
            test_vid_dataset = TVRFramesVisDataSet(
                test_visual_feats, cfg, video_ids=test_video_ids_list, boundaries=boundary_val
            )
            print(f"[builder] {cfg.get('visual_feature')} test videos: {len(test_video_ids_list)}")
        else:
            test_vid_dataset = VisDataSet4PRVR(test_visual_feats, test_video2frames, cfg, video_ids=test_video_ids_list)
        if text_reader is not None:
            test_text_dataset = TxtDataSet4PRVRWithReader(test_caption_files['test'], text_reader, cfg)
        else:
            test_text_dataset = TxtDataSet4PRVR(test_caption_files['test'], text_feat_path, cfg)

    # If val is same as test, reuse test datasets to avoid duplicate loading
    if val_same_as_test:
        val_video_dataset = test_vid_dataset
        val_text_dataset = test_text_dataset


    # Loaders
    # Optional shard-aware sampler to bound per-epoch sample size
    shard_size = int(cfg.get('train_shard_size', 0)) if isinstance(cfg, dict) else 0
    if bool(cfg.get('eval', False)):
        # Build a minimal train_loader to avoid unnecessary RAM usage in eval-only runs
        from torch.utils.data import Dataset
        class _EmptyDataset(Dataset):
            def __len__(self): return 0
            def __getitem__(self, idx): raise IndexError
        train_loader = DataLoader(dataset=_EmptyDataset(), batch_size=cfg['batchsize'], shuffle=False,
                                  pin_memory=cfg['pin_memory'], num_workers=0, collate_fn=lambda x: x)
    else:
        if shard_size and shard_size > 0:
            sampler = ShardEpochSampler(len(train_dataset), shard_size, seed=cfg.get('seed', 0))
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=cfg['batchsize'],
                shuffle=False,
                sampler=sampler,
                pin_memory=cfg['pin_memory'],
                num_workers=cfg['num_workers'],
                persistent_workers=bool(cfg.get('persistent_workers', True)) and cfg['num_workers'] > 0,
                prefetch_factor=int(cfg.get('prefetch_factor', 2)),
                collate_fn=collate_train,
            )
        else:
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=cfg['batchsize'],
                shuffle=True,
                pin_memory=cfg['pin_memory'],
                num_workers=cfg['num_workers'],
                persistent_workers=bool(cfg.get('persistent_workers', True)) and cfg['num_workers'] > 0,
                prefetch_factor=int(cfg.get('prefetch_factor', 2)),
                collate_fn=collate_train,
            )
    context_dataloader = DataLoader(
        val_video_dataset,
        collate_fn=collate_frame_val,
        batch_size=cfg['eval_context_bsz'],
        num_workers=cfg['num_workers'],
        shuffle=False,
        pin_memory=cfg['pin_memory'],
        persistent_workers=bool(cfg.get('persistent_workers', True)) and cfg['num_workers'] > 0,
        prefetch_factor=int(cfg.get('prefetch_factor', 2)),
    )
    query_eval_loader = DataLoader(
        val_text_dataset,
        collate_fn=collate_text_val,
        batch_size=cfg['eval_query_bsz'],
        num_workers=cfg['num_workers'],
        shuffle=False,
        pin_memory=cfg['pin_memory'],
        persistent_workers=bool(cfg.get('persistent_workers', True)) and cfg['num_workers'] > 0,
        prefetch_factor=int(cfg.get('prefetch_factor', 2)),
    )
    test_context_dataloader = DataLoader(
        test_vid_dataset,
        collate_fn=collate_frame_val,
        batch_size=cfg['eval_context_bsz'],
        num_workers=cfg['num_workers'],
        shuffle=False,
        pin_memory=cfg['pin_memory'],
        persistent_workers=bool(cfg.get('persistent_workers', True)) and cfg['num_workers'] > 0,
        prefetch_factor=int(cfg.get('prefetch_factor', 2)),
    )
    test_query_eval_loader = DataLoader(
        test_text_dataset,
        collate_fn=collate_text_val,
        batch_size=cfg['eval_query_bsz'],
        num_workers=cfg['num_workers'],
        shuffle=False,
        pin_memory=cfg['pin_memory'],
        persistent_workers=bool(cfg.get('persistent_workers', True)) and cfg['num_workers'] > 0,
        prefetch_factor=int(cfg.get('prefetch_factor', 2)),
    )

    return cfg, train_loader, context_dataloader, query_eval_loader, test_context_dataloader, test_query_eval_loader
WEBVID_DUMMY_FAMILIES = {'webvid_dummy', 'webvid_dummy_18'}
WEBVID_DUMMY_TEXT_BASE = {'webvid_dummy': 'webvid_dummy', 'webvid_dummy_18': 'webvid_dummy'}
