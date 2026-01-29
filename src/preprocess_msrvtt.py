#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSRVTT → GMMFormer 전처리 (고속 버전)
- Raw 비디오에서 CLIP(ViT-B/32) 프레임 임베딩 추출 (최대 max_frames개)
- 캡션 텍스트 임베딩 배치 추출
- HDF5: 하나의 큰 dataset(features)와 보조 인덱스(keys/video_ids)로 저장
- 스플릿별 caption 파일 및 video2frames.txt 생성

사용 예:
python preprocess_msrvtt_gmmformer_fast.py \
  --msrvtt_data /disk/gjw/msr-vtt/MSRVTT_data.json \
  --csv_path /disk/gjw/msr-vtt/MSRVTT_JSFUSION_train_test_10k.csv \
  --video_dir /disk/gjw/msr-vtt/MSRVTT_Videos \
  --output_dir /disk/gjw/msrvtt \
  --max_frames 12 --device cuda --text_batch 1024 --amp fp16 --overwrite
"""

import os
import json
import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse
import warnings

# CLIP (Transformers)
from transformers import CLIPModel, CLIPTokenizer

# Video I/O
from decord import VideoReader, cpu as decord_cpu
try:
    from decord import gpu as decord_gpu, bridge as decord_bridge
    DECORD_HAS_BRIDGE = True
except Exception:
    DECORD_HAS_BRIDGE = False

# -----------------------------------------------------------------------------
# 유틸
# -----------------------------------------------------------------------------
def set_torch_opts():
    torch.backends.cudnn.benchmark = True  # 변동 없는 입력 크기에서 유리
    warnings.filterwarnings("ignore", category=UserWarning, module="h5py")

def load_msrvtt_annotations(msrvtt_data_path: str, csv_path: str) -> Tuple[Dict, List]:
    """MSRVTT JSON/CSV 로드 및 결합된 캡션 목록 반환."""
    with open(msrvtt_data_path, 'r') as f:
        msrvtt_data = json.load(f)

    df = pd.read_csv(csv_path)

    # video_id -> meta
    video_info = {}
    for v in msrvtt_data['videos']:
        video_info[v['video_id']] = {
            'split': v.get('split', 'unknown'),
            'category': v.get('category', ''),
            'url': v.get('url', ''),
        }

    # captions_data: [{video_id, caption, key, split}, ...]
    captions_data = []
    # 컬럼명 방어적 처리
    sent_col = 'sentence' if 'sentence' in df.columns else 'caption' if 'caption' in df.columns else None
    if sent_col is None:
        raise ValueError(f"CSV에 'sentence' 또는 'caption' 컬럼이 없습니다: {list(df.columns)}")

    key_col = 'key' if 'key' in df.columns else None
    if key_col is None:
        # JSFUSION csv에는 보통 key가 있음. 없으면 고유 키를 합성
        warnings.warn("CSV에 'key' 컬럼이 없어 video_id_인덱스로 대체합니다.")
        df = df.copy()
        df['key'] = [f"{vid}_{i}" for i, vid in enumerate(df['video_id'])]
        key_col = 'key'

    for _, row in df.iterrows():
        vid = str(row['video_id'])
        cap = str(row[sent_col])
        key = str(row[key_col])
        split = video_info.get(vid, {}).get('split', 'unknown')
        captions_data.append({'video_id': vid, 'caption': cap, 'key': key, 'split': split})

    return video_info, captions_data

def sample_frame_indices(total_frames: int, max_frames: int) -> torch.Tensor:
    if total_frames <= 0:
        return torch.zeros(0, dtype=torch.long)
    if total_frames <= max_frames:
        return torch.arange(total_frames, dtype=torch.long)
    return torch.linspace(0, total_frames - 1, steps=max_frames).long()

def _resize_shorter_side(frames: torch.Tensor, target: int = 224) -> torch.Tensor:
    # frames: (T, C, H, W) float32 on device
    import torch.nn.functional as F
    _, _, H, W = frames.shape
    if H == 0 or W == 0:
        return frames
    scale = float(target) / float(min(H, W))
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    frames = F.interpolate(frames, size=(new_h, new_w), mode='bicubic', align_corners=False)
    return frames

def _center_crop(frames: torch.Tensor, size: int = 224) -> torch.Tensor:
    # frames: (T, C, H, W)
    _, _, H, W = frames.shape
    if H < size or W < size:
        # 패딩 대신 간단히 리사이즈에서 보정되므로 여기선 안전하게 슬라이스 클램프
        size_h = min(size, H)
        size_w = min(size, W)
    else:
        size_h = size_w = size
    top = (H - size_h) // 2
    left = (W - size_w) // 2
    return frames[:, :, top:top + size_h, left:left + size_w]

def preprocess_frames_tensor(frames: torch.Tensor, device: torch.device, target: int = 224) -> torch.Tensor:
    """
    frames: (T, H, W, C) uint8 CPU -> returns (T, 3, 224, 224) float32 on device, CLIP 정규화 적용
    """
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
    frames = frames.to(device=device, dtype=torch.float32) / 255.0
    frames = _resize_shorter_side(frames, target=target)
    frames = _center_crop(frames, size=target)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return frames

# -----------------------------------------------------------------------------
# 특징 추출
# -----------------------------------------------------------------------------
def extract_clip_text_features_batched(
    captions: List[str],
    model: CLIPModel,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    batch_size: int = 512,
    amp: str = "fp16",
) -> np.ndarray:
    amp = amp.lower()
    use_autocast = (device.type == 'cuda' and amp in ("fp16", "bf16"))
    autocast_dtype = torch.float16 if amp == "fp16" else torch.bfloat16

    feats_chunks = []
    rng = range(0, len(captions), batch_size)
    for i in tqdm(rng, desc="텍스트 임베딩(배치)"):
        batch = captions[i: i + batch_size]
        toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                    f = model.get_text_features(**toks)  # (B, D)
            else:
                f = model.get_text_features(**toks)
        feats_chunks.append(f.float().cpu().numpy())
    return np.concatenate(feats_chunks, axis=0)

def extract_clip_features_from_video(
    video_path: str,
    model: CLIPModel,
    device: torch.device,
    max_frames: int = 12,
    amp: str = "fp16",
) -> np.ndarray:
    """
    1개 비디오에서 최대 max_frames개의 프레임 임베딩 추출 → (max_frames, 512)
    비디오 없거나 실패하면 제로 반환.
    """
    if not os.path.exists(video_path):
        return np.zeros((max_frames, 512), dtype=np.float32)

    # Decord bridge: torch 텐서 바로 받기
    if DECORD_HAS_BRIDGE:
        try:
            decord_bridge.set_bridge('torch')
        except Exception:
            pass

    try:
        ctx = decord_cpu(0)  # GPU 디코딩 빌드가 없다면 CPU 사용
        # GPU 디코딩 가능 환경이면 아래 주석을 해제하세요.
        # if device.type == 'cuda':
        #     ctx = decord_gpu(0)

        vr = VideoReader(video_path, ctx=ctx)
        total = len(vr)
        if total == 0:
            return np.zeros((max_frames, 512), dtype=np.float32)

        idx = sample_frame_indices(total, max_frames)
        frames = vr.get_batch(idx)  # torch.Tensor (T, H, W, C) 또는 ndarray
        if not torch.is_tensor(frames):
            frames = torch.from_numpy(frames)

        frames = preprocess_frames_tensor(frames, device=device, target=224)

        amp = amp.lower()
        use_autocast = (device.type == 'cuda' and amp in ("fp16", "bf16"))
        autocast_dtype = torch.float16 if amp == "fp16" else torch.bfloat16

        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                    f = model.get_image_features(pixel_values=frames)  # (T, 512)
            else:
                f = model.get_image_features(pixel_values=frames)
        f = f.float().cpu().numpy()

        # pad/trunc to max_frames
        if f.shape[0] < max_frames:
            pad = np.zeros((max_frames - f.shape[0], f.shape[1]), dtype=np.float32)
            f = np.vstack([f, pad])
        elif f.shape[0] > max_frames:
            f = f[:max_frames]

        return f.astype(np.float32)

    except Exception as e:
        print(f"[경고] 비디오 처리 실패: {video_path} ({e})")
        return np.zeros((max_frames, 512), dtype=np.float32)

# -----------------------------------------------------------------------------
# HDF5 저장 (큰 dataset 한 번에)
# -----------------------------------------------------------------------------
def write_text_hdf5(out_path: Path, keys: List[str], feats: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as hf:
        N, D = feats.shape
        hf.create_dataset('features', data=feats, dtype='f4', chunks=(min(1024, N), D))
        dt = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset('keys', data=np.array(keys, dtype=object), dtype=dt)

def write_video_hdf5_rowwise(out_path: Path, video_ids: List[str], max_frames: int, D: int = 512):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    V = len(video_ids)
    with h5py.File(out_path, 'w') as hf:
        feats = hf.create_dataset('features',
                                  shape=(V, max_frames, D),
                                  dtype='f4',
                                  chunks=(1, max_frames, D))  # 비디오 단위 쓰기
        dt = h5py.string_dtype(encoding='utf-8')
        vids = hf.create_dataset('video_ids', shape=(V,), dtype=dt)
        vids[:] = np.array(video_ids, dtype=object)
    return True  # 생성 완료

def fill_video_hdf5(out_path: Path, video_ids: List[str], features_iter):
    """
    features_iter: yield np.ndarray (max_frames, 512) 순서 = video_ids 순서
    """
    with h5py.File(out_path, 'r+') as hf:
        feats = hf['features']
        for i, f in enumerate(tqdm(features_iter, total=len(video_ids), desc="비디오 임베딩 추출")):
            feats[i, :, :] = f

# -----------------------------------------------------------------------------
# 메인 파이프라인
# -----------------------------------------------------------------------------
def create_gmmformer_structure(
    output_dir: str,
    video_info: Dict,
    captions_data: List[Dict],
    video_dir: str,
    model: CLIPModel,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    max_frames: int,
    text_batch: int,
    amp: str,
    overwrite: bool = False,
):
    out = Path(output_dir)
    text_dir = out / 'TextData'
    feat_dir = out / 'FeatureData'
    clip_dir = feat_dir / 'clip'
    text_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1) 스플릿별 caption 파일 생성
    # -----------------------------
    splits = {'train': [], 'val': [], 'test': []}
    for item in captions_data:
        sp = item['split']
        if sp not in splits:
            # unknown → train으로 보냄
            sp = 'train'
        splits[sp].append(item)

    for sp, items in splits.items():
        if not items:
            continue
        cap_path = text_dir / f'msrvtt_{sp}.caption.txt'
        if cap_path.exists() and not overwrite:
            print(f"[SKIP] {cap_path} 존재 (overwrite=False)")
        else:
            with open(cap_path, 'w', encoding='utf-8') as f:
                for it in items:
                    f.write(f"{it['key']} {it['caption']}\n")
            print(f"[OK] {cap_path} 생성")

    # -----------------------------
    # 2) 텍스트 임베딩: 전체 캡션 한 번에
    # -----------------------------
    text_h5 = text_dir / 'clip_ViT_B_32_msrvtt_query_feat.hdf5'
    keys_all = [it['key'] for it in captions_data]
    caps_all = [it['caption'] for it in captions_data]

    if text_h5.exists() and not overwrite:
        print(f"[SKIP] {text_h5} 존재 (overwrite=False)")
    else:
        print("[INFO] 텍스트 임베딩 추출 시작...")
        text_feats = extract_clip_text_features_batched(
            caps_all, model, tokenizer, device, batch_size=text_batch, amp=amp
        )
        assert text_feats.shape[0] == len(keys_all)
        write_text_hdf5(text_h5, keys_all, text_feats)
        print(f"[OK] 텍스트 HDF5 저장 완료 -> {text_h5}")

    # -----------------------------
    # 3) 비디오 임베딩: 전체 비디오 한 번에
    # -----------------------------
    video_ids = sorted(list(video_info.keys()))
    vid_h5 = feat_dir / 'new_clip_vit_32_msrvtt_vid_features.hdf5'

    if vid_h5.exists() and not overwrite:
        print(f"[SKIP] {vid_h5} 존재 (overwrite=False)")
    else:
        print("[INFO] 비디오 임베딩 추출 시작...")
        write_video_hdf5_rowwise(vid_h5, video_ids, max_frames=max_frames, D=512)

        def _iter_video_features():
            for vid in video_ids:
                path = os.path.join(video_dir, f"{vid}.mp4")
                yield extract_clip_features_from_video(
                    path, model=model, device=device, max_frames=max_frames, amp=amp
                )

        fill_video_hdf5(vid_h5, video_ids, _iter_video_features())
        print(f"[OK] 비디오 HDF5 저장 완료 -> {vid_h5}")

    # -----------------------------
    # 4) video2frames.txt (중복 없이 한 번)
    # -----------------------------
    v2f_path = clip_dir / 'video2frames.txt'
    if v2f_path.exists() and not overwrite:
        print(f"[SKIP] {v2f_path} 존재 (overwrite=False)")
    else:
        with open(v2f_path, 'w') as f:
            for vid in video_ids:
                # GMMFormer 포맷: "video_id frame_ids"
                # 여기서는 비디오 전체를 한 단위로 취급하므로 vid 그대로 사용
                f.write(f"{vid} {vid}\n")
        print(f"[OK] {v2f_path} 생성")

    print("\n[완료] 전처리 전체 파이프라인 종료.")

# -----------------------------------------------------------------------------
# 엔트리포인트
# -----------------------------------------------------------------------------
def main():
    set_torch_opts()
    parser = argparse.ArgumentParser(description="MSRVTT → GMMFormer 전처리 (고속 배치/HDF5 단일 Dataset)")
    parser.add_argument('--msrvtt_data', type=str, default='/disk/gjw/msr-vtt/MSRVTT_data.json')
    parser.add_argument('--csv_path', type=str, default='/disk/gjw/msr-vtt/MSRVTT_JSFUSION_train_test_10k.csv')
    parser.add_argument('--video_dir', type=str, default='/disk/gjw/msr-vtt/MSRVTT_Videos')
    parser.add_argument('--output_dir', type=str, default='/disk/gjw/msrvtt')
    parser.add_argument('--max_frames', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda', help='"cuda" or "cpu"')
    parser.add_argument('--text_batch', type=int, default=1024, help='텍스트 임베딩 배치 크기')
    parser.add_argument('--amp', type=str, default='fp16', choices=['none', 'fp16', 'bf16'], help='혼합정밀')
    parser.add_argument('--overwrite', action='store_true', help='기존 산출물 덮어쓰기')
    args = parser.parse_args()

    # 디바이스 설정
    device = torch.device(args.device if (args.device.startswith('cuda') and torch.cuda.is_available()) else 'cpu')
    print(f"[INFO] device={device}, amp={args.amp}")

    # 모델/토크나이저 로드 (1회)
    print("[INFO] CLIP 모델 로드 중...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # 텍스트/비디오 공용: AMP는 autocast에서만 사용 (가중치 자체는 fp32 유지)
    model.eval()

    # 주 데이터 로드
    print("[INFO] MSRVTT 메타/캡션 로드 중...")
    video_info, captions_data = load_msrvtt_annotations(args.msrvtt_data, args.csv_path)
    print(f"[INFO] 비디오 수={len(video_info):,}, 캡션 수={len(captions_data):,}")

    # 파이프라인 실행
    create_gmmformer_structure(
        output_dir=args.output_dir,
        video_info=video_info,
        captions_data=captions_data,
        video_dir=args.video_dir,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_frames=args.max_frames,
        text_batch=args.text_batch,
        amp=args.amp,
        overwrite=args.overwrite,
    )

if __name__ == '__main__':
    main()
