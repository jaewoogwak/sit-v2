#!/usr/bin/env python3
"""
최적화된 MSRVTT 데이터셋 전처리 스크립트
- 중복 제거 최적화
- 배치 처리
- 캐시 시스템
"""

import os
import json
import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import argparse
from hashlib import sha256

# CLIP 관련 임포트 (Transformers 라이브러리 사용)
from transformers import CLIPModel, CLIPTokenizer
from decord import VideoReader, cpu
from torchvision.transforms import InterpolationMode, Resize, CenterCrop


def load_msrvtt_annotations(msrvtt_data_path: str, csv_path: str) -> Tuple[Dict, List]:
    """MSRVTT JSON과 CSV 데이터를 로드"""
    
    # JSON 데이터 로드 (비디오 메타데이터)
    with open(msrvtt_data_path, 'r') as f:
        msrvtt_data = json.load(f)
    
    # CSV 데이터 로드 (train/test 캡션)
    df = pd.read_csv(csv_path)
    
    # 비디오 정보를 딕셔너리로 변환
    video_info = {}
    for video in msrvtt_data['videos']:
        video_id = video['video_id']
        video_info[video_id] = {
            'split': video['split'],
            'category': video['category'],
            'url': video['url']
        }
    
    # 캡션 데이터 구조화
    captions_data = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        sentence = row['sentence']
        key = row['key']
        
        captions_data.append({
            'video_id': video_id,
            'caption': sentence,
            'key': key,
            'split': video_info.get(video_id, {}).get('split', 'train')
        })
    
    return video_info, captions_data


def _caption_id(caption: str) -> str:
    """캡션에서 고유 ID 생성"""
    return sha256(caption.encode("utf-8")).hexdigest()[:16]


def extract_video_features_batch(video_ids: List[str], video_dir: str, 
                                model, device, max_frames: int = 12) -> Dict[str, np.ndarray]:
    """비디오 피쳐를 배치로 추출 (중복 제거)"""
    
    # 이미 처리된 비디오 제외
    unique_video_ids = list(set(video_ids))
    video_features = {}
    
    # 이미지 전처리 설정
    img_resize = Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True)
    img_crop = CenterCrop(224)
    
    print(f"고유 비디오 {len(unique_video_ids)}개 처리 중...")
    
    for video_id in tqdm(unique_video_ids, desc="비디오 피쳐 추출"):
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        
        if not os.path.exists(video_path):
            print(f"경고: {video_path} 파일이 존재하지 않습니다.")
            video_features[video_id] = np.zeros((max_frames, 512), dtype=np.float32)
            continue
        
        try:
            # 비디오 리더로 프레임 추출
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            if total_frames == 0:
                video_features[video_id] = np.zeros((max_frames, 512), dtype=np.float32)
                continue
            
            # 균등하게 프레임 샘플링
            if total_frames <= max_frames:
                indices = list(range(total_frames))
            else:
                indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            
            # 프레임 추출 및 전처리
            frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
            
            processed_frames = []
            for frame in frames:
                # numpy array를 Tensor로 변환 후 전처리
                frame_tensor = torch.from_numpy(frame).float() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1)  # (C, H, W)
                frame_tensor = img_resize(frame_tensor)
                frame_tensor = img_crop(frame_tensor)
                processed_frames.append(frame_tensor)
            
            if processed_frames:
                frames_batch = torch.stack(processed_frames).to(device)  # (T, C, H, W)
                with torch.no_grad():
                    features = model.get_image_features(frames_batch)
                    features = features.cpu().numpy().astype(np.float32)
            else:
                features = np.zeros((max_frames, 512), dtype=np.float32)
            
            # max_frames에 맞게 패딩 또는 자르기
            if len(features) < max_frames:
                padding = np.zeros((max_frames - len(features), 512), dtype=np.float32)
                features = np.vstack([features, padding])
            elif len(features) > max_frames:
                features = features[:max_frames]
            
            video_features[video_id] = features
            
        except Exception as e:
            print(f"비디오 처리 오류 {video_path}: {e}")
            video_features[video_id] = np.zeros((max_frames, 512), dtype=np.float32)
    
    return video_features


def extract_text_features_batch(captions: List[str], model, tokenizer, 
                               device, batch_size: int = 32) -> Dict[str, np.ndarray]:
    """텍스트 피쳐를 배치로 추출 (중복 제거)"""
    
    # 고유 캡션만 추출
    unique_captions = list(set(captions))
    caption_to_id = {cap: _caption_id(cap) for cap in unique_captions}
    text_features = {}
    
    print(f"고유 캡션 {len(unique_captions)}개 처리 중...")
    
    # 배치 처리
    for i in tqdm(range(0, len(unique_captions), batch_size), desc="텍스트 피쳐 추출"):
        batch_captions = unique_captions[i:i+batch_size]
        
        try:
            # 배치 토큰화
            tokens = tokenizer(batch_captions, return_tensors="pt", 
                             padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                batch_features = model.get_text_features(**tokens)
                batch_features = batch_features.cpu().numpy().astype(np.float32)
            
            # 각 캡션별로 저장
            for j, caption in enumerate(batch_captions):
                cap_id = caption_to_id[caption]
                text_features[cap_id] = batch_features[j]
                
        except Exception as e:
            print(f"텍스트 배치 처리 오류: {e}")
            # 기본 피쳐로 대체
            for caption in batch_captions:
                cap_id = caption_to_id[caption]
                text_features[cap_id] = np.zeros(512, dtype=np.float32)
    
    return text_features, caption_to_id


def create_gmmformer_structure_fast(output_dir: str, video_info: Dict, captions_data: List,
                                  video_dir: str, model, tokenizer, device):
    """최적화된 GMMFormer 호환 데이터 구조 생성"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 디렉터리 구조 생성
    text_data_dir = output_path / 'TextData'
    feature_data_dir = output_path / 'FeatureData'
    text_data_dir.mkdir(exist_ok=True)
    feature_data_dir.mkdir(exist_ok=True)
    
    # 전체 데이터에서 고유 비디오와 캡션 추출
    all_video_ids = [item['video_id'] for item in captions_data]
    all_captions = [item['caption'] for item in captions_data]
    
    print(f"총 {len(captions_data)}개 캡션-비디오 쌍")
    print(f"고유 비디오: {len(set(all_video_ids))}개")
    print(f"고유 캡션: {len(set(all_captions))}개")
    
    # 1. 모든 비디오 피쳐 한번에 추출
    print("\n=== 비디오 피쳐 추출 ===")
    video_features = extract_video_features_batch(all_video_ids, video_dir, model, device)
    
    # 2. 모든 텍스트 피쳐 한번에 추출  
    print("\n=== 텍스트 피쳐 추출 ===")
    text_features, caption_to_id = extract_text_features_batch(all_captions, model, tokenizer, device)
    
    # 3. Split별로 데이터 분리 및 저장
    splits = {'train': [], 'val': [], 'test': []}
    
    for item in captions_data:
        split = item['split']
        if split in splits:
            splits[split].append(item)
        else:
            # unknown은 train으로 할당
            splits['train'].append(item)
    
    # 각 split에 대해 파일 생성
    for split_name, split_data in splits.items():
        if not split_data:
            continue
            
        print(f"\n=== {split_name} 데이터 저장 ({len(split_data)}개) ===")
        
        # 캡션 파일 생성 (GMMFormer 형식)
        caption_file = text_data_dir / f'msrvtt_{split_name}.caption.txt'
        
        with open(caption_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                key = item['key']
                caption = item['caption']
                # GMMFormer 형식: key caption
                f.write(f"{key} {caption}\n")
    
    # 4. HDF5 파일에 모든 피쳐 저장 (split 구분 없이)
    print("\n=== HDF5 파일 저장 ===")
    
    # 텍스트 피쳐 저장
    text_feat_file = text_data_dir / 'clip_ViT_B_32_msrvtt_query_feat.hdf5'
    
    with h5py.File(text_feat_file, 'w') as hf:
        for item in tqdm(captions_data, desc="텍스트 피쳐 저장"):
            key = item['key']
            caption = item['caption']
            cap_id = _caption_id(caption)
            if cap_id in text_features:
                hf[key] = text_features[cap_id]
            else:
                hf[key] = np.zeros(512, dtype=np.float32)
    
    # 비디오 피쳐 저장
    video_feat_file = feature_data_dir / 'new_clip_vit_32_msrvtt_vid_features.hdf5'
    
    with h5py.File(video_feat_file, 'w') as hf:
        for video_id, features in tqdm(video_features.items(), desc="비디오 피쳐 저장"):
            hf[video_id] = features
    
    # video2frames.txt 생성
    video2frames_dir = feature_data_dir / 'clip'
    video2frames_dir.mkdir(exist_ok=True)
    video2frames_file = video2frames_dir / 'video2frames.txt'
    
    with open(video2frames_file, 'w') as f:
        for video_id in video_features.keys():
            # video2frames 형식: video_id frame_ids
            f.write(f"{video_id} {video_id}\n")
    
    print(f"\n✅ 전처리 완료!")
    print(f"📁 결과 저장 위치: {output_dir}")
    print(f"📊 통계:")
    print(f"  - 비디오 피쳐: {len(video_features)}개")
    print(f"  - 텍스트 피쳐: {len(text_features)}개")
    print(f"  - 캡션 파일: {len(splits)}개 split")


def main():
    parser = argparse.ArgumentParser(description="최적화된 MSRVTT 데이터셋 전처리")
    
    parser.add_argument('--msrvtt_data', type=str, 
                       default='/disk/gjw/msr-vtt/MSRVTT_data.json',
                       help='MSRVTT JSON 데이터 파일 경로')
    parser.add_argument('--csv_path', type=str,
                       default='/disk/gjw/msr-vtt/MSRVTT_JSFUSION_train_test_10k.csv',
                       help='MSRVTT CSV 캡션 파일 경로')
    parser.add_argument('--video_dir', type=str,
                       default='/disk/gjw/msr-vtt/MSRVTT_Videos',
                       help='MSRVTT 비디오 디렉터리 경로')
    parser.add_argument('--output_dir', type=str,
                       default='/disk/gjw/msrvtt',
                       help='출력 디렉터리 경로')
    parser.add_argument('--max_frames', type=int, default=12,
                       help='비디오당 최대 프레임 수')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='텍스트 처리 배치 크기')
    parser.add_argument('--device', type=str, default='cuda',
                       help='CLIP 모델 실행 디바이스')
    
    args = parser.parse_args()
    
    # CLIP 모델 로드 (Transformers 라이브러리 사용)
    print("🚀 CLIP 모델 로드 중...")
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # 데이터 로드
    print("📂 MSRVTT 데이터 로드 중...")
    video_info, captions_data = load_msrvtt_annotations(args.msrvtt_data, args.csv_path)
    
    print(f"📈 데이터셋 통계:")
    print(f"  - 비디오 수: {len(video_info)}")
    print(f"  - 캡션 수: {len(captions_data)}")
    
    # GMMFormer 구조로 변환 (최적화된 방식)
    create_gmmformer_structure_fast(
        args.output_dir, video_info, captions_data, args.video_dir,
        model, tokenizer, device
    )


if __name__ == '__main__':
    main()