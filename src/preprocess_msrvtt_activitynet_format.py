#!/usr/bin/env python3
"""
MSRVTT 데이터를 ActivityNet 형식으로 전처리하는 스크립트
- 비디오: 1.5 FPS로 샘플링하여 CLIP ViT-B/32로 인코딩
- 텍스트: CLIP ViT-B/32 text encoder로 인코딩
- ActivityNet과 동일한 HDF5 구조 및 caption 파일 형식 생성
"""

import os
import json
import h5py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas as pd

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def setup_clip_model(device='cuda'):
    """CLIP 모델 초기화"""
    model_name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
    else:
        device = 'cpu'
        model = model.to(device)
    
    model.eval()
    return model, processor, device


def extract_video_frames_at_fps(video_path: str, target_fps: float = 1.5) -> List[np.ndarray]:
    """비디오에서 지정된 FPS로 프레임 추출 (OpenCV 사용)"""
    if not os.path.exists(video_path):
        return []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return []
        
        # target_fps에 맞게 프레임 인덱스 계산
        frame_interval = fps / target_fps
        frame_indices = []
        
        current_idx = 0
        while current_idx < total_frames:
            frame_indices.append(int(current_idx))
            current_idx += frame_interval
        
        # 프레임 추출
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
        
    except Exception as e:
        print(f"비디오 로딩 오류 {video_path}: {e}")
        
    return []


def encode_frames_with_clip(frames: List[np.ndarray], model, processor, device) -> np.ndarray:
    """프레임들을 CLIP으로 인코딩"""
    if len(frames) == 0:
        return np.zeros((1, 512), dtype=np.float32)
    
    # PIL Images로 변환
    pil_images = []
    for frame in frames:
        # BGR to RGB 변환 (OpenCV 형식)
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame.astype('uint8'))
        pil_images.append(pil_image)
    
    # CLIP으로 인코딩 (배치 처리)
    batch_size = 8
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(pil_images), batch_size):
            batch_images = pil_images[i:i+batch_size]
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # L2 정규화
            all_features.append(image_features.cpu().numpy())
    
    # 모든 features 결합
    if all_features:
        features = np.concatenate(all_features, axis=0)
        return features.astype(np.float32)
    else:
        return np.zeros((1, 512), dtype=np.float32)


def encode_text_with_clip(texts: List[str], model, processor, device) -> Dict[str, np.ndarray]:
    """텍스트들을 CLIP으로 인코딩"""
    text_features = {}
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = processor(text=batch_texts, padding=True, truncation=True, 
                             return_tensors="pt", max_length=77).to(device)
            features = model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)  # L2 정규화
            
            # 각 텍스트별로 저장
            for j, text in enumerate(batch_texts):
                text_features[text] = features[j].cpu().numpy().astype(np.float32)
    
    return text_features


def load_msrvtt_data(json_path: str) -> Tuple[Dict, List]:
    """MSRVTT JSON 데이터 로드"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 비디오 정보를 딕셔너리로 변환
    videos = {video['video_id']: video for video in data['videos']}
    sentences = data['sentences']
    
    return videos, sentences


def create_train_test_splits(videos: Dict, sentences: List, csv_path: str) -> Tuple[List, List]:
    """Train/Test split 생성 (MSRVTT_train.9k.csv 기준)"""
    import pandas as pd

    # CSV에서 train video_id 로드
    train_csv = pd.read_csv(csv_path)
    train_video_ids = set(train_csv['video_id'].astype(str).tolist())

    train_data = []
    test_data = []
    
    # 비디오별 캡션 모으기
    video_captions = {}
    for sent in sentences:
        video_id = sent['video_id']
        if video_id not in video_captions:
            video_captions[video_id] = []
        video_captions[video_id].append(sent['caption'])
    
    # CSV 기준으로 train/test 나누기
    for video_id, captions in video_captions.items():
        if video_id in train_video_ids:
            for i, caption in enumerate(captions):
                train_data.append({
                    'video_id': video_id,
                    'caption_id': f"{video_id}#enc#{i}",
                    'caption': caption
                })
        else:
            # test set은 첫 번째 caption만 사용
            if captions:
                test_data.append({
                    'video_id': video_id,
                    'caption_id': f"{video_id}#enc#0",
                    'caption': captions[0]
                })
    
    return train_data, test_data


def process_msrvtt_videos(video_dir: str, videos: Dict, model, processor, device, 
                         output_path: str) -> Dict[str, int]:
    """MSRVTT 비디오들을 처리하여 HDF5로 저장"""
    
    video_frame_counts = {}
    
    with h5py.File(output_path, 'w') as f:
        for video_id, video_info in tqdm(videos.items(), desc="비디오 처리"):
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            
            # 프레임 추출 (1.5 FPS)
            frames = extract_video_frames_at_fps(video_path, target_fps=1.5)
            
            if frames:
                # CLIP으로 인코딩
                features = encode_frames_with_clip(frames, model, processor, device)
                video_frame_counts[video_id] = len(features)
            else:
                # 비디오를 찾을 수 없는 경우 기본값
                print(f"⚠️  비디오 없음: {video_path}")
                features = np.zeros((1, 512), dtype=np.float32)
                video_frame_counts[video_id] = 1
            
            # HDF5에 저장 (ActivityNet 형식: video_id -> (N, 512))
            f[video_id] = features
    
    return video_frame_counts


def process_msrvtt_texts(train_data: List, test_data: List, model, processor, device,
                        output_path: str) -> None:
    """MSRVTT 텍스트들을 처리하여 HDF5로 저장"""
    
    # 모든 고유 캡션 수집
    all_captions = set()
    caption_id_to_text = {}
    
    for item in train_data + test_data:
        caption_id = item['caption_id']
        caption = item['caption']
        all_captions.add(caption)
        caption_id_to_text[caption_id] = caption
    
    print(f"고유 캡션 {len(all_captions)}개 인코딩 중...")
    
    # CLIP으로 텍스트 인코딩
    text_features = encode_text_with_clip(list(all_captions), model, processor, device)
    
    # HDF5에 저장 (ActivityNet 형식: caption_id -> (512,))
    with h5py.File(output_path, 'w') as f:
        for caption_id, caption_text in tqdm(caption_id_to_text.items(), desc="텍스트 저장"):
            if caption_text in text_features:
                f[caption_id] = text_features[caption_text]
            else:
                # 인코딩 실패한 경우 기본값
                f[caption_id] = np.zeros(512, dtype=np.float32)


def create_caption_files(train_data: List, test_data: List, output_dir: str) -> None:
    """ActivityNet 형식의 caption 파일들 생성"""
    
    output_path = Path(output_dir)
    text_data_dir = output_path / 'TextData'
    text_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Train caption 파일
    train_file = text_data_dir / 'msrvtttrain.caption.txt'
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(f"{item['caption_id']} {item['caption']}\n")
    
    # Test caption 파일
    test_file = text_data_dir / 'msrvtttest.caption.txt'
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(f"{item['caption_id']} {item['caption']}\n")
    
    # Val caption 파일 (test와 동일)
    val_file = text_data_dir / 'msrvttval.caption.txt'
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(f"{item['caption_id']} {item['caption']}\n")
    
    print(f"✅ Caption 파일들 생성 완료:")
    print(f"  - Train: {len(train_data)}개 ({train_file})")
    print(f"  - Test: {len(test_data)}개 ({test_file})")
    print(f"  - Val: {len(test_data)}개 ({val_file})")


def main():
    # 설정
    msrvtt_video_dir = '../../msr-vtt/MSRVTT_Videos'
    msrvtt_json_path = '../../msr-vtt/MSRVTT_data.json'
    output_dir = '../../msrvtt_activitynet_format'
    csv_path = '../../msr-vtt/MSRVTT_train.9k.csv'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 MSRVTT -> ActivityNet 형식 전처리 시작 (Device: {device})")
    
    # 출력 디렉터리 생성
    output_path = Path(output_dir)
    feature_data_dir = output_path / 'FeatureData'
    text_data_dir = output_path / 'TextData'
    feature_data_dir.mkdir(parents=True, exist_ok=True)
    text_data_dir.mkdir(parents=True, exist_ok=True)
    
    # CLIP 모델 초기화
    print("📥 CLIP 모델 로딩...")
    model, processor, device = setup_clip_model(device)
    
    # MSRVTT 데이터 로드
    print("📂 MSRVTT 데이터 로딩...")
    videos, sentences = load_msrvtt_data(msrvtt_json_path)
    print(f"비디오: {len(videos)}개, 캡션: {len(sentences)}개")
    
    # Train/Test split 생성
    print("🔄 Train/Test split 생성...")
    train_data, test_data = create_train_test_splits(videos, sentences, csv_path)
    print(f"Train: {len(train_data)}개, Test: {len(test_data)}개")
    
    # 비디오 처리
    print("🎥 비디오 처리 중... (1.5 FPS 샘플링)")
    video_hdf5_path = feature_data_dir / 'new_clip_vit_32_msrvtt_vid_features.hdf5'
    video_frame_counts = process_msrvtt_videos(
        msrvtt_video_dir, videos, model, processor, device, str(video_hdf5_path)
    )
    
    # 텍스트 처리
    print("💬 텍스트 처리 중...")
    text_hdf5_path = text_data_dir / 'clip_ViT_B_32_msrvtt_query_feat.hdf5'
    process_msrvtt_texts(
        train_data, test_data, model, processor, device, str(text_hdf5_path)
    )
    
    # Caption 파일 생성
    print("📝 Caption 파일 생성 중...")
    create_caption_files(train_data, test_data, output_dir)
    
    # 통계 출력
    total_frames = sum(video_frame_counts.values())
    avg_frames = total_frames / len(video_frame_counts) if video_frame_counts else 0
    
    print(f"\n✅ 전처리 완료!")
    print(f"📁 출력 디렉터리: {output_dir}")
    print(f"📊 통계:")
    print(f"  - 총 비디오: {len(videos)}개")
    print(f"  - 총 프레임: {total_frames}개 (평균 {avg_frames:.1f}프레임/비디오)")
    print(f"  - Train 캡션: {len(train_data)}개")
    print(f"  - Test 캡션: {len(test_data)}개")
    print(f"🎯 ActivityNet과 동일한 형식으로 생성됨!")


if __name__ == '__main__':
    main()