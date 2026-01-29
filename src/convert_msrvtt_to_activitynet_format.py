#!/usr/bin/env python3
"""
기존 MSRVTT 피쳐 데이터를 ActivityNet 형식으로 변환하는 스크립트
- 기존 HDF5 비디오 피쳐와 텍스트 피쳐를 ActivityNet 형식으로 재구성
- Caption 파일들을 ActivityNet과 동일한 형식으로 생성
"""

import os
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import shutil


def load_msrvtt_data(json_path: str) -> Tuple[Dict, List]:
    """MSRVTT JSON 데이터 로드"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 비디오 정보를 딕셔너리로 변환
    videos = {video['video_id']: video for video in data['videos']}
    sentences = data['sentences']
    
    return videos, sentences


def create_train_test_splits(videos: Dict, sentences: List) -> Tuple[List, List]:
    """Train/Test split 생성 (MSRVTT 표준 split: 처음 9000개 train, 나머지 test)"""
    train_data = []
    test_data = []
    
    # 비디오별로 캡션들 그룹화
    video_captions = {}
    for sent in sentences:
        video_id = sent['video_id']
        if video_id not in video_captions:
            video_captions[video_id] = []
        video_captions[video_id].append(sent['caption'])
    
    # MSRVTT 표준 split: video0-8999는 train, video9000-9999는 test/val
    for video_id in sorted(videos.keys()):  # video0, video1, ..., video9999 순서로
        if video_id in video_captions:
            captions = video_captions[video_id]
            
            # video ID에서 숫자 추출
            video_num = int(video_id.replace('video', ''))
            
            if video_num < 6513:  # 0-6512: train set (기존 MSRVTT train split)
                for i, caption in enumerate(captions):
                    train_data.append({
                        'video_id': video_id,
                        'caption_id': f"{video_id}#enc#{i}",
                        'caption': caption
                    })
            else:  # 6513-9999: test set
                # Test 데이터는 모든 캡션 사용 (ActivityNet과는 달리)
                for i, caption in enumerate(captions):
                    test_data.append({
                        'video_id': video_id,
                        'caption_id': f"{video_id}#enc#{i}",
                        'caption': caption
                    })
    
    return train_data, test_data


def convert_video_features(input_hdf5: str, output_hdf5: str, video_ids: List[str]) -> None:
    """비디오 피쳐를 ActivityNet 형식으로 변환"""
    print(f"📥 비디오 피쳐 변환 중: {input_hdf5} -> {output_hdf5}")
    
    # 기존 MSRVTT HDF5 파일 구조 확인
    with h5py.File(input_hdf5, 'r') as f_in:
        print(f"입력 HDF5 키들: {list(f_in.keys())}")
        
        # 구조 확인
        if 'features' in f_in and 'video_ids' in f_in:
            # 배열 형식: {features: (N, frames, 512), video_ids: (N,)}
            features = f_in['features'][...]  # (N, frames, 512)
            video_id_array = f_in['video_ids'][...]  # (N,)
            
            print(f"Features shape: {features.shape}")
            print(f"Video IDs shape: {video_id_array.shape}")
            
            # ActivityNet 형식으로 변환: {video_id: (frames, 512)}
            with h5py.File(output_hdf5, 'w') as f_out:
                for i, video_id in enumerate(tqdm(video_id_array, desc="비디오 피쳐 변환")):
                    if isinstance(video_id, bytes):
                        video_id = video_id.decode('utf-8')
                    
                    # 각 비디오의 피쳐 추출 (frames, 512)
                    video_feature = features[i]  # (frames, 512)
                    
                    # ActivityNet 형식으로 저장
                    f_out[video_id] = video_feature.astype(np.float32)
        
        elif len(list(f_in.keys())) > 100:  # 이미 video_id 키 형식인 경우
            # 그대로 복사
            print("이미 ActivityNet 형식입니다. 복사 중...")
            shutil.copy2(input_hdf5, output_hdf5)
        
        else:
            print(f"⚠️  알 수 없는 HDF5 구조: {list(f_in.keys())}")


def create_text_features_hdf5(train_data: List, test_data: List, 
                             input_text_hdf5: str, output_hdf5: str) -> None:
    """텍스트 피쳐를 ActivityNet 형식으로 변환"""
    print(f"💬 텍스트 피쳐 변환 중: {input_text_hdf5} -> {output_hdf5}")
    
    # 모든 캡션 ID와 텍스트 매핑
    caption_id_to_text = {}
    for item in train_data + test_data:
        caption_id_to_text[item['caption_id']] = item['caption']
    
    # 기존 텍스트 피쳐 로드 (캐시된 피쳐 사용)
    if os.path.exists(input_text_hdf5):
        print("기존 텍스트 피쳐 파일을 사용합니다.")
        shutil.copy2(input_text_hdf5, output_hdf5)
    else:
        print("⚠️  텍스트 피쳐 파일이 없습니다. 빈 피쳐로 생성합니다.")
        with h5py.File(output_hdf5, 'w') as f:
            for caption_id in tqdm(caption_id_to_text.keys(), desc="빈 텍스트 피쳐 생성"):
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


def create_video2frames_file(output_dir: str, video_hdf5_path: str) -> None:
    """video2frames.txt 파일 생성"""
    
    video2frames = {}
    
    with h5py.File(video_hdf5_path, 'r') as f:
        for video_id in f.keys():
            # MSRVTT는 각 비디오가 하나의 키로 저장되므로 자기 자신을 프레임으로 설정
            video2frames[video_id] = [video_id]
    
    print(f"총 {len(video2frames)}개 비디오 처리됨")
    
    # FeatureData/clip 디렉터리 생성
    output_path = Path(output_dir)
    feature_clip_dir = output_path / 'FeatureData' / 'clip'
    feature_clip_dir.mkdir(parents=True, exist_ok=True)
    
    # video2frames.txt 파일 작성
    output_file = feature_clip_dir / 'video2frames.txt'
    with open(output_file, 'w') as f:
        f.write(str(video2frames))
    
    print(f"✅ video2frames.txt 생성 완료: {output_file}")
    print(f"예시: {list(video2frames.items())[:3]}")


def main():
    # 설정
    msrvtt_json_path = '/disk/gjw/msr-vtt/MSRVTT_data.json'
    input_video_hdf5 = '/disk/gjw/msrvtt/FeatureData/new_clip_vit_32_msrvtt_vid_features.hdf5'
    input_text_hdf5 = '/disk/gjw/msrvtt/TextData/clip_ViT_B_32_msrvtt_query_feat.hdf5'
    output_dir = '/disk/gjw/msrvtt_activitynet_format'
    
    print(f"🚀 MSRVTT -> ActivityNet 형식 변환 시작")
    
    # 출력 디렉터리 생성
    output_path = Path(output_dir)
    feature_data_dir = output_path / 'FeatureData'
    text_data_dir = output_path / 'TextData'
    feature_data_dir.mkdir(parents=True, exist_ok=True)
    text_data_dir.mkdir(parents=True, exist_ok=True)
    
    # MSRVTT 데이터 로드
    print("📂 MSRVTT 메타데이터 로딩...")
    videos, sentences = load_msrvtt_data(msrvtt_json_path)
    print(f"비디오: {len(videos)}개, 캡션: {len(sentences)}개")
    
    # Train/Test split 생성
    print("🔄 Train/Test split 생성...")
    train_data, test_data = create_train_test_splits(videos, sentences)
    print(f"Train: {len(train_data)}개, Test: {len(test_data)}개")
    
    # 비디오 피쳐 변환
    print("🎥 비디오 피쳐 변환 중...")
    video_hdf5_path = feature_data_dir / 'new_clip_vit_32_msrvtt_vid_features.hdf5'
    convert_video_features(input_video_hdf5, str(video_hdf5_path), list(videos.keys()))
    
    # 텍스트 피쳐 변환
    print("💬 텍스트 피쳐 변환 중...")
    text_hdf5_path = text_data_dir / 'clip_ViT_B_32_msrvtt_query_feat.hdf5'
    create_text_features_hdf5(train_data, test_data, input_text_hdf5, str(text_hdf5_path))
    
    # Caption 파일 생성
    print("📝 Caption 파일 생성 중...")
    create_caption_files(train_data, test_data, output_dir)
    
    # video2frames.txt 파일 생성
    print("🗂️  video2frames.txt 생성 중...")
    create_video2frames_file(output_dir, str(video_hdf5_path))
    
    print(f"\n✅ ActivityNet 형식 변환 완료!")
    print(f"📁 출력 디렉터리: {output_dir}")
    print(f"📊 통계:")
    print(f"  - 총 비디오: {len(videos)}개")
    print(f"  - Train 캡션: {len(train_data)}개")
    print(f"  - Test 캡션: {len(test_data)}개")
    print(f"🎯 ActivityNet과 동일한 형식으로 변환됨!")


if __name__ == '__main__':
    main()