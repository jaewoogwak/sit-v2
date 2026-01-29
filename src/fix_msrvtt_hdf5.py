#!/usr/bin/env python3
"""
MSRVTT HDF5 파일을 TVR/ActivityNet과 같은 구조로 변경하는 스크립트
"""

import h5py
import numpy as np

def fix_msrvtt_hdf5():
    """MSRVTT HDF5 파일을 올바른 구조로 변경"""
    
    input_file = '/disk/gjw/msrvtt/FeatureData/new_clip_vit_32_msrvtt_vid_features.hdf5'
    output_file = '/disk/gjw/msrvtt/FeatureData/new_clip_vit_32_msrvtt_vid_features_fixed.hdf5'
    
    print("MSRVTT HDF5 파일 구조 수정 중...")
    
    # 기존 파일 읽기
    with h5py.File(input_file, 'r') as f_in:
        features = f_in['features'][:]  # (10000, 12, 512)
        video_ids = f_in['video_ids'][:]  # (10000,)
        
        print(f"원본 데이터: {features.shape}, 비디오 수: {len(video_ids)}")
        
        # 새 파일에 올바른 구조로 저장
        with h5py.File(output_file, 'w') as f_out:
            for i, video_id in enumerate(video_ids):
                # bytes를 string으로 변환
                if isinstance(video_id, bytes):
                    video_id = video_id.decode('utf-8')
                
                # 각 비디오를 개별 키로 저장 (TVR/ActivityNet 형식)
                f_out[video_id] = features[i]  # (12, 512)
                
                if i % 1000 == 0:
                    print(f"진행률: {i+1}/{len(video_ids)}")
    
    print(f"✅ 수정 완료: {output_file}")
    
    # 원본 파일 백업하고 새 파일로 교체
    import os
    backup_file = input_file + '.backup'
    os.rename(input_file, backup_file)
    os.rename(output_file, input_file)
    
    print(f"✅ 원본 파일 백업: {backup_file}")
    print(f"✅ 새 파일로 교체 완료: {input_file}")

if __name__ == '__main__':
    fix_msrvtt_hdf5()