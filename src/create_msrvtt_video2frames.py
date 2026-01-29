#!/usr/bin/env python3
"""
MSRVTT video2frames.txt 파일을 HDF5 구조에 맞게 생성
"""

import h5py

def create_video2frames():
    """HDF5 파일의 키를 기반으로 video2frames.txt 생성"""
    
    hdf5_file = '/disk/gjw/msrvtt/FeatureData/new_clip_vit_32_msrvtt_vid_features.hdf5'
    output_file = '/disk/gjw/msrvtt/FeatureData/clip/video2frames.txt'
    
    video2frames = {}
    
    with h5py.File(hdf5_file, 'r') as f:
        for video_id in f.keys():
            # MSRVTT는 각 비디오가 하나의 키로 저장되므로 자기 자신을 프레임으로 설정
            video2frames[video_id] = [video_id]
    
    print(f"총 {len(video2frames)}개 비디오 처리됨")
    
    # video2frames.txt 파일 작성
    with open(output_file, 'w') as f:
        f.write(str(video2frames))
    
    print(f"✅ video2frames.txt 생성 완료: {output_file}")
    print(f"예시: {list(video2frames.items())[:3]}")

if __name__ == '__main__':
    create_video2frames()