#!/usr/bin/env python3
"""
MSRVTT caption 파일을 올바른 train/test split으로 분리하는 스크립트
ActivityNet과 같은 구조로 만들기: {dataset}train.caption.txt, {dataset}test.caption.txt
"""

import pandas as pd
import argparse
from pathlib import Path
import json


def create_msrvtt_splits(train_csv_path: str, test_csv_path: str, 
                        msrvtt_json_path: str, output_dir: str):
    """MSRVTT 데이터를 ActivityNet과 같은 구조로 분리"""
    
    output_path = Path(output_dir)
    text_data_dir = output_path / 'TextData'
    text_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Train 데이터 처리
    print("📂 Train 데이터 처리 중...")
    
    # Train CSV는 video_id 목록만 있음
    train_df = pd.read_csv(train_csv_path)
    train_video_ids = set(train_df['video_id'].astype(str))
    
    print(f"Train 비디오 수: {len(train_video_ids)}")
    
    # MSRVTT JSON에서 전체 캡션 데이터 로드
    with open(msrvtt_json_path, 'r') as f:
        msrvtt_data = json.load(f)
    
    # 각 비디오의 캡션들을 추출
    train_captions = []
    train_count = 0
    
    for video in msrvtt_data['videos']:
        video_id = video['video_id']
        
        if video_id in train_video_ids:
            # 해당 비디오의 모든 캡션 찾기
            video_captions = []
            for sentence in msrvtt_data['sentences']:
                if sentence['video_id'] == video_id:
                    video_captions.append(sentence['caption'])
            
            # ActivityNet 형식으로 변환: video_id#enc#caption_num caption
            for i, caption in enumerate(video_captions):
                key = f"{video_id}#enc#{i}"
                train_captions.append(f"{key} {caption}")
                train_count += 1
    
    # Train caption 파일 저장
    train_caption_file = text_data_dir / 'msrvtttrain.caption.txt'
    with open(train_caption_file, 'w', encoding='utf-8') as f:
        for caption_line in train_captions:
            f.write(caption_line + '\n')
    
    print(f"✅ Train caption 파일 생성: {train_caption_file}")
    print(f"   - 캡션 수: {train_count}")
    
    # 2. Test 데이터 처리
    print("\n📂 Test 데이터 처리 중...")
    
    # Test CSV는 key, video_id, sentence가 모두 있음
    test_df = pd.read_csv(test_csv_path)
    
    test_captions = []
    test_count = 0
    for _, row in test_df.iterrows():
        video_id = str(row['video_id'])
        sentence = str(row['sentence'])
        # Train과 같은 형식으로 변경: video_id#enc#0
        key = f"{video_id}#enc#0"
        test_captions.append(f"{key} {sentence}")
        test_count += 1
    
    # Test caption 파일 저장
    test_caption_file = text_data_dir / 'msrvtttest.caption.txt'
    with open(test_caption_file, 'w', encoding='utf-8') as f:
        for caption_line in test_captions:
            f.write(caption_line + '\n')
    
    print(f"✅ Test caption 파일 생성: {test_caption_file}")
    print(f"   - 캡션 수: {test_count}")
    
    # 3. Validation 데이터 처리 (Test와 동일하게 설정)
    print("\n📂 Validation 데이터 처리 중...")
    
    val_caption_file = text_data_dir / 'msrvttval.caption.txt'  
    # Val은 Test와 동일하게 설정 (일반적인 MSRVTT 평가 방식)
    with open(val_caption_file, 'w', encoding='utf-8') as f:
        for caption_line in test_captions:
            f.write(caption_line + '\n')
    
    print(f"✅ Val caption 파일 생성: {val_caption_file}")
    print(f"   - 캡션 수: {test_count} (test와 동일)")
    
    # 4. 기존 잘못된 파일 정리
    old_train_file = text_data_dir / 'msrvtt_train.caption.txt'
    old_val_file = text_data_dir / 'msrvtt_val.caption.txt'
    old_test_file = text_data_dir / 'msrvtt_test.caption.txt'
    
    for old_file in [old_train_file, old_val_file, old_test_file]:
        if old_file.exists():
            backup_file = old_file.with_suffix('.caption.txt.backup')
            old_file.rename(backup_file)
            print(f"🔄 기존 파일 백업: {old_file} → {backup_file}")
    
    print(f"\n🎉 MSRVTT split 분리 완료!")
    print(f"📁 출력 디렉터리: {text_data_dir}")
    print(f"📊 최종 파일:")
    print(f"   - msrvtttrain.caption.txt: {train_count}개 캡션")
    print(f"   - msrvttval.caption.txt: {test_count}개 캡션")  
    print(f"   - msrvtttest.caption.txt: {test_count}개 캡션")


def main():
    parser = argparse.ArgumentParser(description="MSRVTT caption 파일을 올바른 train/test split으로 분리")
    
    parser.add_argument('--train_csv', type=str, 
                       default='/disk/gjw/msr-vtt/MSRVTT_train.9k.csv',
                       help='MSRVTT train CSV 파일 경로')
    parser.add_argument('--test_csv', type=str,
                       default='/disk/gjw/msr-vtt/MSRVTT_JSFUSION_test.csv', 
                       help='MSRVTT test CSV 파일 경로')
    parser.add_argument('--msrvtt_json', type=str,
                       default='/disk/gjw/msr-vtt/MSRVTT_data.json',
                       help='MSRVTT JSON 데이터 파일 경로')
    parser.add_argument('--output_dir', type=str,
                       default='/disk/gjw/msrvtt',
                       help='출력 디렉터리 경로')
    
    args = parser.parse_args()
    
    print("🚀 MSRVTT split 분리 시작...")
    print(f"📁 Train CSV: {args.train_csv}")
    print(f"📁 Test CSV: {args.test_csv}")
    print(f"📁 MSRVTT JSON: {args.msrvtt_json}")
    print(f"📁 출력 디렉터리: {args.output_dir}")
    
    create_msrvtt_splits(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv, 
        msrvtt_json_path=args.msrvtt_json,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()