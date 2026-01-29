**개요**
- 목적: GMMFormer로 WebVid 대규모(≈10.7M) 데이터 학습을 안정적으로 돌리기 위한 구현 요약과 가이드.
- 핵심: 대용량 HDF5 샤딩(+메타/매니페스트), 정렬형(aligned) 데이터셋, 에폭당 샘플 수 제한 또는 샤드 회전 샘플링, 적절한 실행 플래그.

**데이터 준비**
- 변환 스크립트: WebVid PT 샤드를 GMMFormer 형식의 HDF5/캡션으로 변환.
  - 파일: `src/preprocess_webvid_to_hdf5.py`
  - 출력 레이아웃(기본): `<out_root>/webvid/`
    - `FeatureData/new_clip_vit_32_webvid_vid_features.hdf5` (또는 `video_shard_***.hdf5` 다중 파일)
    - `TextData/clip_ViT_B_32_webvid_query_feat.hdf5` (또는 `text_shard_***.hdf5` 다중 파일)
    - `TextData/webvidtrain.caption.txt`, `TextData/webvidtest.caption.txt`
    - 선택: `FeatureData/video_manifest.json`, `TextData/text_manifest.json`
    - 선택: `FeatureData/video_meta.json`, `TextData/text_meta.json`
- 키 규약:
  - 비디오 키: `vid_{비디오인덱스}`
  - 텍스트 키: `vid_{비디오인덱스}#cap_{캡션인덱스}`
- 권장 샤딩 변환(대용량):
  - 샤드 단위로 HDF5 파일을 쓰고 매니페스트를 생성해 메모리 사용을 최소화.
  - 예시: `python src/preprocess_webvid_to_hdf5.py --input_dir /path/to/webvid_pt --out_root /path/to/data_root --use_shard_local_pairs --videos_per_h5 50000 --caps_per_h5 500000`
- 소규모/단일 파일 실험(옵션):
  - 예시: `python src/preprocess_webvid_to_hdf5.py --input_dir /path/to/webvid_pt --out_root /path/to/data_root --max_videos 100000 --caps_per_video 1`

**메타/매니페스트 구조**
- 매니페스트(JSON): 키 → HDF5 상대경로 매핑
  - 비디오: `FeatureData/video_manifest.json`
  - 텍스트: `TextData/text_manifest.json`
- 메타(JSON): 샤드 인덱싱 정보
  - 필드: `{"per_shard": 50000, "pattern": "video_shard_%03d.hdf5", "total": N}`
  - 파일: `FeatureData/video_meta.json`, `TextData/text_meta.json`
- 자동 메타 생성:
  - 메타가 없고 매니페스트만 있을 때, 로더가 `text_meta.auto.json`, `video_meta.auto.json`을 생성해 인덱싱을 가능하게 함.
  - 코드: `src/Datasets/builder.py:40`, `src/Datasets/builder.py:66`, `src/Datasets/builder.py:121`, `src/Datasets/builder.py:142`

**데이터 로딩 경로**
- 설정: `src/Configs/webvid.py`에서 `data_root`와 폴더 구조를 지정.
- 데이터셋 빌더: `src/Datasets/builder.py`
  - WebVid 학습용: `WebVidAlignedDataset4PRVR` 사용(대용량 캡션 텍스트를 메모리에 올리지 않음).
    - 파일: `src/Datasets/webvid_aligned_dataset.py`
    - 가정: 학습 시 1:1 매핑(캡션 인덱스 == 비디오 인덱스)
      - 비디오 ID: `vid_{i}`
      - 캡션 ID: `vid_{i}#cap_{i}`
  - 피쳐 리더:
    - 비디오: `IndexedVideoHDF5`(메타), `MultiHDF5File`(매니페스트), `HDF5File`(단일 파일)
    - 텍스트: `IndexedTextH5`(메타), `MultiTextH5`(매니페스트), 단일 HDF5 리더
    - 파일: `src/Utils/basic_utils.py`
  - 검증/테스트: `webvidtest.caption.txt` 기반의 소규모 평가 셋을 MSRVTT 로더 구성 재사용으로 처리.

**학습 에폭 구성**
- 기본 스텝 제한: `--steps_per_epoch` 기본 8000 스텝 × 배치 128 ≈ 1.024M 샘플/에폭.
  - 코드: `src/main.py:40`, `src/main.py:63`, `src/main.py:65`, `src/main.py:81`, `src/Configs/webvid.py:40`
- 샤드 에폭 샘플러: `--train_shard_size N` 지정 시 에폭마다 서로 다른 구간(shard)만 소진하고 다음 에폭에 다음 샤드로 회전.
  - 구현: `ShardEpochSampler`가 에폭마다 윈도우 이동 + 셔플(`seed+epoch`).
  - 코드: `src/Datasets/samplers.py:6`, `src/Datasets/builder.py:300`, `src/Datasets/builder.py:302`, `src/main.py:220`
- 슈퍼에폭 개념: `ceil(total / train_shard_size)` 에폭이 모여 전체 데이터 1패스.
  - 예: 총 10.7M, `train_shard_size=1,000,000` → 첫 11에폭이 1패스, 12에폭부터 재방문.
  - 중복: 같은 슈퍼에폭 내에서는 샤드 경계가 불연속이므로 중복 없음(샤드 내부는 셔플만).
- 전량 소진 vs 랜덤 샘플링:
  - 전량 소진(권장, 중복 없음): `--train_shard_size 1000000 --steps_per_epoch -1`
  - 랜덤 1M/에폭(기본, 중복 가능): `--steps_per_epoch 8000` 유지, 샤드 미사용

**실행 예시**
- 중복 없이 11에폭으로 10.7M 1회 커버(권장):
  - `python src/main.py -d webvid --train_shard_size 1000000 --steps_per_epoch -1`
- 기본(랜덤 1M/에폭, 중복 가능):
  - `python src/main.py -d webvid`
- 한 에폭에 전체 데이터 소진(메모리/시간 부담 큼):
  - `python src/main.py -d webvid --steps_per_epoch -1`
- 성능 관련 옵션:
  - `--amp`: 혼합 정밀도 사용
  - `--accum_steps K`: 그라디언트 누적 K스텝
  - `--grad_clip 1.0`: 최대 그라디언트 노름 클리핑
  - `--gpu 0`: 사용 GPU 지정

**검증/체크포인트**
- 에폭 종료마다 검증 실행, 최고 성능을 `best.ckpt`로 저장.
  - 코드: `src/main.py:131`–`src/main.py:173`
- 테스트 실행: `--eval --resume /path/to/best.ckpt`로 평가만 수행.

**주의/체크리스트**
- 메타 total 값: `FeatureData/video_meta.json`의 `total`이 실제 학습 샘플 수와 일치해야 함.
  - 불일치 시 에폭 길이, 샤드 개수 계산이 어긋남.
  - 메타가 없으면 빌더가 캡션 라인수를 세어 대체(`webvidtrain.caption.txt`).
- 키 일관성: HDF5/매니페스트의 키가 `vid_*` 및 `vid_*#cap_*` 규약을 따라야 함.
- 경로 구조: `data_root/webvid/{FeatureData,TextData}` 구조를 맞추고, `Configs/webvid.py`의 `data_root`를 환경에 맞게 변경.
- 워커/메모리: 기본 설정은 `num_workers=1`, `pin_memory=False`로 보수적. 리소스 여건에 맞게 조정 가능.

**다른 프로젝트 이식 포인트**
- 필수 컴포넌트:
  - 샤드 회전 샘플러: `src/Datasets/samplers.py:6`
  - 인덱스 기반/매니페스트 기반 HDF5 리더: `src/Utils/basic_utils.py`
  - 대용량 친화적 변환 스크립트와 키 규약: `src/preprocess_webvid_to_hdf5.py`
  - 정렬형 데이터셋(1:1 매핑) 아이디어: `src/Datasets/webvid_aligned_dataset.py`
  - 에폭 스텝 제한/해제 플로우: `src/main.py:40`, `src/main.py:63`, `src/main.py:65`, `src/main.py:81`
- 통합 절차:
  - 동일한 키 규약으로 데이터 변환 → 메타/매니페스트 생성 → HDF5 리더/샘플러 연결 → 학습 루프에서 에폭 스텝 제한 또는 샤드 회전 적용.

**빠른 디버깅 팁**
- 부분 변환: `--max_videos`, `--caps_per_video`, `--test_max_videos`로 작은 서브셋 생성.
- 길이 확인: 학습 시작 로그에 피쳐 shape와 데이터셋 길이가 출력되므로 총량 점검.
- 슈퍼에폭 스케줄: `ceil(total/shard)` 주기로 학습률 스케줄을 조정하면 전체 1패스 기준 튜닝이 쉬움.

**관련 파일**
- 변환: `src/preprocess_webvid_to_hdf5.py`
- 설정: `src/Configs/webvid.py`
- 데이터셋/샘플러: `src/Datasets/builder.py`, `src/Datasets/webvid_aligned_dataset.py`, `src/Datasets/samplers.py`
- HDF5 리더: `src/Utils/basic_utils.py`
- 메인 루프: `src/main.py`

