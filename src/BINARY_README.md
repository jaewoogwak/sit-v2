# GMMFormer + Binary Hashing Integration

이 프로젝트는 GMMFormer 모델에 binary hashing 기능을 통합하여 효율적인 비디오-텍스트 검색을 지원합니다.

## 주요 특징

- **GMMFormer 백본**: Gaussian Mixture Model 기반 Transformer 아키텍처
- **Binary Projection**: 384차원 → 3008차원 binary embedding
- **효율적 검색**: Hamming distance 기반 고속 검색
- **하이브리드 평가**: Float cosine similarity + Binary hamming distance

## 파일 구조

```
src/
├── Models/gmmformer/
│   └── hybrid_model.py          # GMMFormerBinary 하이브리드 모델
├── Datasets/
│   └── binary_dataset.py        # Binary-compatible 데이터셋
├── Validations/
│   └── binary_validation.py     # Binary evaluation 함수들
├── binary_train.py              # 통합 훈련 스크립트
└── Configs/tvr.py              # Binary 설정이 추가된 config
```

## 사용법

### 1. 환경 설정

```bash
# GMMFormer 환경 설정 (기존)
conda create -n prvr python=3.9
conda activate prvr
conda install pytorch==1.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt

# Binary index 라이브러리 (선택사항 - Hamming distance 검색용)
# pip install binary-index  # 또는 해당 라이브러리
```

### 2. 캐시 파일 생성

```bash
cd src
python binary_train.py --create_cache --config tvr
```

### 3. 훈련

#### 기본 훈련 (처음부터)
```bash
python binary_train.py --config tvr --binary_epochs 50
```

#### 사전 훈련된 GMMFormer로부터 시작
```bash
python binary_train.py --config tvr --gmmformer_checkpoint path/to/gmmformer.ckpt
```

#### 하이퍼파라미터 조정
```bash
python binary_train.py --config tvr \
  --binary_lr 1e-3 \
  --binary_batch_size 256 \
  --binary_temp 0.05 \
  --binary_loss_weight 0.7
```

### 4. 평가만 실행

```bash
python binary_train.py --eval_only --config tvr
```

## 주요 클래스 및 함수

### GMMFormerBinary
- `GMMFormer_Net`을 상속받은 하이브리드 모델
- Binary projection layer 추가
- Contrastive learning 지원

```python
from Models.gmmformer.hybrid_model import GMMFormerBinary

model = GMMFormerBinary(config)
outputs = model(batch)  # GMMFormer + Binary 출력 모두 포함
```

### BinaryTVRDataset
- GMMFormer와 binary evaluation 모두 지원
- Moment-level과 video-level 검색 지원

```python
from Datasets.binary_dataset import BinaryTVRDataset

# GMMFormer 훈련용
dataset = BinaryTVRDataset('train', cfg, binary_mode=False)

# Binary 평가용
dataset = BinaryTVRDataset('val', cfg, binary_mode=True, moment_split=True)
```

### Binary Evaluation
- Float similarity와 Binary hamming distance 평가
- Moment retrieval 지원

```python
from Validations.binary_validation import evaluate_binary_hamming, evaluate_float_similarity

# Float 평가
float_results, float_sumr = evaluate_float_similarity(model, cfg, 'val')

# Binary 평가
binary_results, binary_sumr = evaluate_binary_hamming(model, cfg, 'val')
```

## 설정 파라미터

### Binary Projection 설정 (Configs/tvr.py)

```python
# Binary projection dimensions
cfg['binary_dim'] = 3008        # Binary embedding dimension (must be divisible by 64)
cfg['binary_act'] = 'tanh'      # Activation function: 'tanh', 'relu', 'gelu', 'sigmoid'

# Training parameters
cfg['binary_temp'] = 0.07       # Temperature for contrastive loss
cfg['binary_epochs'] = 50       # Number of training epochs
cfg['binary_lr'] = 5e-3         # Learning rate for binary projection
cfg['binary_wd'] = 1e-2         # Weight decay
cfg['binary_batch_size'] = 128  # Batch size
cfg['binary_loss_weight'] = 0.5 # Weight balance: GMMFormer vs Binary loss
```

## 모델 아키텍처

```
Input: Text(768) + Video(I3D+ResNet)
    ↓
GMMFormer Encoder (Text/Video → 384차원)
    ↓
Binary Projection Layer (384 → 3008차원)
    ↓
Activation Function (tanh)
    ↓
Binary Packing (3008 → 47×64-bit integers)
    ↓
Hamming Distance Search
```

## 평가 메트릭

- **Recall@K**: K개 후보 중 정답이 포함될 확률 (K=1,5,10,100)
- **SumR**: 모든 Recall@K의 합
- **Float vs Binary**: Cosine similarity vs Hamming distance 비교
- **Moment Retrieval**: 비디오 세그먼트 단위 검색 성능

## 예상 결과

| Method | R@1 | R@5 | R@10 | R@100 | SumR |
|--------|-----|-----|------|-------|------|
| GMMFormer (Float) | XX.X% | XX.X% | XX.X% | XX.X% | XX.X |
| GMMFormer+Binary | XX.X% | XX.X% | XX.X% | XX.X% | XX.X |

## 문제 해결

### 메모리 부족
- `binary_batch_size` 줄이기
- `eval_query_bsz`, `eval_context_bsz` 줄이기

### 성능 저하
- `binary_loss_weight` 조정 (0.3-0.7 범위)
- `binary_temp` 조정 (0.05-0.1 범위)
- 사전 훈련된 GMMFormer 체크포인트 사용

### Binary index 없음
- Binary hamming distance 평가는 건너뛰고 float similarity만 사용
- `binary_index` 라이브러리 설치 또는 대체 구현

## 추가 데이터셋

ActivityNet Captions와 Charades-STA에도 동일하게 적용 가능:

```bash
# ActivityNet
python binary_train.py --config act

# Charades-STA  
python binary_train.py --config cha
```

각 데이터셋의 config 파일에도 동일한 binary 설정이 추가되어야 합니다.