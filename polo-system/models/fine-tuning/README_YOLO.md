# YOLO 논문 Easy 모델 파인튜닝

YOLO 논문 데이터를 사용하여 Easy 모델을 파인튜닝하는 프로젝트입니다.

## 📁 파일 구조

```
fine-tuning/
├── training/
│   ├── qlora_yolo.py          # YOLO 전용 파인튜닝 스크립트
│   ├── train_yolo.py          # 파라미터 분석 스크립트
│   └── yolo_v1.json           # YOLO 논문 학습 데이터
├── outputs/
│   ├── llama32-3b-qlora/      # 기존 Easy 모델 체크포인트
│   │   ├── checkpoint-3000/
│   │   ├── checkpoint-3500/
│   │   └── checkpoint-4000/
│   └── yolo-easy-qlora/       # YOLO 모델 체크포인트 (생성됨)
├── app.py                     # 기존 Easy 모델 서빙
├── app_yolo.py                # YOLO 모델 서빙
├── check_checkpoint.py        # 체크포인트 확인 및 설정
├── run_yolo.py                # YOLO 실행 스크립트
├── docker-compose.yml         # Docker Compose 설정 (3개 서비스)
├── dockerfile                 # Docker 이미지 설정
└── README_YOLO.md            # 이 파일
```

## 🚀 빠른 시작

### **방법 1: 자동 실행 스크립트 (권장)**
```bash
cd C:\POLO\POLO\polo-system\models\fine-tuning
python run_yolo.py
```

실행 옵션:
- **1. YOLO 학습만 실행**: 처음부터 학습 시작
- **2. YOLO 서빙만 실행**: 학습된 모델로 서빙 시작
- **3. 학습 후 서빙 실행**: 학습 완료 후 자동으로 서빙 시작
- **4. 자동 재개 학습**: 기존 체크포인트에서 이어서 학습 (Early Stopping 활성화)
- **5. 종료**

### **방법 2: Docker Compose 직접 실행**
```bash
# YOLO 학습만
docker-compose up yolo-train

# YOLO 서빙만  
docker-compose up yolo-llm

# 모든 서비스 (기존 Easy + YOLO)
docker-compose up
```

## 📊 Early Stopping 기능

학습 중 loss가 수렴하면 자동으로 중단됩니다:

- **Patience**: 3번의 evaluation에서 개선이 없으면 중단
- **Threshold**: 0.001 미만의 개선은 무시
- **Eval Steps**: 100 step마다 validation 실행
- **Best Model**: 가장 좋은 성능의 모델을 자동으로 저장

### Early Stopping 설정 변경
`docker-compose.yml`에서 다음 값들을 조정할 수 있습니다:
```yaml
--early_stopping_patience 3      # patience 횟수
--early_stopping_threshold 0.001  # 개선 임계값
--eval_steps 100                  # evaluation 주기
```

## 🔧 QLoRA 파라미터 수정 방식

### 1. 전체 모델 파라미터
- **Llama-3.2-3B-Instruct**: 약 3.2B 파라미터
- **전체 모델**: 모든 레이어의 가중치 행렬

### 2. LoRA (Low-Rank Adaptation)
- **타겟 모듈**: `q_proj`, `k_proj`, `v_proj`, `o_proj` (Attention 레이어)
- **LoRA Rank**: 16 (낮은 랭크 행렬 분해)
- **LoRA Alpha**: 32 (스케일링 팩터)
- **학습 파라미터**: 전체의 약 0.1% (3.2M 파라미터)

### 3. 4-bit Quantization
- **원본**: 32-bit float → **양자화**: 4-bit int
- **메모리 절약**: 75% 감소
- **성능 유지**: NF4 양자화로 품질 보존

## 📊 데이터 전처리

### YOLO JSON 구조
```json
{
  "id": "1.1.1",
  "section_major": 1,
  "section_minor": 1,
  "original": "Humans glance at an image...",
  "simplified": "Humans glance at an image..."
}
```

### 전처리 과정
1. **문단별 그룹화**: `section_major.section_minor` 단위로 문장들을 합침
2. **반복 문장 제거**: "In simple terms..." 같은 반복 패턴 제거
3. **프롬프트 변환**: Easy 모델 형식으로 변환

## 🚀 실행 방법

### 1. 파라미터 분석
```bash
cd polo-system/models/fine-tuning
python training/train_yolo.py
```

### 2. 체크포인트 확인
```bash
python check_checkpoint.py
```

### 3. Docker로 학습 실행
```bash
# 처음부터 학습
docker-compose up yolo-train

# 체크포인트에서 이어서 학습 (자동 감지)
docker-compose up yolo-train
```

### 4. 학습 모니터링
```bash
# 로그 확인
docker logs -f yolo-train

# TensorBoard (선택사항)
tensorboard --logdir outputs/yolo-easy-qlora/logs
```

## ⚙️ 학습 설정

### 하이퍼파라미터
- **Epochs**: 5
- **Batch Size**: 1 × 4 (gradient accumulation)
- **Learning Rate**: 2e-4
- **Max Length**: 1024
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.05

### 메모리 요구사항
- **GPU**: 8GB+ VRAM (RTX 3070 이상 권장)
- **RAM**: 16GB+ 시스템 메모리
- **Storage**: 10GB+ 여유 공간

## 📈 예상 결과

### 학습 전
```
원문: "Humans glance at an image and instantly know what objects are in the image, where they are, and how they interact."
```

### 학습 후 (예상)
```
간소화: "사람들은 이미지를 한 번 보면 즉시 어떤 물체가 있는지, 어디에 있는지, 어떻게 상호작용하는지 알 수 있습니다."
```

## 🔄 체크포인트 관리

### 자동 체크포인트 감지
- `check_checkpoint.py`가 자동으로 최신 체크포인트를 찾아 설정
- 체크포인트가 없으면 처음부터 학습
- 체크포인트가 있으면 해당 지점부터 이어서 학습

### 수동 체크포인트 설정
```bash
# 특정 체크포인트에서 시작
docker-compose run yolo-train --resume_from_checkpoint outputs/yolo-easy-qlora/checkpoint-2000
```

## 🐛 문제 해결

### 메모리 부족
```bash
# 배치 크기 줄이기
--per_device_train_batch_size 1
--gradient_accumulation_steps 2
```

### 학습 속도 느림
```bash
# 시퀀스 길이 줄이기
--max_seq_length 512
```

### 체크포인트 오류
```bash
# 처음부터 학습
docker-compose run yolo-train --resume_from_checkpoint null
```

## 📝 로그 예시

```
[파라미터] 전체: 3,200,000,000 | 학습가능: 3,200,000 (0.10%)
[LoRA] LoRA 파라미터: 3,200,000
[데이터] 전처리 완료: 45개 문단
[학습] 시작 - 5 epochs, 45 samples
[학습] 배치 크기: 1 × 4 = 4
[학습] 학습률: 0.0002
```

## 🎯 성공 지표

1. **Loss 감소**: 3.0 → 1.5 이하
2. **일관성**: 같은 입력에 대해 비슷한 출력
3. **품질**: 원문 의미 보존 + 쉬운 설명
4. **용어 주석**: 기술 용어에 적절한 한국어 설명

## 📚 참고 자료

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Llama-3.2 Model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
