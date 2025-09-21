FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 시스템 패키지 업데이트 및 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 심볼릭 링크 생성
RUN ln -s /usr/bin/python3 /usr/bin/python

# 작업 디렉토리 설정
WORKDIR /workspace

# Python 패키지 설치
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir \
    transformers==4.35.0 \
    peft==0.6.0 \
    trl==0.7.0 \
    datasets==2.14.0 \
    bitsandbytes==0.41.0 \
    accelerate==0.24.0 \
    huggingface_hub==0.17.0 \
    tensorboard==2.14.0 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0

# 학습 스크립트 복사
COPY . /workspace/

# 실행 권한 부여
RUN chmod +x /workspace/*.py

# 기본 명령어
CMD ["python", "qlora_yolo.py", "--help"]
