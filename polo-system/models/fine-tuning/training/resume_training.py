#!/usr/bin/env python3
"""
YOLO Easy 모델 학습 재개 스크립트
손실값이 0.1에 가까워지면 자동으로 학습을 중단합니다.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # 현재 디렉토리를 training 폴더로 설정
    training_dir = Path(__file__).parent
    os.chdir(training_dir)
    
    print("🚀 YOLO Easy 모델 학습 재개 시작")
    print("=" * 60)
    
    # 체크포인트 경로 확인
    checkpoint_dir = Path("../outputs/yolo-easy-qlora")
    if not checkpoint_dir.exists():
        print("❌ 체크포인트 디렉토리가 없습니다. 처음부터 학습을 시작합니다.")
        checkpoint_path = None
    else:
        # 가장 최근 체크포인트 찾기
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            checkpoint_path = str(latest_checkpoint)
            print(f"✅ 최신 체크포인트 발견: {latest_checkpoint.name}")
        else:
            print("❌ 체크포인트를 찾을 수 없습니다. 처음부터 학습을 시작합니다.")
            checkpoint_path = None
    
    # 학습 명령어 구성
    cmd = [
        "python", "qlora_yolo.py",
        "--model_name_or_path", "meta-llama/Llama-3.2-3B-Instruct",
        "--yolo_json_path", "yolo_v1.json",
        "--output_dir", "../outputs/yolo-easy-qlora",
        "--num_train_epochs", "100.0",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "8",
        "--learning_rate", "5e-5",
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.1",
        "--max_seq_length", "1024",
        "--train_fraction", "1.0",
        "--seed", "42",
        "--target_loss", "0.1",
        "--early_stopping_patience", "5",
        "--early_stopping_min_delta", "0.001",
        "--early_stopping_min_epochs", "10",
        "--eval_steps", "50",
        "--logging_steps", "5",
        "--save_every_steps", "200",
        "--save_total_limit", "3",
        "--lora_r", "32",
        "--lora_alpha", "64",
        "--lora_dropout", "0.05",
        "--target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "--bnb_4bit",
        "--bnb_4bit_quant_type", "nf4",
        "--bnb_4bit_compute_dtype", "bfloat16",
        "--bf16",
        "--gradient_checkpointing",
        "--report_to_tensorboard"
    ]
    
    # 체크포인트가 있으면 재개
    if checkpoint_path:
        cmd.extend(["--resume_from_checkpoint", checkpoint_path])
        print(f"🔄 체크포인트에서 학습 재개: {checkpoint_path}")
    else:
        print("🆕 처음부터 학습 시작")
    
    print("\n📋 학습 설정:")
    print(f"   - 목표 손실값: 0.1")
    print(f"   - 최대 에포크: 100")
    print(f"   - 학습률: 5e-5")
    print(f"   - 배치 크기: 2 × 8 = 16")
    print(f"   - LoRA rank: 32, alpha: 64")
    print(f"   - Patience: 5 에포크")
    print(f"   - 최소 에포크: 10")
    
    print("\n🎯 Early Stopping 조건:")
    print("   1. 손실값이 0.1 이하로 떨어지면 즉시 중단")
    print("   2. 5 에포크 동안 개선이 없으면 중단")
    print("   3. 최소 10 에포크는 학습 진행")
    
    print("\n" + "=" * 60)
    print("학습을 시작합니다...")
    print("=" * 60)
    
    try:
        # 학습 실행
        result = subprocess.run(cmd, check=True)
        print("\n✅ 학습이 성공적으로 완료되었습니다!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 학습 중 오류가 발생했습니다: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 학습이 중단되었습니다.")
        sys.exit(0)

if __name__ == "__main__":
    main()
