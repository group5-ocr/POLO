#!/usr/bin/env python3
"""
YOLO 논문 Easy 모델 파인튜닝 실행 스크립트
파라미터 정보와 함께 학습 진행
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import json

def analyze_model_parameters(model_path: str = "meta-llama/Llama-3.2-3B-Instruct"):
    """모델 파라미터 분석"""
    print("🔍 모델 파라미터 분석 중...")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"📝 토크나이저 vocab 크기: {len(tokenizer)}")
    
    # 모델 로드 (CPU에서 파라미터만 확인)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # 정확한 계산을 위해 float32
        device_map="cpu",
        trust_remote_code=True
    )
    
    # 전체 파라미터 계산
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 전체 파라미터: {total_params:,} ({total_params/1e9:.2f}B)")
    
    # 레이어별 파라미터 분석
    print("\n📋 레이어별 파라미터 분석:")
    layer_params = {}
    for name, param in model.named_parameters():
        layer_type = name.split('.')[0] if '.' in name else name
        if layer_type not in layer_params:
            layer_params[layer_type] = 0
        layer_params[layer_type] += param.numel()
    
    for layer, count in sorted(layer_params.items()):
        print(f"  {layer}: {count:,} ({count/total_params*100:.1f}%)")
    
    # LoRA 설정 시뮬레이션
    print("\n🔧 LoRA 설정 시뮬레이션:")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_r = 16
    lora_alpha = 32
    
    # 각 타겟 모듈의 파라미터 계산
    lora_params = 0
    for name, param in model.named_parameters():
        if any(target in name for target in target_modules):
            # LoRA 파라미터 = 2 * rank * (input_dim + output_dim)
            # 실제로는 더 복잡하지만 근사치
            if 'weight' in name:
                input_dim = param.shape[1]
                output_dim = param.shape[0]
                lora_param_count = 2 * lora_r * (input_dim + output_dim)
                lora_params += lora_param_count
                print(f"  {name}: {lora_param_count:,} LoRA 파라미터")
    
    print(f"\n📈 LoRA 파라미터 예상: {lora_params:,} ({lora_params/total_params*100:.3f}%)")
    print(f"💾 메모리 절약: {(total_params - lora_params)/total_params*100:.1f}%")
    
    return total_params, lora_params

def check_yolo_data(json_path: str = "training/yolo_v1.json"):
    """YOLO 데이터 분석"""
    print(f"\n📚 YOLO 데이터 분석: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"❌ 파일을 찾을 수 없습니다: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📄 총 문장 수: {len(data)}")
    
    # 섹션별 분석
    sections = {}
    for item in data:
        major = item['section_major']
        minor = item['section_minor']
        key = f"{major}.{minor}"
        if key not in sections:
            sections[key] = 0
        sections[key] += 1
    
    print("📊 섹션별 문장 수:")
    for key in sorted(sections.keys()):
        print(f"  {key}: {sections[key]} 문장")
    
    # 텍스트 길이 분석
    original_lengths = [len(item['original']) for item in data]
    simplified_lengths = [len(item['simplified']) for item in data]
    
    print(f"\n📏 텍스트 길이 분석:")
    print(f"  원문 평균: {sum(original_lengths)/len(original_lengths):.0f} 문자")
    print(f"  간소화 평균: {sum(simplified_lengths)/len(simplified_lengths):.0f} 문자")
    print(f"  확장 비율: {sum(simplified_lengths)/sum(original_lengths):.2f}x")

def main():
    print("🚀 YOLO 논문 Easy 모델 파인튜닝 준비")
    print("=" * 50)
    
    # 모델 파라미터 분석
    total_params, lora_params = analyze_model_parameters()
    
    # YOLO 데이터 분석
    check_yolo_data()
    
    print("\n" + "=" * 50)
    print("✅ 분석 완료!")
    print("\n📋 학습 설정 요약:")
    print("  - 모델: meta-llama/Llama-3.2-3B-Instruct")
    print("  - 전체 파라미터: {:,} ({:.2f}B)".format(total_params, total_params/1e9))
    print("  - LoRA 파라미터: {:,} ({:.3f}%)".format(lora_params, lora_params/total_params*100))
    print("  - 학습 방식: QLoRA (4-bit quantization)")
    print("  - 타겟 모듈: q_proj, k_proj, v_proj, o_proj")
    print("  - LoRA rank: 16, alpha: 32")
    
    print("\n🎯 다음 단계:")
    print("1. 체크포인트 확인: python check_checkpoint.py")
    print("2. 학습 시작: docker-compose up yolo-train")
    print("3. 학습 모니터링: docker logs -f yolo-train")

if __name__ == "__main__":
    main()
