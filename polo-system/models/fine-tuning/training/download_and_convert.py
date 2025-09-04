"""
데이터셋 다운로드 및 JSONL 변환 자동화 스크립트

지원 데이터셋:
- bogdancazan/wikilarge-text-simplification (영어 문장 단순화)
- facebook/asset (문장 단순화)

사용법:
  python training/download_and_convert.py --dataset wikilarge
  python training/download_and_convert.py --dataset asset
  python training/download_and_convert.py --dataset wikilarge --sample_size 10000
"""

import argparse
import os
import json
from typing import Dict, List, Optional
from datasets import load_dataset


def download_wikilarge(sample_size: Optional[int] = None) -> List[Dict]:
    """WikiLarge 데이터셋 다운로드 및 변환"""
    print("[1/3] WikiLarge 데이터셋 다운로드 중...")
    dataset = load_dataset("bogdancazan/wikilarge-text-simplification")
    train_data = dataset["train"]
    
    if sample_size:
        train_data = train_data.select(range(min(sample_size, len(train_data))))
        print(f"[1/3] 샘플 크기 제한: {len(train_data)}개")
    
    print(f"[1/3] 총 {len(train_data)}개 문장 쌍 다운로드 완료")
    
    # 변환
    print("[2/3] JSONL 형식으로 변환 중...")
    converted_data = []
    for i, example in enumerate(train_data):
        if i % 10000 == 0:
            print(f"[2/3] 진행률: {i}/{len(train_data)} ({i/len(train_data)*100:.1f}%)")
        
        # 원문과 단순화된 문장 추출
        original = example["Normal"].strip()
        simplified = example["Simple"].strip()
        
        if original and simplified and len(original) > 10 and len(simplified) > 5:
            converted_data.append({
                "instruction": "아래 문장을 쉬운 한국어로 설명하세요.",
                "input": original,
                "output": simplified
            })
    
    print(f"[2/3] 변환 완료: {len(converted_data)}개 유효한 문장 쌍")
    return converted_data


def download_asset(sample_size: Optional[int] = None) -> List[Dict]:
    """ASSET 데이터셋 다운로드 및 변환"""
    print("[1/3] ASSET 데이터셋 다운로드 중...")
    dataset = load_dataset("facebook/asset")
    train_data = dataset["simplification"]
    
    if sample_size:
        train_data = train_data.select(range(min(sample_size, len(train_data))))
        print(f"[1/3] 샘플 크기 제한: {len(train_data)}개")
    
    print(f"[1/3] 총 {len(train_data)}개 문장 쌍 다운로드 완료")
    
    # 변환
    print("[2/3] JSONL 형식으로 변환 중...")
    converted_data = []
    for i, example in enumerate(train_data):
        if i % 10000 == 0:
            print(f"[2/3] 진행률: {i}/{len(train_data)} ({i/len(train_data)*100:.1f}%)")
        
        # 원문과 단순화된 문장들 추출
        original = example["original"].strip()
        simplifications = example["simplifications"]
        
        if original and simplifications:
            # 첫 번째 단순화 버전 사용
            simplified = simplifications[0].strip() if simplifications else ""
            
            if original and simplified and len(original) > 10 and len(simplified) > 5:
                converted_data.append({
                    "instruction": "아래 문장을 쉬운 한국어로 설명하세요.",
                    "input": original,
                    "output": simplified
                })
    
    print(f"[2/3] 변환 완료: {len(converted_data)}개 유효한 문장 쌍")
    return converted_data


def save_jsonl(data: List[Dict], output_path: str):
    """JSONL 파일로 저장"""
    print(f"[3/3] JSONL 파일 저장 중: {output_path}")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"[3/3] 저장 완료: {len(data)}개 문장 쌍")
    print(f"[3/3] 파일 크기: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="데이터셋 다운로드 및 JSONL 변환")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["wikilarge", "asset"],
                       help="다운로드할 데이터셋 선택")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="샘플 크기 제한 (테스트용)")
    parser.add_argument("--output", type=str, 
                       default="training/train.jsonl",
                       help="출력 JSONL 파일 경로")
    
    args = parser.parse_args()
    
    print(f"=== {args.dataset.upper()} 데이터셋 다운로드 및 변환 시작 ===")
    
    # 데이터셋 다운로드 및 변환
    if args.dataset == "wikilarge":
        data = download_wikilarge(args.sample_size)
    elif args.dataset == "asset":
        data = download_asset(args.sample_size)
    
    # JSONL 저장
    save_jsonl(data, args.output)
    
    print(f"=== 완료! {args.output} 파일이 생성되었습니다. ===")
    print(f"총 {len(data)}개 문장 쌍으로 QLoRA 학습을 시작할 수 있습니다.")


if __name__ == "__main__":
    main()
