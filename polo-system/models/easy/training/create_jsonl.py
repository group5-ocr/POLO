"""
문장→쉬운문장 데이터셋을 QLoRA 학습용 JSONL로 변환합니다.

지원 입력 소스(둘 중 하나):
- HF datasets 이름: --dataset_name squad_like/name 등
- 로컬 파일: --src_file path/to/file.(csv|json|jsonl)

출력 스키마(JSONL, 줄당 1개):
  {"instruction": "지시문", "input": "원문", "output": "쉬운문장"}

사용 예시:
  python training/create_jsonl.py \
    --src_file data/pairs.csv \
    --text_col hard_text --easy_col easy_text \
    --out_file training/train.jsonl

  python training/create_jsonl.py \
    --dataset_name your/dataset \
    --text_col source --easy_col target \
    --out_file training/train.jsonl
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Optional


def write_jsonl(rows, out_file: str):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {len(rows)} records -> {out_file}")


def from_hf_dataset(dataset_name: str, text_col: str, easy_col: str, instruction: str) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset_name)
    split = ds["train"] if "train" in ds else list(ds.values())[0]
    rows = []
    for ex in split:
        src = str(ex.get(text_col, "")).strip()
        tgt = str(ex.get(easy_col, "")).strip()
        if not tgt:
            continue
        rows.append({"instruction": instruction, "input": src, "output": tgt})
    return rows


def from_local_file(src_file: str, text_col: str, easy_col: str, instruction: str) -> list[dict]:
    ext = os.path.splitext(src_file)[1].lower()
    rows = []
    if ext == ".csv":
        import pandas as pd
        df = pd.read_csv(src_file)
        for _, r in df.iterrows():
            src = str(r.get(text_col, "")).strip()
            tgt = str(r.get(easy_col, "")).strip()
            if not tgt:
                continue
            rows.append({"instruction": instruction, "input": src, "output": tgt})
    elif ext == ".json":
        with open(src_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", [])
        for ex in data:
            src = str(ex.get(text_col, "")).strip()
            tgt = str(ex.get(easy_col, "")).strip()
            if not tgt:
                continue
            rows.append({"instruction": instruction, "input": src, "output": tgt})
    elif ext == ".jsonl":
        with open(src_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                src = str(ex.get(text_col, "")).strip()
                tgt = str(ex.get(easy_col, "")).strip()
                if not tgt:
                    continue
                rows.append({"instruction": instruction, "input": src, "output": tgt})
    else:
        raise ValueError("지원하지 않는 확장자입니다. csv/json/jsonl만 지원합니다.")
    return rows


def build_parser():
    p = argparse.ArgumentParser(description="Create JSONL for QLoRA from HF dataset or local file")
    # 입력 소스
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--src_file", type=str, default=None)
    # 컬럼 매핑
    p.add_argument("--text_col", type=str, required=True, help="원문 문장 컬럼명")
    p.add_argument("--easy_col", type=str, required=True, help="쉬운 문장 컬럼명")
    # 출력/옵션
    p.add_argument("--out_file", type=str, default="training/train.jsonl")
    p.add_argument("--instruction", type=str, default="아래 문장을 쉬운 한국어로 설명하세요.")
    return p


def main():
    args = build_parser().parse_args()
    if not args.dataset_name and not args.src_file:
        raise ValueError("--dataset_name 또는 --src_file 중 하나는 반드시 지정해야 합니다.")

    if args.dataset_name:
        rows = from_hf_dataset(args.dataset_name, args.text_col, args.easy_col, args.instruction)
    else:
        rows = from_local_file(args.src_file, args.text_col, args.easy_col, args.instruction)

    write_jsonl(rows, args.out_file)


if __name__ == "__main__":
    main()


