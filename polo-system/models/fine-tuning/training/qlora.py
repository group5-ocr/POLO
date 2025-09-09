# qlora.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import argparse
from typing import Dict, Optional

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# =============================
# 1) 프롬프트 유틸
# =============================
SYS_PROMPT = (
    "당신은 어려운 논문을 쉽게 풀어 설명하는 수학 및 과학 커뮤니케이터입니다. "
    "핵심만 정확히, 쉬운 한국어로 단계적으로 설명하세요."
)

INSTRUCT_TEMPLATE = (
    "<|system|>\n{sys}\n</|system|>\n"
    "<|user|>\n{instr}\n{inp}\n</|user|>\n"
    "<|assistant|>\n{out}\n</|assistant|>"
)

PLAIN_TEMPLATE = (
    "<|system|>\n{sys}\n</|system|>\n"
    "<|user|>\n{txt}\n</|user|>\n"
    "<|assistant|>\n"
)

def build_text_from_row(row: Dict) -> str:
    # 지원 포맷 1) {"instruction": "...", "input": "...", "output": "..."}
    if all(k in row for k in ("instruction", "output")):
        instr = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", "")).strip()
        inp_block = f"\n입력:\n{inp}" if inp else ""
        return INSTRUCT_TEMPLATE.format(sys=SYS_PROMPT, instr=instr, inp=inp_block, out=out)
    # 지원 포맷 2) {"text": "..."}   (이미 완성된 프롬프트/텍스트)
    elif "text" in row:
        return PLAIN_TEMPLATE.format(sys=SYS_PROMPT, txt=str(row["text"]).strip())
    # 그 외 포맷은 통째로 문자열화
    else:
        txt = json.dumps(row, ensure_ascii=False)
        return PLAIN_TEMPLATE.format(sys=SYS_PROMPT, txt=txt)


# =============================
# 2) 모델/토크나이저 준비
# =============================
def str2dtype(name: str):
    name = (name or "").lower()
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if name in ["fp16", "float16", "half"]:
        return torch.float16
    if name in ["fp32", "float32"]:
        return torch.float32
    # 기본값은 bfloat16
    return torch.bfloat16

def build_bnb_config(load_in_4bit: bool, quant_type: str, compute_dtype: str | torch.dtype):
    if isinstance(compute_dtype, str):
        compute_dtype = str2dtype(compute_dtype)
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type=quant_type,   # "nf4" 권장
        bnb_4bit_compute_dtype=compute_dtype,
    )

def prepare_model_and_tokenizer(
    model_name_or_path: str,
    use_bf16: bool = True,
    load_in_4bit: bool = True,
    bnb_quant_type: str = "nf4",
    bnb_compute_dtype: str = "bfloat16",
    gradient_checkpointing: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = build_bnb_config(
        load_in_4bit=load_in_4bit,
        quant_type=bnb_quant_type,
        compute_dtype=bnb_compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    # 일부 환경에서 더 안정적
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass

    # k-bit 학습 준비 + 입력 requires_grad 연결
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    try:
        model.enable_input_require_grads()  # transformers >= 4.39
    except Exception:
        # 구버전 대응
        model.get_input_embeddings().requires_grad_(True)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # 반드시 비활성화

    return model, tokenizer


# =============================
# 3) 데이터셋 로딩
# =============================
def load_training_dataset(dataset_name: Optional[str], train_file: Optional[str]) -> Dataset:
    if dataset_name:
        ds = load_dataset(dataset_name)
        ds = ds["train"] if "train" in ds else list(ds.values())[0]
    elif train_file:
        ext = os.path.splitext(train_file)[1].lower()
        if ext in [".jsonl", ".json"]:
            ds = load_dataset("json", data_files=train_file)["train"]
        else:
            raise ValueError("지원하지 않는 train_file 확장자입니다. (.jsonl / .json 권장)")
    else:
        raise ValueError("--dataset_name 또는 --train_file 중 하나는 필요합니다.")

    return ds.map(
        lambda ex: {"text": build_text_from_row(ex)},
        remove_columns=[c for c in ds.column_names if c != "text"]
    )


# =============================
# 4) 학습 루틴
# =============================
def print_trainable_parameters(model):
    total, trainable = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = (trainable / total * 100.0) if total else 0.0
    print(f"[param] Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")

def train(args):
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[cuda] Using device: {dev_name}")

    model, tokenizer = prepare_model_and_tokenizer(
        args.model_name_or_path,
        use_bf16=args.bf16,
        load_in_4bit=args.bnb_4bit,
        bnb_quant_type=args.bnb_4bit_quant_type,
        bnb_compute_dtype=args.bnb_4bit_compute_dtype,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # target_modules 기본값 강제
    default_targets = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    targets = (
        [t.strip() for t in args.target_modules.split(",") if t.strip()]
        if args.target_modules else default_targets
    )
    if args.target_modules is None:
        print("⚠️  target_modules 미지정 → 기본값 사용:", ",".join(default_targets))

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )

    dataset = load_training_dataset(args.dataset_name, args.train_file)
    print("[데이터 샘플]", dataset[0])

    if args.train_fraction < 1.0:
        dataset = dataset.shuffle(seed=args.seed)
        dataset = dataset.select(range(int(len(dataset) * args.train_fraction)))
        print(f"[data] train_fraction={args.train_fraction} → {len(dataset)} samples")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_every_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=True,  # 모델 측과 일관
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        optim="paged_adamw_8bit",
        report_to=["tensorboard"] if args.report_to_tensorboard else [],
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_first_step=True,
        remove_unused_columns=True,
        overwrite_output_dir=False,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
    )

    checkpoint_path = args.resume_from_checkpoint or None
    if checkpoint_path:
        print(f"[resume] 체크포인트에서 이어서 학습: {checkpoint_path}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=lora_config,
        train_dataset=dataset,
        args=sft_config,
        dataset_text_field="text",  # 여기에서 명시 (TRL 0.9.x 안정)
    )

    print_trainable_parameters(model)
    print("[debug] Grad 체크:")
    any_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad:
            any_grad = True
            print(f" - ✅ {name} requires_grad")
    if not any_grad:
        raise RuntimeError("❌ 학습 가능한 파라미터가 없습니다. LoRA target_modules를 확인하세요.")

    trainer.train(resume_from_checkpoint=checkpoint_path)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[완료] 모델 저장: {args.output_dir}")


# =============================
# 5) LoRA 병합 (옵션)
# =============================
def merge_adapters(args):
    if not args.adapter_path:
        raise ValueError("--adapter_path 가 필요합니다.")

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, args.adapter_path).merge_and_unload()

    os.makedirs(args.merge_save_dir, exist_ok=True)
    merged.save_pretrained(args.merge_save_dir, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tok.save_pretrained(args.merge_save_dir)
    print(f"[완료] 병합 저장: {args.merge_save_dir}")


# =============================
# 6) CLI
# =============================
def build_parser():
    p = argparse.ArgumentParser()
    # 데이터/모델
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./outputs")

    # 로깅/저장
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_every_steps", type=int, default=1000)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--report_to_tensorboard", action="store_true")

    # 학습 하이퍼파라미터
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--packing", action="store_true")
    p.add_argument("--train_fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # 4bit/정밀도
    p.add_argument("--bnb_4bit", action="store_true")
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    p.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    # 재개/타깃/LoRA
    p.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default=None,
    help="Fine-tuning을 중단한 시점의 체크포인트 경로를 지정합니다."
    )   
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default=None)  # 콤마 분리 문자열

    # 병합 옵션
    p.add_argument("--merge_adapters", action="store_true")
    p.add_argument("--adapter_path", type=str, default=None)
    p.add_argument("--merge_save_dir", type=str, default="./merged")
    return p

def main():
    args = build_parser().parse_args()
    if args.merge_adapters:
        merge_adapters(args)
    else:
        train(args)

if __name__ == "__main__":
    main()
