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
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

# =============================
# 1. 전처리 유틸
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

PLAIN_TEMPLATE = "<|system|>\n{sys}\n</|system|>\n<|user|>\n{txt}\n</|user|>\n<|assistant|>\n"


def build_text_from_row(row: Dict) -> str:
    if all(k in row for k in ("instruction", "output")):
        instr = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", "")).strip()
        inp_block = f"\n입력:\n{inp}" if inp else ""
        return INSTRUCT_TEMPLATE.format(sys=SYS_PROMPT, instr=instr, inp=inp_block, out=out)
    elif "text" in row:
        return PLAIN_TEMPLATE.format(sys=SYS_PROMPT, txt=str(row["text"]).strip())
    else:
        txt = json.dumps(row, ensure_ascii=False)
        return PLAIN_TEMPLATE.format(sys=SYS_PROMPT, txt=txt)


# =============================
# 2. 모델/토크나이저 로딩
# =============================

def prepare_model_and_tokenizer(model_name_or_path: str,
                                use_bf16: bool = True,
                                load_in_4bit: bool = True,
                                bnb_quant_type: str = "nf4",
                                bnb_compute_dtype: str = "bfloat16",
                                gradient_checkpointing: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_dtype = torch.bfloat16 if bnb_compute_dtype.lower().startswith("bf") else torch.float16
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_quant_type,
            bnb_4bit_compute_dtype=quant_dtype,
            bnb_4bit_use_double_quant=True,
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA(GPU)가 감지되지 않았습니다.")
    torch.cuda.set_device(0)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, tokenizer


# =============================
# 3. 데이터셋 로딩
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
            raise ValueError("지원하지 않는 train_file 확장자입니다.")
    else:
        raise ValueError("--dataset_name 또는 --train_file 중 하나는 필요합니다.")

    return ds.map(lambda ex: {"text": build_text_from_row(ex)},
                  remove_columns=[c for c in ds.column_names if c != "text"])


# =============================
# 4. 학습 함수
# =============================

def train(args):
    print("[cuda]", torch.cuda.get_device_name(0))

    model, tokenizer = prepare_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        use_bf16=args.bf16,
        load_in_4bit=args.bnb_4bit,
        bnb_quant_type=args.bnb_4bit_quant_type,
        bnb_compute_dtype=args.bnb_4bit_compute_dtype,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules.split(",") if args.target_modules else None,
    )

    dataset = load_training_dataset(args.dataset_name, args.train_file)

    if args.train_fraction and 0 < args.train_fraction < 1.0:
        dataset = dataset.shuffle(seed=args.seed)
        dataset = dataset.select(range(int(len(dataset) * args.train_fraction)))

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
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        optim="paged_adamw_8bit",
        report_to=["tensorboard"] if args.report_to_tensorboard else [],
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_first_step=True,
        remove_unused_columns=True,
        overwrite_output_dir=False,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=args.packing,
    )

    checkpoint_path = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    if checkpoint_path:
        print(f"[resume] 체크포인트에서 이어서 학습: {checkpoint_path}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=lora_config,
        train_dataset=dataset,
        args=sft_config,
    )

    trainer.train(resume_from_checkpoint=checkpoint_path)

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[완료] 모델 저장 완료: {args.output_dir}")


# =============================
# 5. LoRA 병합
# =============================

def merge_adapters(args):
    if not args.adapter_path:
        raise ValueError("--adapter_path 가 필요합니다.")

    base = args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        base,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    merged = model.merge_and_unload()
    os.makedirs(args.merge_save_dir, exist_ok=True)
    merged.save_pretrained(args.merge_save_dir, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    tok.save_pretrained(args.merge_save_dir)
    print(f"[완료] 병합된 모델 저장 완료: {args.merge_save_dir}")


# =============================
# 6. CLI 진입점
# =============================

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_every_steps", type=int, default=1000)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--report_to_tensorboard", action="store_true")
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
    p.add_argument("--bnb_4bit", action="store_true")
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    p.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default=None)
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
