# models/easy/training/qlora.py
import argparse
import os
from typing import List
import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from dataload import JsonlSFTDataset
from formatters import simplification_formatter
from collate import SimplifyCollator
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", nargs="+", required=True)
    ap.add_argument("--valid_jsonl", nargs="+", default=[])
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--output_dir", default="./checkpoints/qwen2.5-7b-easy")
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    return ap.parse_args()
def load_datasets(paths: List[str]):
    datasets = []
    for p in paths:
        datasets.append(JsonlSFTDataset(p, simplification_formatter))
    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 4bit 로드
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        quantization_config=bnb.nn.quantization.QuantizationConfig(load_in_4bit=True)
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    collator = SimplifyCollator(tokenizer, max_length=args.max_len)
    train_ds = load_datasets(args.train_jsonl)
    eval_ds = load_datasets(args.valid_jsonl) if args.valid_jsonl else None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps" if eval_ds else "no",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_ds else None,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        dataloader_pin_memory=False,
        report_to=["tensorboard"],
        save_total_limit=2
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
if __name__ == "__main__":
    main()

