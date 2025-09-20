from __future__ import annotations
import os
import json
import argparse
import torch
from typing import Dict, Optional, List
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
import re

token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not token:
    raise RuntimeError("HF token not found in env vars.")
login(token=token, add_to_git_credential=True)

# =============================
# 1) YOLO 데이터 전처리 및 프롬프트 유틸
# =============================
SYS_PROMPT = (
    "너는 AI 논문 텍스트를 원문 의미를 유지한 채 한국어로 쉽게 풀어쓰는 전문가다.\n"
    "규칙:\n"
    "- 영문 전문용어는 영어 표기 그대로 두고, 바로 뒤에 (짧은 한국어 풀이)를 1회만 붙여라. 같은 문단에서 동일 용어는 다시 풀이하지 마라.\n"
    "- 수식/코드/링크/토큰은 변형 금지. 입력에 ⟦MATH_i⟧, ⟦CODE_i⟧ 같은 마스크가 오면 그대로 보존하고, 출력에도 동일하게 남겨라.\n"
    "- 숫자/기호/단위/약어를 바꾸지 마라. 추측/과장은 금지.\n"
    "- 3~5 문단, 각 2~4문장. 매 문단은 연결어(먼저/다음/마지막 등)로 자연스럽게 잇는다.\n"
    "- 불필요한 반복/군더더기 제거. \"방법/코딩방식\" 같은 보조풀이를 과도하게 넣지 마라.\n"
    "\n"
    "형식:\n"
    "- 한국어 본문만 출력. 머리말/꼬리말/설명 금지.\n"
    "\n"
    "주의: ⟦MATH_i⟧, ⟦CODE_i⟧, ⟦URL_i⟧ 같은 토큰은 **그대로** 출력에 복사하라. 내용/순서/공백을 바꾸지 마라."
)

INSTRUCT_TEMPLATE = (
    "<|begin_of_text|>\n"
    "<|start_header_id|>system<|end_header_id|>\n"
    "{sys}\n"
    "<|eot_id|>\n"
    "<|start_header_id|>user<|end_header_id|>\n"
    "[SECTION] {section_title}\n"
    "[TEXT]\n{input}\n\n[REWRITE in Korean]\n"
    "<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "{output}<|eot_id|>"
)

def get_section_name(section_major: int) -> str:
    """섹션 번호를 섹션 이름으로 변환"""
    section_names = {
        0: "Abstract",
        1: "Introduction", 
        2: "Unified Detection",
        3: "Training",
        4: "Experiments",
        5: "Real-Time Detection",
        6: "Conclusion"
    }
    return section_names.get(section_major, f"Section {section_major}")

def clean_yolo_text(text: str) -> str:
    """YOLO 데이터의 반복 문장 제거"""
    # "In simple terms, the method keeps steps minimal..." 반복 제거
    text = re.sub(r"In simple terms, the method keeps steps minimal, uses full-image context, and directly predicts useful numbers \(like box positions\) and labels together\.", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_yolo_data(json_path: str) -> List[Dict]:
    """YOLO JSON 데이터를 Easy 모델용으로 전처리"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    
    # 문단별로 그룹화 (섹션.서브섹션 단위)
    paragraphs = {}
    for item in data:
        key = f"{item['section_major']}.{item['section_minor']}"
        if key not in paragraphs:
            paragraphs[key] = {
                "section_major": item['section_major'],
                "section_minor": item['section_minor'],
                "original_sentences": [],
                "simplified_sentences": []
            }
        
        # 원문과 간소화 문장을 각각 저장
        paragraphs[key]["original_sentences"].append(item["original"])
        paragraphs[key]["simplified_sentences"].append(clean_yolo_text(item["simplified"]))
    
    # 문단별로 학습 데이터 생성
    for key, para in paragraphs.items():
        section_title = get_section_name(para['section_major'])
        
        # 원문과 간소화 텍스트를 문단으로 합치기
        original_text = " ".join(para["original_sentences"])
        simplified_text = " ".join(para["simplified_sentences"])
        
        # 빈 텍스트 스킵
        if not original_text.strip() or not simplified_text.strip():
            continue
            
        processed_data.append({
            "instruction": f"다음 {section_title} 섹션을 쉽게 풀어서 설명해주세요:",
            "input": original_text,
            "output": simplified_text,
            "section_title": section_title,
            "section_major": para['section_major'],
            "section_minor": para['section_minor']
        })
    
    return processed_data

def build_text_from_row(row: Dict) -> str:
    """Easy 모델의 프롬프트 형식으로 변환"""
    section_title = row.get("section_title", "Unknown Section")
    input_text = str(row.get("input", "")).strip()
    output_text = str(row.get("output", "")).strip()
    
    return INSTRUCT_TEMPLATE.format(
        sys=SYS_PROMPT,
        section_title=section_title,
        input=input_text,
        output=output_text
    )

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
    return torch.bfloat16

def build_bnb_config(load_in_4bit: bool, quant_type: str, compute_dtype: str | torch.dtype):
    if isinstance(compute_dtype, str):
        compute_dtype = str2dtype(compute_dtype)
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type=quant_type,
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

    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    try:
        model.enable_input_require_grads()
    except Exception:
        model.get_input_embeddings().requires_grad_(True)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, tokenizer

# =============================
# 3) 데이터셋 로딩
# =============================
def load_yolo_dataset(yolo_json_path: str, train_fraction: float = 0.9) -> tuple[Dataset, Dataset]:
    """YOLO JSON 데이터를 로드하고 train/validation으로 분할"""
    print(f"[데이터] YOLO JSON 로딩: {yolo_json_path}")
    processed_data = preprocess_yolo_data(yolo_json_path)
    print(f"[데이터] 전처리 완료: {len(processed_data)}개 문단")
    
    # Dataset으로 변환
    dataset = Dataset.from_list(processed_data)
    
    # 텍스트 변환
    dataset = dataset.map(
        lambda ex: {"text": build_text_from_row(ex)},
        remove_columns=[c for c in dataset.column_names if c != "text"]
    )
    
    # Train/Validation 분할 (데이터가 적을 때는 train만 사용)
    if len(dataset) < 10:
        print(f"[데이터] 데이터가 적어서 train/validation 분할을 건너뜁니다.")
        print(f"[데이터] Train: {len(dataset)}개, Eval: 0개")
        return dataset, None
    
    dataset = dataset.shuffle(seed=42)
    split_idx = int(len(dataset) * train_fraction)
    
    # 최소 1개는 validation에 남겨두기
    if split_idx >= len(dataset):
        split_idx = len(dataset) - 1
    
    train_dataset = dataset.select(range(split_idx))
    eval_dataset = dataset.select(range(split_idx, len(dataset)))
    
    print(f"[데이터] Train: {len(train_dataset)}개, Eval: {len(eval_dataset)}개")
    
    return train_dataset, eval_dataset

# =============================
# 4) 학습 루틴
# =============================
def print_trainable_parameters(model):
    """학습 가능한 파라미터 개수 출력"""
    total, trainable = 0, 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = (trainable / total * 100.0) if total else 0.0
    print(f"[파라미터] 전체: {total:,} | 학습가능: {trainable:,} ({pct:.2f}%)")
    
    # LoRA 파라미터 상세 분석
    lora_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad and ('lora' in name.lower() or 'adapter' in name.lower()):
            lora_params += p.numel()
    print(f"[LoRA] LoRA 파라미터: {lora_params:,}")

def print_model_info(model):
    """모델 구조 정보 출력"""
    print("\n=== 모델 구조 분석 ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"전체 파라미터: {total_params:,}")
    print(f"학습 가능 파라미터: {trainable_params:,}")
    print(f"학습 비율: {trainable_params/total_params*100:.2f}%")
    
    # 레이어별 파라미터 분석
    print("\n=== 레이어별 파라미터 ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,} ({param.shape})")

def train(args):
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[CUDA] 사용 디바이스: {dev_name}")
    print(f"[메모리] GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # 모델과 토크나이저 준비
    model, tokenizer = prepare_model_and_tokenizer(
        args.model_name_or_path,
        use_bf16=args.bf16,
        load_in_4bit=args.bnb_4bit,
        bnb_quant_type=args.bnb_4bit_quant_type,
        bnb_compute_dtype=args.bnb_4bit_compute_dtype,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # LoRA 설정
    default_targets = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    targets = (
        [t.strip() for t in args.target_modules.split(",") if t.strip()]
        if args.target_modules else default_targets
    )
    
    print(f"[LoRA] 타겟 모듈: {targets}")
    print(f"[LoRA] rank={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )

    # YOLO 데이터셋 로드 (train/validation 분할)
    train_dataset, eval_dataset = load_yolo_dataset(args.yolo_json_path, args.train_fraction)
    print(f"[데이터] Train 샘플 수: {len(train_dataset)}")
    print(f"[데이터] Eval 샘플 수: {len(eval_dataset)}")
    print(f"[데이터] 첫 번째 샘플 미리보기:")
    print("=" * 50)
    print(train_dataset[0]["text"][:500] + "...")
    print("=" * 50)

    # 학습 설정
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_every_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=True,
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
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_strategy="steps" if eval_dataset is not None else "no",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
    )

    # 체크포인트 확인
    checkpoint_path = None
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            checkpoint_path = args.resume_from_checkpoint
            print(f"[체크포인트] 이어서 학습: {checkpoint_path}")
        else:
            print(f"[경고] 체크포인트를 찾을 수 없음: {args.resume_from_checkpoint}")
            print("[체크포인트] 처음부터 학습을 시작합니다.")

    # Early Stopping 콜백 생성 (eval_dataset이 있을 때만)
    callbacks = []
    if eval_dataset is not None:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )
        callbacks.append(early_stopping_callback)
        print(f"[Early Stopping] 활성화됨 (patience={args.early_stopping_patience})")
    else:
        print(f"[Early Stopping] 비활성화됨 (validation 데이터 없음)")
    
    # 트레이너 생성
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=lora_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        dataset_text_field="text",
        callbacks=callbacks,
    )

    # 파라미터 정보 출력
    print_trainable_parameters(model)
    print_model_info(model)

    # 학습 가능한 파라미터 확인
    print("\n[디버그] 학습 가능한 파라미터 체크:")
    any_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad:
            any_grad = True
            print(f" - ✅ {name}: {p.numel():,} parameters")
    if not any_grad:
        raise RuntimeError("❌ 학습 가능한 파라미터가 없습니다. LoRA target_modules를 확인하세요.")

    # 학습 시작
    print(f"\n[학습] 시작 - {args.num_train_epochs} epochs, {len(train_dataset)} samples")
    print(f"[학습] 배치 크기: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"[학습] 학습률: {args.learning_rate}")
    
    trainer.train(resume_from_checkpoint=checkpoint_path)
    
    # 모델 저장
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
    p = argparse.ArgumentParser(description="YOLO 논문 Easy 모델 파인튜닝")
    
    # 데이터/모델
    p.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--yolo_json_path", type=str, default="training/yolo_v1.json")
    p.add_argument("--output_dir", type=str, default="./outputs/yolo-easy-qlora")

    # 로깅/저장
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_every_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--report_to_tensorboard", action="store_true")

    # 학습 하이퍼파라미터
    p.add_argument("--num_train_epochs", type=float, default=15.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--packing", action="store_true")
    p.add_argument("--train_fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    
    # Early Stopping 설정
    p.add_argument("--early_stopping_patience", type=int, default=3, help="Number of evaluations with no improvement after which training will be stopped.")
    p.add_argument("--early_stopping_threshold", type=float, default=0.001, help="Minimum change in the monitored quantity to qualify as an improvement.")
    p.add_argument("--eval_steps", type=int, default=100, help="Run evaluation every N steps.")

    # 4bit/정밀도
    p.add_argument("--bnb_4bit", action="store_true")
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    p.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    # 재개/타깃/LoRA
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

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
