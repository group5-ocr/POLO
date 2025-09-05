@echo off
echo Starting QLoRA Training...
echo.

cd /d "%~dp0"

.\venv\Scripts\python.exe training\qlora.py ^
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct ^
  --train_file training\train.jsonl ^
  --output_dir outputs\qwen25-3b-qlora ^
  --train_fraction 0.1 ^
  --num_train_epochs 5 ^
  --save_every_steps 1000 ^
  --bf16 True ^
  --bnb_4bit True ^
  --bnb_4bit_quant_type nf4 ^
  --per_device_train_batch_size 1 ^
  --gradient_accumulation_steps 16 ^
  --max_seq_length 512 ^
  --gradient_checkpointing True

echo.
echo Training completed!
pause
