## 도커
1. 빌드 생성 (Dockerfile, requirements.train.txt 변경)
docker compose build --no-cache easy-train

2. qlora.py, train.jsonl, 파라미터 수정 시에
docker compose up easy-train

3. 직접 도커 실행
docker exec -it easy-train bash
python training/qlora.py \
  --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
  --train_file training/train.jsonl \
  --output_dir outputs/llama32-qlora \
  --report_to_tensorboard \
  --train_fraction 0.3 \
  --num_train_epochs 3 \
  --save_every_steps 500 \
  --logging_steps 10 \
  --bf16 \
  --bnb_4bit \
  --bnb_4bit_quant_type nf4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 256 \
  --gradient_checkpointing \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --target_modules q_proj,k_proj,v_proj,o_proj

--------

## 서버

