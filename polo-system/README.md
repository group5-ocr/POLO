# POLO
Paper Only Look Once,  AI 분야 논문 보조 시각화 프로그램

## 개인 공부 링크
1. 이론 공부
https://www.notion.so/2025-08-28-POLO-25ea304ac11280d08b02cfd0e4d23861?source=copy_link

2. 코드 분석 및 개인 공부
https://www.notion.so/POLO-25aa304ac11280fdaadce9ba2ee41d16?source=copy_link


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
  --target_modules q_proj,k_proj,v_proj,o_proj  # ✅ 꼭 지정!
