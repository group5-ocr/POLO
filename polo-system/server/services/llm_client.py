import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "meta-llama/Llama-3.2-3B")  # 예시
WEIGHTS_DIR   = os.getenv("MODEL_WEIGHTS_DIR", "/weights/easy-lora")
LORA_WEIGHTS  = os.path.join(WEIGHTS_DIR, "adapter_model.safetensors")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(base, WEIGHTS_DIR)