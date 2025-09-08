import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_BASE = os.getenv("MODEL_BASE", "meta-llama/Llama-3.2-3B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "/models/fine-tuning/outputs/llama32-3b-qlora/checkpoint-600")

class EasyLLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_BASE,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(self.model, ADAPTER_PATH)
        self.model.eval()

    def generate(self, text: str, max_new_tokens: int = 512):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

easy_llm = EasyLLM()