# models/easy/training/collate.py
import torch
class SimplifyCollator:
    """
    - prompt + target을 이어붙여 토크나이즈
    - labels에서 prompt 구간은 -100 마스킹, target 구간만 학습
    """
    def __init__(self, tokenizer, max_length=8192):
        self.tok = tokenizer
        self.max_length = max_length
    def __call__(self, batch):
        prompts = [b["prompt"] for b in batch]
        targets = [b["target"] for b in batch]
        # 각각 토크나이즈 (길이 측정용)
        prompt_enc = self.tok(prompts, add_special_tokens=False)
        target_enc = self.tok(targets, add_special_tokens=False)
        input_ids = []
        attention_mask = []
        labels = []
        for p_ids, t_ids in zip(prompt_enc["input_ids"], target_enc["input_ids"]):
            ids = p_ids + t_ids + [self.tok.eos_token_id]  # EOS 보장
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
            attn = [1] * len(ids)
            # 라벨: 프롬프트 길이만큼 -100, 나머지는 그대로
            lab = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
            # 패딩은 Trainer가 pad_to_multiple_of로 처리할 수도 있지만 여기선 직접 맞춤
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attention_mask.append(torch.tensor(attn, dtype=torch.long))
            labels.append(torch.tensor(lab[:len(ids)], dtype=torch.long))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tok.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }