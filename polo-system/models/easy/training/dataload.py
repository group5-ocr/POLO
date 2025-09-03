# models/easy/training/dataload.py
from torch.utils.data import Dataset
import json
class JsonlSFTDataset(Dataset):
    """
    각 줄: {"input": "...", "output": "..."}
    __getitem__은 {"prompt": str, "target": str} 반환
    """
    def __init__(self, path, formatter, max_len=8192):
        self.items = []
        self.formatter = formatter
        self.max_len = max_len
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                self.items.append(ex)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        ex = self.items[i]
        prompt, target = self.formatter(ex["input"], ex["output"])
        return {"prompt": prompt, "target": target}