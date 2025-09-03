# models/easy/training/prepare/prepare_easy_jsonl.py
import json
import argparse
from pathlib import Path
def normalize_pair_list(objs):
    pairs = []
    for ex in objs:
        # WikiLarge 스타일
        if "Normal" in ex and "Simple" in ex:
            src = (ex["Normal"] or "").strip()
            tgt = (ex["Simple"] or "").strip()
            if src and tgt:
                pairs.append({"input": src, "output": tgt})
            continue
        # ASSET 스타일
        if "original" in ex and "simplifications" in ex and isinstance(ex["simplifications"], list):
            src = (ex["original"] or "").strip()
            for s in ex["simplifications"]:
                s = (s or "").strip()
                if src and s:
                    pairs.append({"input": src, "output": s})
            continue
    return pairs
def load_any_json(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        else:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
    return items
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".json 또는 .jsonl")
    ap.add_argument("--output_jsonl", required=True)
    args = ap.parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    objs = load_any_json(in_path)
    pairs = normalize_pair_list(objs)
    with out_path.open("w", encoding="utf-8") as w:
        for p in pairs:
            w.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} lines to {out_path}")
if __name__ == "__main__":
    main()