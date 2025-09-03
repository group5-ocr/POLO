"""
ccdv/arxiv-summarization -> 우리 파이프라인 입력(gemini_inputs.jsonl) 변환기
- 입력: HF datasets 'ccdv/arxiv-summarization' (config: "document" 권장)
- 출력: {SAVE_ROOT}/summarization/gemini_inputs.jsonl
  레코드 = {doc_id, title, abstract, priority_paragraphs, chunks}
- 프롬프트는 저장하지 않음(파일엔 원문 조각만)
"""

import argparse, hashlib, json, re
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def split_paragraphs(text: str, min_len: int = 60):
    rough = re.split(r"(?<=[\.\?\!])\s+", text or "")
    paras, buf = [], []
    for s in rough:
        if not s: continue
        buf.append(s)
        if len(" ".join(buf)) >= min_len:
            paras.append(clean(" ".join(buf))); buf = []
    if buf: paras.append(clean(" ".join(buf)))
    return paras

def score_and_topk(paras, k=12):
    kws = [r"\bwe (propose|introduce|present)\b", r"\bin this paper\b", r"\bcontribution(s)?\b",
           r"\bmethod(s|ology)\b", r"\bresults? (show|demonstrate)\b", r"\bstate[- ]of[- ]the[- ]art\b",
           r"\boutperform(s|ed)?\b", r"\blimitation(s)?\b", r"\bfuture work\b"]
    scored = []
    for i, p in enumerate(paras):
        s = 0.0; L = len(p); low = p.lower()
        for rx in kws:
            if re.search(rx, low): s += 1.0
        if L < 80: s -= 0.4
        if L > 1200: s -= 0.4
        if i <= 3: s += 0.2
        scored.append((s, i, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, _, p in scored[:k]]

def make_chunks(text: str, max_chars=3200, overlap=280, cap=9000):
    text = (text or "")[:cap]
    out, i, N = [], 0, len(text)
    while i < N:
        j = min(i + max_chars, N)
        out.append(clean(text[i:j]))
        if j >= N: break
        i = j - overlap
    return out

def get_title(ex):
    for key in ("title", "paper_title"):
        if key in ex and isinstance(ex[key], str) and ex[key].strip():
            return ex[key].strip()
    first = (ex.get("article") or "").strip().split("\n", 1)[0]
    return first[:160] if first else ""

def get_doc_id(ex, idx, split):
    for key in ("id", "paper_id", "arxiv_id"):
        if key in ex and isinstance(ex[key], str) and ex[key].strip():
            return ex[key].strip()
    h = hashlib.md5((ex.get("article","")[:2048]).encode("utf-8")).hexdigest()[:12]
    return f"ccdv_{split}_{idx}_{h}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-root", default="./arxiv_data", type=str, help="산출물 루트")
    ap.add_argument("--split", default="train", choices=["train","validation","test"], help="HF split")
    ap.add_argument("--config", default="document", choices=["document","section"], help="HF config")
    ap.add_argument("--limit", type=int, default=0, help="샘플 제한(0=무제한)")
    ap.add_argument("--min_body_chars", type=int, default=200, help="본문 최소 길이")
    ap.add_argument("--chunks_cap", type=int, default=9000, help="본문 최대 사용 글자")
    ap.add_argument("--k_priority", type=int, default=12, help="우선 문단 개수")
    args = ap.parse_args()

    SAVE_ROOT = Path(args.save_root)
    SUM_DIR   = SAVE_ROOT / "summarization"
    SUM_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = SUM_DIR / "gemini_inputs.jsonl"

    ds = load_dataset("ccdv/arxiv-summarization", args.config, split=args.split)

    kept = 0
    with out_jsonl.open("w", encoding="utf-8") as wf:
        for i, ex in enumerate(tqdm(ds, desc=f"build {args.split}/{args.config}")):
            if args.limit and kept >= args.limit:
                break
            article = (ex.get("article") or "").strip()
            abstract = (ex.get("abstract") or "").strip()
            if len(article) < args.min_body_chars:
                continue

            title  = get_title(ex)
            doc_id = get_doc_id(ex, i, args.split)

            paras  = split_paragraphs(article)
            prios  = score_and_topk(paras, k=args.k_priority)
            chunks = make_chunks(article, cap=args.chunks_cap)

            rec = {
                "doc_id": doc_id,
                "title": title,
                "abstract": abstract,
                "priority_paragraphs": prios,
                "chunks": chunks
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"wrote {out_jsonl}  (samples={kept})")

if __name__ == "__main__":
    main()

