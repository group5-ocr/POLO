import os, re, json, time, random, itertools
from pathlib import Path
import google.generativeai as genai

SAVE_ROOT = "./arxiv_data"
SUM_DIR   = Path(SAVE_ROOT) / "summarization"
inp = SUM_DIR / "gemini_inputs.jsonl"      # 프롬프트 없는 입력
out = SUM_DIR / "gemini_responses.jsonl"   # {"doc_id","summary_ko"}만 저장

# ── 환경 변수 ───────────────────────────────────────────
# export GEMINI_API_KEY=...
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash-8b")
model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={"temperature": 0.2, "max_output_tokens": 1536}
)

# ── 길이/속도 파라미터 ─────────────────────────────────
CHUNKS_PER_DOC       = int(os.environ.get("SUMM_CHUNKS_PER_DOC", "2"))
PRIOS_JOIN_MAX_CHARS = int(os.environ.get("SUMM_PRIOS_MAX", "1800"))
CHUNK_TRIM_CHARS     = int(os.environ.get("SUMM_CHUNK_TRIM", "1500"))
TOTAL_BUDGET_CHARS   = int(os.environ.get("SUMM_TOTAL_BUDGET", "7000"))
SLEEP_BETWEEN_CALLS  = float(os.environ.get("GEMINI_SLEEP", "6.0"))
MAX_DOCS             = int(os.environ.get("GEMINI_MAX_DOCS", "0"))  # 0=무제한

PROMPT_SUM = (
    "당신은 AI 논문 전문 분석가입니다.\n"
    "아래 원문 조각(우선순위 문단 + 본문 청크)을 근거로, 다음을 한국어로 간결히 작성하세요:\n"
    "1) 한 줄 핵심 요약(<= 25자)\n"
    "2) 기여(3~5개, 번호 목록)\n"
    "3) 방법 개요(수식/핵심 아이디어)\n"
    "4) 실험 설정(데이터/메트릭)\n"
    "5) 주요 결과(정량/정성)\n"
    "6) 한계와 향후 과제(있다면)\n"
    "모든 주장에는 원문 근거(가능하면 문장 그대로)와 섹션/문단 힌트를 함께 제공하세요.\n"
    "※ 한국어만 출력."
)

def safe_join(parts, max_chars):
    out, total = [], 0
    for p in parts or []:
        p = (p or "").strip()
        if not p: continue
        if total + len(p) > max_chars: break
        out.append(p); total += len(p)
    return "\n".join(out)

def trim(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n]

def enforce_budget(prio_blob, sel_chunks, budget):
    joined = (prio_blob + "\n\n" + "\n\n---\n".join(sel_chunks)).strip()
    if len(joined) <= budget:
        return prio_blob, sel_chunks
    # 초과 시 뒤 청크부터 줄이기 → 마지막 청크 제거/축소
    out_chunks = sel_chunks[:]
    while out_chunks and len((prio_blob + "\n\n" + "\n\n---\n".join(out_chunks))) > budget:
        c = out_chunks[-1]
        if len(c) > 600:      # 너무 크면 잘라서 한번 더 시도
            out_chunks[-1] = c[: max(300, len(c)//2)]
        else:
            out_chunks.pop()
    return prio_blob, out_chunks

def gen_with_backoff(content, max_retries=5, base_delay=4.0):
    for attempt in range(max_retries):
        try:
            return model.generate_content(content)
        except Exception as e:
            msg = str(e)
            if ("TooManyRequests" in msg) or ("429" in msg) or ("quota" in msg.lower()):
                sleep = base_delay * (2 ** attempt) + random.uniform(0, 1.0)
                print(f"[429] backoff {sleep:.1f}s (try {attempt+1}/{max_retries})")
                time.sleep(sleep); continue
            raise
    raise RuntimeError("429 지속 발생: 쿼터 초과 추정")

processed = set()
if out.exists():
    with out.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try: processed.add(json.loads(line)["doc_id"])
            except: pass

assert inp.exists(), f"입력 없음: {inp}"
written = 0
count   = 0

with inp.open(encoding="utf-8") as fi, out.open("a", encoding="utf-8") as fo:
    for line in fi:
        if not line.strip(): continue
        if MAX_DOCS and count >= MAX_DOCS: break

        o = json.loads(line)
        did = o["doc_id"]
        if did in processed: 
            continue

        prios  = o.get("priority_paragraphs", [])
        chunks = o.get("chunks", [])

        prio_blob  = safe_join(prios, PRIOS_JOIN_MAX_CHARS)
        sel_chunks = list(itertools.islice(chunks, CHUNKS_PER_DOC))
        sel_chunks = [trim(c, CHUNK_TRIM_CHARS) for c in sel_chunks]

        # 총 예산 강제
        prio_blob, sel_chunks = enforce_budget(prio_blob, sel_chunks, TOTAL_BUDGET_CHARS)
        body_blob  = "\n\n---\n".join(sel_chunks)

        content = f"{PROMPT_SUM}\n\n[우선순위 문단]\n{prio_blob}\n\n[본문 청크]\n{body_blob}"
        try:
            r = gen_with_backoff(content, max_retries=5, base_delay=4.0)
        except RuntimeError as e:
            print(f"[STOP] {e} — 지금까지 생성된 결과는 저장됨: {out}")
            break

        text = (r.text or "").strip()
        fo.write(json.dumps({"doc_id": did, "summary_ko": text}, ensure_ascii=False) + "\n")
        written += 1
        count   += 1
        time.sleep(SLEEP_BETWEEN_CALLS)

print(f"Gemini 요약 생성: {written}개 → {out}")