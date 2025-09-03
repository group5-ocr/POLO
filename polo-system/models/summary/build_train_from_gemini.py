from pathlib import Path
import json

SUM_DIR = Path("./arxiv_data/summarization")
inp  = SUM_DIR / "gemini_inputs.jsonl"     # 원문 조각(프롬프트 없음)
resp = SUM_DIR / "gemini_responses.jsonl"  # 한국어 요약만``
out  = SUM_DIR / "train_io_only.jsonl"     # SFT용 (input/output)

by_id = {json.loads(l)["doc_id"]: json.loads(l) for l in inp.open(encoding="utf-8") if l.strip()}

kept = 0
with resp.open(encoding="utf-8") as fr, out.open("w", encoding="utf-8") as fw:
    for line in fr:
        if not line.strip(): continue
        r = json.loads(line)
        did = r.get("doc_id")
        if did not in by_id: continue
        src = by_id[did]

        parts=[]
        if src.get("title"): parts.append(f"[제목]\n{src['title']}")
        if src.get("priority_paragraphs"): parts.append("[우선순위 문단]\n- " + "\n- ".join(src["priority_paragraphs"][:12]))
        if src.get("chunks"): parts.append("[본문]\n" + "\n\n---\n".join(src["chunks"]))
        input_text = "\n\n".join(parts)[:9000]

        fw.write(json.dumps({"doc_id": did, "input": input_text, "output": r.get("summary_ko","")}, ensure_ascii=False) + "\n")
        kept += 1

print(f"[OK] train_io_only.jsonl 샘플 수: {kept}")

# 문서별 요약 TXT (검토/공유용)
txt_dir = SUM_DIR/"ko_txt"; txt_dir.mkdir(parents=True, exist_ok=True)
made = 0
with resp.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        o = json.loads(line)
        (txt_dir/f"{o['doc_id']}.txt").write_text(o.get("summary_ko","").strip(), encoding="utf-8")
        made += 1
print(f"TXT 생성: {made} → {txt_dir}")  