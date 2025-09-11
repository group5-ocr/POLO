# app.py
# -*- coding: utf-8 -*-
"""
raw 폴더 → 전처리 파이프라인 실행 → 전달용 payload 구성(+ 로컬 저장)
- auto_merge 포함된 pipeline을 호출
- transport payload: meta + merged_tex + counts + 파일 경로 + (옵션) 일부 인라인 샘플
- 동시에 JSONL(.gz) 산출물은 기존처럼 out_dir에 저장
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, gzip, time, sys
from typing import Any

# 우리 모듈
from src.texprep.utils.cfg import load_cfg  # 네가 만든 cfg 로더
from src.texprep.pipeline import run_pipeline

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _read_text(p: Path, limit_chars: int | None = None) -> str:
    s = p.read_text(encoding="utf-8", errors="ignore")
    return s if not limit_chars else s[:limit_chars]

def _head_jsonl(path: Path, n: int = 3) -> list[dict[str, Any]]:
    """jsonl 또는 jsonl.gz 앞 N줄만 파싱."""
    items: list[dict[str, Any]] = []
    if path.suffix == ".gz":
        f = gzip.open(path, "rt", encoding="utf-8")
    else:
        f = open(path, "r", encoding="utf-8")
    with f:
        for i, line in enumerate(f):
            if i >= n: break
            line = line.strip()
            if not line: continue
            try:
                items.append(json.loads(line))
            except Exception:
                # 샘플이니 조용히 스킵
                pass
    return items

def _count_lines(path: Path) -> int:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return sum(1 for _ in f)
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def build_transport_payload(run: dict[str, Any], inline: bool = True, head_n: int = 3, body_chars: int = 20000) -> dict[str, Any]:
    """
    전달용 구조화. 무식하게 전부 실어 나르지 말고, 샘플·카운트·경로 위주.
    필요하면 inline=True로 일부만 동봉.
    """
    files = {k: Path(v) for k, v in (run.get("files") or {}).items()}
    out_dir = Path(run["out_dir"])

    merged_tex_path = out_dir / "merged_body.tex"
    # 
    payload: dict[str, Any] = {
        "meta": {
            "doc_id": run["doc_id"],
            "mode": run.get("mode", "one"),
            "ts": _now(),
            "main": run.get("main"),
            "merged_roots": run.get("merged_roots", []),
            "counts": {
                "chunks": run["chunks"],
                "equations": run["equations"],
                "inline_equations": run["inline_equations"],
                "assets": run["assets"],
            },
        },
        "artifacts": {
            "merged_body_tex": {
                "path": str(merged_tex_path),
                **({"preview": _read_text(merged_tex_path, body_chars)} if inline and merged_tex_path.exists() else {}),
            },
            "chunks": {
                "path": str(files.get("chunks", "")),
                "count": _count_lines(files["chunks"]) if "chunks" in files else 0,
                **({"head": _head_jsonl(files["chunks"], head_n)} if inline and "chunks" in files else {}),
            },
            "equations": {
                "path": str(files.get("equations", "")),
                "count": _count_lines(files["equations"]) if "equations" in files else 0,
                **({"head": _head_jsonl(files["equations"], head_n)} if inline and "equations" in files else {}),
            },
            "assets": {
                "path": str(files.get("assets", "")),
                "count": _count_lines(files["assets"]) if "assets" in files else 0,
                **({"head": _head_jsonl(files["assets"], head_n)} if inline and "assets" in files else {}),
            },
            "xref_mentions": {
                "path": str(files.get("xref_mentions", "")),
                "count": _count_lines(files["xref_mentions"]) if "xref_mentions" in files else 0,
                **({"head": _head_jsonl(files["xref_mentions"], head_n)} if inline and "xref_mentions" in files else {}),
            },
            "xref_edges": {
                "path": str(files.get("xref_edges", "")),
                "count": _count_lines(files["xref_edges"]) if "xref_edges" in files else 0,
                **({"head": _head_jsonl(files["xref_edges"], head_n)} if inline and "xref_edges" in files else {}),
            },
        },
    }
    return payload

def main():
    ap = argparse.ArgumentParser(description="texprep app: raw → pipeline → transport payload + files")
    ap.add_argument("--config", required=True, help="configs/default.yaml")
    # 둘 중 하나: --main(앵커 tex) 또는 --root(폴더) 제공
    ap.add_argument("--main", help="anchor .tex path")
    ap.add_argument("--root", help="folder containing raw .tex")
    ap.add_argument("--inline", action="store_true", help="transport에 미리보기/샘플 포함")
    ap.add_argument("--head", type=int, default=3, help="각 jsonl 헤드 샘플 개수")
    ap.add_argument("--body-chars", type=int, default=20000, help="merged_body.tex 미리보기 글자수")
    ap.add_argument("--save-transport", help="결과 transport.json 저장 경로(기본: out_dir/<doc_id>/transport.json)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # 앵커 결정
    anchor: str | None = None
    if args.main:
        anchor = args.main
    elif args.root:
        # root만 왔으면 대충 그 폴더 아래 아무 .tex 하나를 앵커로 사용
        # (pipeline은 auto_merge면 폴더 기준으로 병합함)
        root = Path(args.root)
        any_tex = next(root.rglob("*.tex"), None)
        if not any_tex:
            print("*.tex 없음: --root 확인해라.", file=sys.stderr); sys.exit(2)
        anchor = str(any_tex)
    else:
        print("--main 또는 --root를 줘.", file=sys.stderr); sys.exit(2)

    # 파이프라인 실행
    run = run_pipeline(cfg, main_tex=anchor, sink="json")

    # transport payload 구성
    payload = build_transport_payload(run, inline=args.inline, head_n=args.head, body_chars=args.body_chars)

    # 저장 위치
    out_dir = Path(run["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save_transport) if args.save_transport else (out_dir / "transport.json")
    save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 요약 출력
    print(json.dumps({
        "doc_id": run["doc_id"],
        "mode": run.get("mode"),
        "out_dir": run["out_dir"],
        "transport": str(save_path),
        "counts": payload["meta"]["counts"],
        "files": run.get("files", {}),
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
