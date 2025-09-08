# src/texprep/pipeline.py
# -*- coding: utf-8 -*-
"""
TeX 전처리 파이프라인
- main.tex 추정/결정 -> \input 확장 -> 코멘트/잡동사니 제거
- figure/table 추출(플레이스홀더 치환)
- 수식 추출(디스플레이 + 라인 전용 인라인 승격)
- xref(라벨 참조/플레이스홀더 인덱스) 수집
- 청킹(JSONL용)
- 심볼 테이블/프리림 요약(+ 원하면 Qwen 입력 페이로드)
- 결과를 out_dir/<doc_id>/ 에 JSONL로 저장

cfg 예시 키:
  root_dir: "./data/raw"
  out_dir: "./data/out"
  drop_envs: ["tikzpicture","minted","lstlisting","verbatim","Verbatim"]
  chunk: { max_chars: 3800, overlap: 1 }
  export: { compress: true }           # 선택
  math: { prelim_chars: 1200, question: "" }  # question은 선택
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

# IO / TeX
from texprep.io.discover import guess_main
from texprep.tex.expander import expand_file
from texprep.tex.strip import clean_text
from texprep.tex.blocks import extract_assets
from texprep.tex.math import extract_math_with_promotion
from texprep.tex.chunk import choose_chunking
from texprep.tex.xref import build_mentions_map, propose_xref_edges, attach_mentions_to_assets

# Enrich
from texprep.enrich.symbol_table import (
    build_symbol_table,
    summarize_preliminaries,
    build_qwen_math_payload,
)

# Export
from texprep.exporters.jsonl import dump_all


def _doc_id_from_path(p: Path) -> str:
    """파일명 기준 간단 doc_id."""
    return p.stem.replace(" ", "_")


def run_pipeline(cfg: dict[str, Any], main_tex: str | None = None, *, sink: str = "json") -> dict[str, Any]:
    """
    파이프라인 실행. 반환: 요약 메타와 출력 파일 경로들.
    sink: "json" | "pg" | "both"  (지금은 json만 처리, pg는 TODO)
    """
    root_dir = Path(cfg.get("root_dir", ".")).resolve()
    out_root = Path(cfg.get("out_dir", "./data/out")).resolve()
    drop_envs = cfg.get("drop_envs") or ["tikzpicture", "minted", "lstlisting", "verbatim", "Verbatim"]
    chunk_cfg = cfg.get("chunk", {}) or {}
    max_chars = int(chunk_cfg.get("max_chars", 3800))
    overlap = int(chunk_cfg.get("overlap", 1))
    compress = bool(cfg.get("export", {}).get("compress", False))
    math_cfg = cfg.get("math", {}) or {}
    prelim_chars = int(math_cfg.get("prelim_chars", 1200))
    question = (math_cfg.get("question") or "").strip()

    # 1) main.tex 결정
    main_path = Path(main_tex).resolve() if main_tex else Path(guess_main(str(root_dir))).resolve()
    if not main_path.exists():
        raise FileNotFoundError(f"main tex 없음: {main_path}")
    doc_id = _doc_id_from_path(main_path)
    out_dir = out_root / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) \input 확장
    expanded_text, deps = expand_file(str(main_path))

    # 3) 정리(코멘트/불필요 환경 제거)
    stripped = clean_text(expanded_text, drop_env_list=drop_envs, also_drop_inline_todos=True)

    # 4) figure/table 블록 추출(본문 치환)
    text_no_floats, assets = extract_assets(stripped)

    # 5) 수식 추출(디스플레이 + 라인 전용 인라인 승격)
    body, equations, inline_equations = extract_math_with_promotion(text_no_floats)

    # 6) xref: 라벨 언급/에지
    xref_mentions = build_mentions_map(body)
    xref_edges = propose_xref_edges(body)
    assets_with_mentions = attach_mentions_to_assets(assets, xref_mentions)

    # 7) 청킹
    chunks = choose_chunking(body, max_chars=max_chars, overlap=overlap, mode="paragraph", prefer_section_boundary=True)

    # 8) 심볼 테이블 / 프리림 요약
    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    symtab = build_symbol_table(paragraphs, equations)
    prelim_summary = summarize_preliminaries(paragraphs, max_chars=prelim_chars)

    # 9) (선택) 수학 LLM 전달 페이로드(메시지 아님)
    payloads: dict[str, dict[str, Any]] | None = None
    if question:
        qwen_payload = build_qwen_math_payload(
            question=question,
            equations=equations,
            symbol_table=symtab,
            prelim_summary=prelim_summary,
        )
        payloads = {"qwen_payload": qwen_payload}

    # 10) Export (JSONL)
    files = {}
    if sink in ("json", "both"):
        files = dump_all(
            out_dir=str(out_dir),
            doc_id=doc_id,
            chunks=chunks,
            display_equations=equations,
            inline_equations=inline_equations,
            assets=assets_with_mentions,
            symbol_table=symtab,
            xref_mentions=xref_mentions,
            xref_edges=xref_edges,
            payloads=payloads,
            compress=compress,
        )

    # 11) PG는 나중에
    if sink in ("pg", "both"):
        # TODO: PostgreSQL 적재. 지금은 자리표시자만.
        # raise NotImplementedError("Postgres sink는 나중에 붙여라. json으로 충분히 디버깅하고.")
        pass

    # 요약 리포트
    return {
        "doc_id": doc_id,
        "main": str(main_path),
        "deps_count": len(deps),
        "chars": len(body),
        "chunks": len(chunks),
        "equations": len(equations),
        "inline_equations": len(inline_equations),
        "assets": len(assets),
        "files": {k: str(v) for k, v in files.items()},
        "out_dir": str(out_dir),
    }
