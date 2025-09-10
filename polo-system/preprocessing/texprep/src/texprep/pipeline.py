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
from texprep.io.auto_merge import auto_merge_corpus
from texprep.tex.expander import expand_file
from texprep.tex.strip import preclean_for_body, clean_text, drop_after_markers
from texprep.tex.blocks import extract_assets
from texprep.tex.math import extract_math_with_promotion
from texprep.tex.chunk import choose_chunking
from texprep.tex.xref import build_mentions_map, propose_xref_edges, attach_mentions_to_assets

# Enrich -> 계획 변경으로 사용하지 않음. 로그로 남겨둠
# from texprep.enrich.symbol_table import (
#     build_symbol_table,
#     summarize_preliminaries,
#     build_qwen_math_payload,
# )

# Export
from texprep.exporters.jsonl import dump_all


def _doc_id_from_path(p: Path) -> str:
    """파일명 기준 간단 doc_id."""
    return p.stem.replace(" ", "_")

def run_pipeline(cfg: dict[str, Any], main_tex: str | None = None, *, sink: str = "json") -> dict[str, Any]:
    root_dir = Path(cfg.get("root_dir", ".")).resolve()
    out_root = Path(cfg.get("out_dir", "./data/out")).resolve()
    # drop_envs 기본 + 데모박스류 추가
    drop_envs_base = cfg.get("drop_envs") or ["tikzpicture","minted","lstlisting","verbatim","Verbatim"]
    drop_envs_full = sorted(set([*drop_envs_base, "framed", "mdframed", "tcolorbox"]))

    chunk_cfg = cfg.get("chunk", {}) or {}
    max_chars = int(chunk_cfg.get("max_chars", 3800))
    overlap = int(chunk_cfg.get("overlap", 1))
    compress = bool(cfg.get("export", {}).get("compress", False))
    math_cfg = cfg.get("math", {}) or {}
    prelim_chars = int(math_cfg.get("prelim_chars", 1200))
    question = (math_cfg.get("question") or "").strip()
    cut_patterns = (cfg.get("filters", {}) or {}).get("cut_after", [
        r"\\appendix\b",
        r"\\section\*?\{Generations\}",
        r"\\section\*?\{Bias\}",
    ])
    select_mode = (cfg.get("select", {}) or {}).get("mode", "one").lower()

    # 1) main.tex 결정(앵커만 필요)
    main_path = Path(main_tex).resolve() if main_tex else Path(guess_main(str(root_dir))).resolve()
    if not main_path.exists():
        raise FileNotFoundError(f"main tex 없음: {main_path}")

    # doc_id: auto_merge면 폴더명, 아니면 파일명
    doc_id = (main_path.parent.name if select_mode == "auto_merge" else main_path.stem).replace(" ", "_")
    out_dir = out_root / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) 소스 본문 만들기
    merged_roots: list[str] = []
    if select_mode == "auto_merge":
        # 여러 루트 후보 펼쳐서 본문 병합(중복 제거까지 완료)
        merged = auto_merge_corpus(str(main_path.parent), drop_envs_full)
        source_text = merged["text"]
        merged_roots = merged.get("roots", [])
        deps = merged_roots  # 리포트용
    else:
        # 단일 루트 확장 + 프리앰블 제거 + 컷 + 클린
        expanded_text, deps = expand_file(str(main_path))
        body_only = preclean_for_body(expanded_text)
        body_only = drop_after_markers(body_only, cut_patterns)
        source_text = clean_text(body_only, drop_env_list=tuple(drop_envs_full), also_drop_inline_todos=True)

    # 3) figure/table 추출(본문 치환)
    text_no_floats, assets = extract_assets(source_text)

    # 4) 수식 추출
    body, equations, inline_equations = extract_math_with_promotion(text_no_floats)

    # 5) xref
    xref_mentions = build_mentions_map(body)
    xref_edges = propose_xref_edges(body)
    assets_with_mentions = attach_mentions_to_assets(assets, xref_mentions)

    # 6) 청킹
    chunks = choose_chunking(body, max_chars=max_chars, overlap=overlap, mode="paragraph", prefer_section_boundary=True)

    # 7) 심볼/프리림
    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    symtab = build_symbol_table(paragraphs, equations)
    prelim_summary = summarize_preliminaries(paragraphs, max_chars=prelim_chars)

    # 8) 선택 페이로드
    payloads = None
    if question:
        payloads = {"qwen_payload": build_qwen_math_payload(
            question=question, equations=equations, symbol_table=symtab, prelim_summary=prelim_summary
        )}

    # 9) Export
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

    if sink in ("pg", "both"):
        pass  # TODO

    return {
        "doc_id": doc_id,
        "mode": select_mode,
        "main": str(main_path),
        "merged_roots": merged_roots,
        "deps_count": len(deps),
        "chars": len(body),
        "chunks": len(chunks),
        "equations": len(equations),
        "inline_equations": len(inline_equations),
        "assets": len(assets),
        "files": {k: str(v) for k, v in files.items()},
        "out_dir": str(out_dir),
    }