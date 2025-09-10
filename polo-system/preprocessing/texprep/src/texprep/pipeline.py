# src/texprep/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Any

from texprep.io.discover import guess_main
from texprep.io.auto_merge import auto_merge_corpus
from texprep.tex.expander import expand_file
from texprep.tex.strip import preclean_for_body, clean_text, drop_after_markers
from texprep.tex.blocks import extract_assets
from texprep.tex.math import extract_math_with_promotion
from texprep.tex.chunk import choose_chunking
from texprep.tex.xref import build_mentions_map, propose_xref_edges, attach_mentions_to_assets
from texprep.exporters.jsonl import dump_all

def _doc_id_from_path(p: Path) -> str:
    return p.stem.replace(" ", "_")

def run_pipeline(cfg: dict[str, Any], main_tex: str | None = None, *, sink: str = "json") -> dict[str, Any]:
    root_dir  = Path(cfg.get("root_dir", ".")).resolve()
    out_root  = Path(cfg.get("out_dir", "./data/out")).resolve()
    drop_envs_base = cfg.get("drop_envs") or ["tikzpicture","minted","lstlisting","verbatim","Verbatim"]
    drop_envs_full = sorted(set([*drop_envs_base, "framed", "mdframed", "tcolorbox"]))

    chunk_cfg   = cfg.get("chunk", {}) or {}
    max_chars   = int(chunk_cfg.get("max_chars", 3800))
    overlap     = int(chunk_cfg.get("overlap", 1))
    compress    = bool(cfg.get("export", {}).get("compress", False))
    cut_patterns = (cfg.get("filters", {}) or {}).get("cut_after", [
        r"\\appendix\b",
        r"\\section\*?\{Generations\}",
        r"\\section\*?\{Bias\}",
    ])
    select_mode = (cfg.get("select", {}) or {}).get("mode", "one").lower()

    # 1) 앵커 결정
    main_path = Path(main_tex).resolve() if main_tex else Path(guess_main(str(root_dir))).resolve()
    if not main_path.exists():
        raise FileNotFoundError(f"main tex 없음: {main_path}")

    doc_id = (main_path.parent.name if select_mode == "auto_merge" else main_path.stem).replace(" ", "_")
    out_dir = out_root / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) 소스 본문 만들기
    merged_roots: list[str] = []
    if select_mode == "auto_merge":
        merged      = auto_merge_corpus(str(main_path.parent), drop_envs_full)
        source_text = merged["text"]
        merged_roots = merged.get("roots", [])
        deps = merged_roots
    else:
        expanded_text, deps = expand_file(str(main_path))
        body_only = preclean_for_body(expanded_text)
        body_only = drop_after_markers(body_only, cut_patterns)
        source_text = clean_text(body_only, drop_env_list=tuple(drop_envs_full), also_drop_inline_todos=True)

    # 2.5) 병합 본문 저장(수학 LLM용 원문 LaTeX)
    merged_tex_path = out_dir / "merged_body.tex"
    merged_tex_path.write_text(source_text, encoding="utf-8")

    # 3) 그림/표만 플레이스홀더로 치환. 수식은 그대로 남김.
    text_no_floats, assets = extract_assets(source_text)

    # 4) 수식 리스트만 추출(본문은 변경하지 않음)
    _, equations, inline_equations = extract_math_with_promotion(text_no_floats)

    # 5) xref는 LaTeX 포함 텍스트에서
    text_for_chunks = text_no_floats  # ← 청킹 소스: 수식 포함
    xref_mentions = build_mentions_map(text_for_chunks)
    xref_edges    = propose_xref_edges(text_for_chunks)
    assets_with_mentions = attach_mentions_to_assets(assets, xref_mentions)

    # 6) 청킹(LaTeX 포함)
    chunks = choose_chunking(
        text_for_chunks,
        max_chars=max_chars,
        overlap=overlap,
        mode="paragraph",
        prefer_section_boundary=True,
    )

    # 7) Export(JSONL). 심볼/페이로드 없음.
    files = {}
    if sink in ("json", "both"):
        files = dump_all(
            out_dir=str(out_dir),
            doc_id=doc_id,
            chunks=chunks,
            display_equations=equations,
            inline_equations=inline_equations,
            assets=assets_with_mentions,
            symbol_table=None,      # exporter가 None 허용해야 함
            xref_mentions=xref_mentions,
            xref_edges=xref_edges,
            payloads=None,
            compress=compress,
        )
        files["merged_body_tex"] = str(merged_tex_path)

    if sink in ("pg", "both"):
        pass  # TODO

    return {
        "doc_id": doc_id,
        "mode": select_mode,
        "main": str(main_path),
        "merged_roots": merged_roots,
        "deps_count": len(deps),
        "chars": len(text_for_chunks),   # ← 청크 소스 기준
        "chunks": len(chunks),
        "equations": len(equations),
        "inline_equations": len(inline_equations),
        "assets": len(assets),
        "files": {k: str(v) for k, v in files.items()},
        "out_dir": str(out_dir),
    }
