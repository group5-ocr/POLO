# src/texprep/exporters/jsonl.py
# -*- coding: utf-8 -*-
"""
JSONL Exporter (modern typing)
- 한 레코드 = 한 줄 JSON
- 공통 메타(doc_id, schema_version, ts) 자동 주입
- .jsonl 또는 .jsonl.gz로 저장
- chunks / equations / inline_equations / assets / symbol_table / xref_mentions / xref_edges / payload 지원
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
import gzip
import io
import json
import time
import os

SCHEMA_VERSION = "0.1"

# -----------------------
# 기본 유틸
# -----------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _open_text(path: Path) -> io.TextIOBase:
    if str(path).endswith(".gz"):
        # gzip 텍스트 모드
        return io.TextIOWrapper(gzip.open(path, "wb"), encoding="utf-8", newline="\n")
    return open(path, "w", encoding="utf-8", newline="\n")

def _write_jsonl(path: Path, records: Iterable[dict[str, Any]], meta: dict[str, Any]) -> int:
    """레코드들에 meta를 병합해 JSONL로 쓴다. 반환: 라인 수"""
    _ensure_dir(path)
    n = 0
    with _open_text(path) as f:
        for rec in records:
            row = {**meta, **rec}
            # ensure_ascii=False로 유니코드 그대로 저장
            s = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            f.write(s + "\n")
            n += 1
    return n

def _meta(doc_id: str | int | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {"doc_id": doc_id, "schema_version": SCHEMA_VERSION, "ts": _now_iso()}
    return {**base, **(extra or {})}

# -----------------------
# to-records 어댑터들
# -----------------------

def _rec_chunks(chunks: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for ch in chunks:
        yield {
            "type": "chunk",
            "chunk_index": ch.get("chunk_index"),
            "text": ch.get("text", ""),
            "char_count": ch.get("char_count", 0),
            "start_para": ch.get("start_para"),
            "end_para": ch.get("end_para"),
            "start_sent": ch.get("start_sent"),
            "end_sent": ch.get("end_sent"),
            "placeholder": ch.get("placeholder", {}),
        }

def _rec_equations(eqs: list[dict[str, Any]], eq_type: str) -> Iterable[dict[str, Any]]:
    # eq_type: "display" | "inline"
    for e in eqs:
        yield {
            "type": "equation",
            "eq_kind": eq_type,
            "id": e.get("id"),
            "env": e.get("env"),
            "tex": e.get("tex"),
            "display": bool(e.get("display", eq_type == "display")),
            "promoted": bool(e.get("promoted", False)),
            "location": e.get("location"),
        }

def _rec_assets(assets: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for a in assets:
        yield {
            "type": "asset",
            "id": a.get("id"),
            "kind": a.get("kind"),         # figure | table
            "env": a.get("env"),
            "label": a.get("label"),
            "caption": a.get("caption"),
            "graphics": a.get("graphics", []),
            "children": a.get("children", []),
            "mentions": a.get("mentions", []),
        }

def _rec_symbol_table(symtab: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for r in symtab:
        yield {
            "type": "symbol",
            "symbol": r.get("symbol"),
            "defs": r.get("defs", []),     # [{text, para_index, sent_index, score}]
            "freq_eq": r.get("freq_eq", 0),
            "first_eq_id": r.get("first_eq_id"),
            "eq_ids": r.get("eq_ids", []),
            "tags": r.get("tags", []),
        }

def _rec_xref_mentions(mentions: dict[str, list[dict[str, Any]]]) -> Iterable[dict[str, Any]]:
    # label -> list[mention]
    for label, lst in mentions.items():
        for m in lst:
            yield {
                "type": "xref_mention",
                "label": label,
                "para_index": m.get("para_index"),
                "sent_index": m.get("sent_index"),
                "cmd": m.get("cmd"),
                "text": m.get("text"),
            }

def _rec_xref_edges(edges: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for e in edges:
        yield {
            "type": "xref_edge",
            "src_para_index": e.get("src_para_index"),
            "cmd": e.get("cmd"),
            "target_label": e.get("target_label"),
        }

def _rec_payload_one(name: str, obj: dict[str, Any]) -> Iterable[dict[str, Any]]:
    # 큰 덩어리 하나를 한 줄에
    yield {"type": name, **obj}

# -----------------------
# 퍼사드 API
# -----------------------

def dump_chunks(out_dir: str | os.PathLike[str], doc_id: str | int | None, chunks: list[dict[str, Any]], compress: bool = False) -> Path:
    path = Path(out_dir) / ("chunks.jsonl.gz" if compress else "chunks.jsonl")
    _write_jsonl(path, _rec_chunks(chunks), _meta(doc_id))
    return path

def dump_equations(out_dir: str | os.PathLike[str], doc_id: str | int | None,
                   display_eqs: list[dict[str, Any]], inline_eqs: list[dict[str, Any]] | None = None,
                   compress: bool = False) -> Path:
    path = Path(out_dir) / ("equations.jsonl.gz" if compress else "equations.jsonl")
    records = list(_rec_equations(display_eqs, "display"))
    if inline_eqs:
        records += list(_rec_equations(inline_eqs, "inline"))
    _write_jsonl(path, records, _meta(doc_id))
    return path

def dump_assets(out_dir: str | os.PathLike[str], doc_id: str | int | None, assets: list[dict[str, Any]], compress: bool = False) -> Path:
    path = Path(out_dir) / ("assets.jsonl.gz" if compress else "assets.jsonl")
    _write_jsonl(path, _rec_assets(assets), _meta(doc_id))
    return path

def dump_symbol_table(out_dir: str | os.PathLike[str], doc_id: str | int | None, symtab: list[dict[str, Any]], compress: bool = False) -> Path:
    path = Path(out_dir) / ("symbols.jsonl.gz" if compress else "symbols.jsonl")
    _write_jsonl(path, _rec_symbol_table(symtab), _meta(doc_id))
    return path

def dump_xrefs(out_dir: str | os.PathLike[str], doc_id: str | int | None,
               mentions: dict[str, list[dict[str, Any]]] | None = None,
               edges: list[dict[str, Any]] | None = None,
               compress: bool = False) -> list[Path]:
    paths: list[Path] = []
    if mentions is not None:
        p = Path(out_dir) / ("xref_mentions.jsonl.gz" if compress else "xref_mentions.jsonl")
        _write_jsonl(p, _rec_xref_mentions(mentions), _meta(doc_id))
        paths.append(p)
    if edges is not None:
        p = Path(out_dir) / ("xref_edges.jsonl.gz" if compress else "xref_edges.jsonl")
        _write_jsonl(p, _rec_xref_edges(edges), _meta(doc_id))
        paths.append(p)
    return paths

def dump_payload(out_dir: str | os.PathLike[str], doc_id: str | int | None,
                 name: str, obj: dict[str, Any], compress: bool = False) -> Path:
    """
    거대한 덩어리 하나를 한 줄로: 예) qwen_payload, prelim_summary 등
    name은 레코드 type으로 들어간다.
    """
    path = Path(out_dir) / (f"{name}.jsonl.gz" if compress else f"{name}.jsonl")
    _write_jsonl(path, _rec_payload_one(name, obj), _meta(doc_id))
    return path

def dump_all(
    out_dir: str | os.PathLike[str],
    doc_id: str | int | None,
    *,
    chunks: list[dict[str, Any]] | None = None,
    display_equations: list[dict[str, Any]] | None = None,
    inline_equations: list[dict[str, Any]] | None = None,
    assets: list[dict[str, Any]] | None = None,
    symbol_table: list[dict[str, Any]] | None = None,
    xref_mentions: dict[str, list[dict[str, Any]]] | None = None,
    xref_edges: list[dict[str, Any]] | None = None,
    payloads: dict[str, dict[str, Any]] | None = None,  # {"qwen_payload": {...}, ...}
    compress: bool = False,
) -> dict[str, Path]:
    """
    필요한 것만 골라 한 번에 덤프. 반환: 파일 경로 맵.
    """
    out: dict[str, Path] = {}
    if chunks is not None:
        out["chunks"] = dump_chunks(out_dir, doc_id, chunks, compress=compress)
    if display_equations is not None or inline_equations is not None:
        out["equations"] = dump_equations(out_dir, doc_id, display_equations or [], inline_equations, compress=compress)
    if assets is not None:
        out["assets"] = dump_assets(out_dir, doc_id, assets, compress=compress)
    if symbol_table is not None:
        out["symbols"] = dump_symbol_table(out_dir, doc_id, symbol_table, compress=compress)
    if xref_mentions is not None or xref_edges is not None:
        paths = dump_xrefs(out_dir, doc_id, mentions=xref_mentions, edges=xref_edges, compress=compress)
        if xref_mentions is not None: out["xref_mentions"] = paths[0]
        if xref_edges is not None: out["xref_edges"] = paths[-1 if len(paths) > 1 else 0]
    if payloads:
        for name, obj in payloads.items():
            out[name] = dump_payload(out_dir, doc_id, name, obj, compress=compress)
    return out
