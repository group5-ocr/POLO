# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import re
from typing import Any

INPUT_PAT = re.compile(r"\\(?:input|include|subfile)\{([^}]+)\}")
HAS_DOCCLASS = re.compile(r"\\documentclass\b")
HAS_BEGIN = re.compile(r"\\begin\{document\}")
HAS_TITLE = re.compile(r"\\title\{")

def _read(p: Path) -> str:
    try: return p.read_text(encoding="utf-8")
    except: return p.read_text(encoding="latin-1", errors="ignore")

def _signals(p: Path, s: str) -> dict[str, int]:
    return {
        "docclass": int(bool(HAS_DOCCLASS.search(s))),
        "begin": int(bool(HAS_BEGIN.search(s))),
        "title": int(bool(HAS_TITLE.search(s))),
        "depth": -len(p.parts),  # 얕을수록 가산
    }

def _name_score(p: Path, pos: list[str], neg: list[str]) -> int:
    name = p.name.lower()
    sc = 0
    for k in pos:
        if k in name: sc += 5
    for k in neg:
        if k in name: sc -= 4
    return sc

def _edges(root: Path) -> dict[Path, set[Path]]:
    g: dict[Path, set[Path]] = {}
    for p in root.rglob("*.tex"):
        s = _read(p)
        cur = set()
        for m in INPUT_PAT.finditer(s):
            rel = m.group(1).strip()
            cand = (p.parent / rel)
            if not cand.suffix:
                cand = cand.with_suffix(".tex")
            if cand.exists():
                cur.add(cand.resolve())
        g[p.resolve()] = cur
    return g

def select_roots(root_dir: str, cfg: dict[str, Any]) -> list[Path]:
    root = Path(root_dir).resolve()
    pos = [w.lower() for w in (cfg.get("select", {}).get("filename_weights", {}).get("positive") or [])]
    neg = [w.lower() for w in (cfg.get("select", {}).get("filename_weights", {}).get("negative") or [])]
    mode = (cfg.get("select", {}).get("mode") or "one").lower()

    g = _edges(root)
    indeg: dict[Path, int] = {u: 0 for u in g}
    for u, vs in g.items():
        for v in vs:
            indeg[v] = indeg.get(v, 0) + 1

    # 루트 후보: in-degree 0 또는 외부에서만 참조되는 최상위 .tex
    cands = [p for p, d in indeg.items() if d == 0]
    if not cands:  # 전부 참조된다면 그냥 최상위 깊이 기준
        cands = sorted(g.keys(), key=lambda p: len(p.parts))[:5]

    scored: list[tuple[int, Path]] = []
    for p in cands:
        s = _read(p)
        sig = _signals(p, s)
        score = 0
        score += 10*sig["docclass"] + 6*sig["begin"] + 3*sig["title"] + sig["depth"]
        score += _name_score(p, pos, neg)
        scored.append((score, p))
    scored.sort(key=lambda t: t[0], reverse=True)

    if mode == "one":
        return [scored[0][1]] if scored else []
    if mode == "multi":
        return [p for _, p in scored]
    if mode == "merge":
        out = [scored[0][1]] if scored else []
        # supp/appendix를 상위 N개에서 추가
        max_supp = int(cfg.get("select", {}).get("max_supp", 2))
        for _, p in scored[1:]:
            name = p.name.lower()
            if any(k in name for k in ("supp", "appendix")):
                out.append(p)
                if len(out) >= 1 + max_supp:
                    break
        return out
    return [scored[0][1]] if scored else []
