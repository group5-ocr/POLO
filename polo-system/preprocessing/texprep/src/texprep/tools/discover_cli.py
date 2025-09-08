# src/texprep/tools/discover_cli.py
from __future__ import annotations
import argparse
from pathlib import Path
import re
from typing import List, Tuple

MAGIC_ROOT = re.compile(r"^\s*%+\s*!TEX\s+root\s*=\s*(?P<root>[^\s]+)", re.I | re.M)
SUBFILES  = re.compile(r"\\documentclass\[(?P<main>[^]\s]+)\]\{subfiles\}")

NAME_HINTS = {"main.tex","paper.tex","root.tex","ms.tex"}

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def signals(path: Path, text: str) -> dict:
    return {
        "documentclass": ("\\documentclass" in text),
        "begin_document": ("\\begin{document}" in text),
        "title_or_author": ("\\title{" in text or "\\author{" in text),
        "name_hint": (path.name.lower() in NAME_HINTS),
        "depth": len(path.parts),
        "magic_root": bool(MAGIC_ROOT.search(text)),
        "subfiles": bool(SUBFILES.search(text)),
    }

def score_from_signals(sig: dict) -> Tuple[int, dict]:
    score = 0
    if sig["documentclass"]:   score += 20
    if sig["begin_document"]:  score += 40
    if sig["title_or_author"]: score += 10
    if sig["name_hint"]:       score += 50
    score += max(0, 20 - sig["depth"])  # 얕을수록 가산
    return score, sig

def follow_magic_root(root: Path, p: Path, text: str) -> Path | None:
    m = MAGIC_ROOT.search(text)
    if not m: return None
    cand = (p.parent / m.group("root")).resolve()
    if cand.exists(): return cand
    alt = (root / m.group("root")).resolve()
    return alt if alt.exists() else None

def follow_subfiles(p: Path, text: str) -> Path | None:
    m = SUBFILES.search(text)
    if not m: return None
    cand = (p.parent / m.group("main")).resolve()
    return cand if cand.exists() else None

def rank_candidates(root_dir: str) -> Tuple[Path, List[Tuple[int, Path, dict]]]:
    root = Path(root_dir).resolve()
    cands = list(root.rglob("*.tex"))
    if not cands:
        raise FileNotFoundError(f".tex 없음: {root}")
    # 명시 지시 먼저 확인
    specials = []
    for p in cands:
        t = read_text(p)
        if m := follow_magic_root(root, p, t):
            best_text = read_text(m)
            s, sig = score_from_signals(signals(m, best_text))
            return m, [(s, m, sig)]
        if s := follow_subfiles(p, t):
            specials.append(s)
    if specials:
        scored = []
        seen = set()
        for m in specials:
            if m in seen: continue
            seen.add(m)
            st = read_text(m)
            s, sig = score_from_signals(signals(m, st))
            scored.append((s, m, sig))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1], scored

    # 점수 기반
    scored = []
    for p in cands:
        t = read_text(p)
        s, sig = score_from_signals(signals(p, t))
        scored.append((s, p, sig))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], scored

def main():
    ap = argparse.ArgumentParser(description="Guess main .tex and explain why.")
    ap.add_argument("--root", required=True, help="논문 소스 루트 디렉토리")
    ap.add_argument("--top", type=int, default=5, help="상위 N개 후보 표시")
    ap.add_argument("--explain", action="store_true", help="점수 근거까지 출력")
    args = ap.parse_args()

    best, scored = rank_candidates(args.root)

    print(f"[BEST] {best}")
    topn = scored[:args.top]
    width = max(len(str(p)) for _, p, _ in topn)
    print("\n[TOP CANDIDATES]")
    print(f"{'score':>5}  path")
    for s, p, _ in topn:
        print(f"{s:5d}  {p}")

    if args.explain:
        print("\n[EXPLAIN]")
        for s, p, sig in topn:
            print(f"- {p}")
            flags = []
            for k in ("documentclass","begin_document","title_or_author","name_hint","subfiles","magic_root"):
                if sig.get(k): flags.append(k)
            print(f"  score={s}, signals={','.join(flags) or 'none'}, depth={sig['depth']}")

if __name__ == "__main__":
    main()
