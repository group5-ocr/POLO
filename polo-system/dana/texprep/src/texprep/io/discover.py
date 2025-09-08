# src/texprep/io/discover.py
from __future__ import annotations
from pathlib import Path
import re

# % !TEX root = main.tex 같은 매직 코멘트 패턴
MAGIC_ROOT = re.compile(r"^\s*%+\s*!TEX\s+root\s*=\s*(?P<root>[^\s]+)", re.I | re.M)

def _score(path: Path, text: str) -> int:
    """
    후보 .tex 파일의 '메인일 확률'을 점수화한다.
    규칙:
      - \documentclass 있으면 +20
      - \begin{document} 있으면 +40
      - \title{}나 \author{} 있으면 +10
      - 이름이 main.tex / paper.tex / root.tex 이면 +50
      - 경로 깊이가 얕을수록(상위에 있을수록) 점수↑
    """
    name = path.name.lower()
    score = 0
    if "documentclass" in text: score += 20
    if "\\begin{document}" in text: score += 40
    if "\\title{" in text or "\\author{" in text: score += 10
    if name in {"main.tex","paper.tex","root.tex"}: score += 50
    score += max(0, 20 - len(path.parts))  # 경로 깊이 보너스
    return score

def _follow_magic_root(root: Path, p: Path, text: str) -> Path | None:
    """
    % !TEX root = some.tex 같은 코멘트가 있으면
    그걸 메인 파일로 간주해서 경로 반환.
    """
    m = MAGIC_ROOT.search(text)
    if not m: 
        return None
    tgt = (p.parent / m.group("root")).resolve()
    if tgt.exists():
        return tgt
    # 상대경로 꼬일 수 있으니 루트 기준도 시도
    alt = (root / m.group("root")).resolve()
    return alt if alt.exists() else None

def _follow_subfiles(path: Path, text: str) -> Path | None:
    """
    subfiles 패키지를 사용하는 경우
    \documentclass[main.tex]{subfiles}
    이런 식으로 메인을 가리키니까 그 경로를 반환.
    """
    m = re.search(r"\\documentclass\[(?P<main>[^]\s]+)\]\{subfiles\}", text)
    if not m:
        return None
    tgt = (path.parent / m.group("main")).resolve()
    return tgt if tgt.exists() else None

def guess_main(root_dir: str) -> str:
    """
    루트 디렉토리 밑의 .tex 파일들 중에서
    가장 '메인'일 확률이 높은 파일을 찾아서 반환한다.
    """
    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"root_dir 없음: {root}")

    # 모든 .tex 후보를 재귀적으로 수집
    cands = list(root.rglob("*.tex"))
    if not cands:
        raise FileNotFoundError(f".tex 못 찾음: {root}")

    # 1) 매직 코멘트나 subfiles 패키지로 메인 직접 지정된 경우 우선
    specials = []
    for p in cands:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if m := _follow_magic_root(root, p, txt):
            return str(m)  # 바로 메인 확정
        if s := _follow_subfiles(p, txt):
            specials.append(s)
    if specials:
        # 여러 개 나오면 점수 높은 것 하나 고른다
        best = max(
            set(specials),
            key=lambda q: _score(q, q.read_text(encoding="utf-8", errors="ignore"))
        )
        return str(best)

    # 2) 위 케이스 없으면 점수 기반으로 최고점 선택
    best = max(
        cands,
        key=lambda q: _score(q, q.read_text(encoding="utf-8", errors="ignore"))
    )
    return str(best)
