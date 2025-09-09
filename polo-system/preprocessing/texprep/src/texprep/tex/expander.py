# -*- coding: utf-8 -*-
"""
TeX 입력 확장기 (modern typing)
- \input, \include, \InputIfFileExists 재귀 확장
- 상대경로는 호출 파일 기준
- 사이클/깊이 제한, verbatim류 보호 마스킹
- 포함된 파일 목록(deps) 반환
"""
from __future__ import annotations
from pathlib import Path
import re
from typing import Iterable

INPUT_CMDS = (r"\input", r"\include", r"\InputIfFileExists")
TEX_EXTS = (".tex",)
PROTECT_ENVS = ("verbatim", "Verbatim", "lstlisting", "lstlisting*", "minted", "tikzpicture")
_PROTECT_TOKEN = "§§PROTECT_BLOCK_{}§§"

def _read_text(p: Path) -> str:
    try:
        s = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        s = p.read_text(encoding="latin-1", errors="ignore")
    return s.replace("\r\n", "\n").replace("\r", "\n")

def _mask_protected_blocks(text: str) -> tuple[str, list[tuple[str, str]]]:
    """verbatim류 환경은 통째로 토큰으로 치환했다가 마지막에 복구."""
    masked: list[tuple[str, str]] = []
    out = text
    for env in PROTECT_ENVS:
        pat = re.compile(rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}", re.S)
        while True:
            m = pat.search(out)
            if not m:
                break
            block = m.group(0)
            tok = _PROTECT_TOKEN.format(len(masked))
            masked.append((tok, block))
            out = out[:m.start()] + tok + out[m.end():]
    return out, masked

def _unmask_protected_blocks(text: str, masked: list[tuple[str, str]]) -> str:
    out = text
    for tok, block in masked:
        out = out.replace(tok, block)
    return out

def _resolve_candidates(base_dir: Path, name: str) -> list[Path]:
    """
    \input{foo} → foo, foo.tex 순서로 시도.
    절대/상대 모두 지원. base_dir는 호출 파일 디렉토리.
    """
    name = name.strip()
    cand = Path(name)
    paths: list[Path] = []
    if cand.is_absolute():
        paths.append(cand)
        for ext in TEX_EXTS:
            if not cand.suffix:
                paths.append(cand.with_suffix(ext))
    else:
        p = base_dir / cand
        paths.append(p)
        for ext in TEX_EXTS:
            if not p.suffix:
                paths.append(p.with_suffix(ext))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.exists() and rp.is_file():
            out.append(rp)
    return out

def _expand_once(text: str, cur_dir: Path, visited: set[Path], deps: list[Path]) -> tuple[str, bool]:
    """
    한 패스 치환. 변경 여부 반환.
    \InputIfFileExists{a}{then}{else} 지원(then만 재귀 확장).
    """
    changed = False

    # \input{...} / \include{...}
    pat_simple = re.compile(rf"(?:{'|'.join(map(re.escape, INPUT_CMDS[:2]))})\{{([^}}]+)\}}")
    def repl_simple(m: re.Match) -> str:
        nonlocal changed
        target = m.group(1)
        for rp in _resolve_candidates(cur_dir, target):
            if rp in visited:
                return ""  # 사이클 방지
            visited.add(rp)
            changed = True
            content = _read_text(rp)
            expanded, _ = expand_string(content, rp.parent, visited, deps)
            deps.append(rp)
            return expanded
        return m.group(0)  # 못 찾으면 원문 유지

    text = pat_simple.sub(repl_simple, text)

    # \InputIfFileExists{file}{then}{else}
    pat_if = re.compile(r"\\InputIfFileExists\{([^}]+)\}\{([^}]*)\}\{([^}]*)\}", re.S)
    def repl_if(m: re.Match) -> str:
        nonlocal changed
        fname, then_part, else_part = m.group(1), m.group(2), m.group(3)
        cands = _resolve_candidates(cur_dir, fname)
        if cands:
            changed = True
            expanded_then, _ = expand_string(then_part, cur_dir, visited, deps)
            return expanded_then
        return else_part

    text = pat_if.sub(repl_if, text)
    return text, changed

def expand_string(
    text: str,
    base_dir: Path,
    visited: set[Path] | None = None,
    deps: list[Path] | None = None,
    max_depth: int = 20,
) -> tuple[str, list[Path]]:
    """문자열 입력을 재귀 확장."""
    visited = visited or set()
    deps = deps or []
    masked_text, masked = _mask_protected_blocks(text)

    depth = 0
    cur = masked_text
    while depth < max_depth:
        cur, changed = _expand_once(cur, base_dir, visited, deps)
        if not changed:
            break
        depth += 1

    if depth >= max_depth:
        cur = "% WARNING: max expansion depth reached\n" + cur

    cur = _unmask_protected_blocks(cur, masked)
    return cur, deps

def expand_file(main_tex_path: str, max_depth: int = 20) -> tuple[str, list[Path]]:
    """파일 경로 입력을 재귀 확장."""
    main = Path(main_tex_path).resolve()
    if not main.exists():
        raise FileNotFoundError(f"파일 없음: {main}")
    text = _read_text(main)
    expanded, deps = expand_string(text, main.parent, visited={main}, deps=[main], max_depth=max_depth)
    return expanded, deps
