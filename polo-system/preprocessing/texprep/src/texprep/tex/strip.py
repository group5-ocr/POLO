# src/texprep/tex/strip.py
# -*- coding: utf-8 -*-
"""
코멘트/불필요 환경 제거 + 프리앰블 정리 (modern typing)

기능:
- 개행 통일
- 본문만 추출: \begin{document} ... \end{document}
- 프리앰블/설정 블록 제거: \lstset, \lstdefinelanguage, \makeatletter ... \makeatother
- 문서 내 잡명령 제거: \maketitle, \looseness, \vspace, \phantom 등
- 코멘트(%) 제거. 단 \%, \verb, verbatim 계열 환경 안의 %는 보존
- 지정 환경 통째 제거: tikzpicture, minted, lstlisting, Verbatim 등
- 인라인 잡동사니 명령 제거: \todo{..}, \marginpar{..}, \iffalse ... \fi

권장 사용 순서:
  preclean_for_body()  ->  clean_text()
"""

from __future__ import annotations
import re
from typing import Iterable

__all__ = [
    "normalize_newlines",
    "extract_document_body",
    "drop_setup_blocks",
    "drop_noise_commands",
    "strip_comments",
    "drop_envs",
    "drop_inline_commands",
    "clean_text",
    "preclean_for_body",
]

# 보호할 환경(여기 안은 그대로 보존)
PROTECT_ENVS_DEFAULT = (
    "verbatim", "Verbatim", "lstlisting", "lstlisting*",
    "minted", "tikzpicture",
)

# \verb 인라인 보호
VERB_INLINE_RE = re.compile(
    r"""\\verb\*?(?P<delim>[^A-Za-z0-9\s])(?P<body>.*?)(?P=delim)"""
)
_PROTECT_BLOCK = "§§PROTECT_BLOCK_{}§§"
_PROTECT_VERB  = "§§PROTECT_VERB_{}§§"

# -------- 기본 유틸 --------

def normalize_newlines(text: str) -> str:
    """윈도우/맥 개행을 LF로 통일."""
    return text.replace("\r\n", "\n").replace("\r", "\n")

# -------- 프리앰블/본문 정리 --------

def extract_document_body(text: str) -> str:
    """\\begin{document} .. \\end{document} 사이만 추출. 없으면 원문 유지."""
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", text, re.S)
    return m.group(1) if m else text

def drop_setup_blocks(text: str) -> str:
    """프리앰블 설정 블록 제거: lst 정의/설정, makeatletter 블록."""
    patterns = [
        r"\\lstdefinelanguage\{[^}]+\}\s*\{.*?\}",
        r"\\lstset\{.*?\}",
        r"\\makeatletter.*?\\makeatother",
    ]
    out = text
    for pat in patterns:
        out = re.sub(pat, "", out, flags=re.S)
    return out

def drop_noise_commands(text: str) -> str:
    """본문 안의 시각·레이아웃 보조 명령 제거."""
    names = ["maketitle", "looseness", "vspace", "phantom"]
    out = text
    for n in names:
        out = re.sub(rf"\\{n}\*?(?:\[[^\]]*\])?(?:\{{[^{{}}]*\}})?", "", out)
    return out

# -------- 코멘트/환경/잡명령 제거 --------

def _mask_inline_verbs(text: str) -> tuple[str, list[tuple[str, str]]]:
    """\\verb 구간은 %가 나와도 텍스트. 토큰으로 보호."""
    out = text
    masked: list[tuple[str, str]] = []
    i = 0
    while True:
        m = VERB_INLINE_RE.search(out)
        if not m:
            break
        block = m.group(0)
        tok = _PROTECT_VERB.format(i)
        masked.append((tok, block))
        out = out[:m.start()] + tok + out[m.end():]
        i += 1
    return out, masked

def _mask_protect_envs(text: str, envs: Iterable[str]) -> tuple[str, list[tuple[str, str]]]:
    """verbatim류 환경 전체 보호 마스킹."""
    out = text
    masked: list[tuple[str, str]] = []
    for env in envs:
        pat = re.compile(rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}", re.S)
        while True:
            m = pat.search(out)
            if not m:
                break
            block = m.group(0)
            tok = _PROTECT_BLOCK.format(len(masked))
            masked.append((tok, block))
            out = out[:m.start()] + tok + out[m.end():]
    return out, masked

def _unmask(text: str, masked_pairs: list[tuple[str, str]]) -> str:
    out = text
    for tok, block in masked_pairs:
        out = out.replace(tok, block)
    return out

def strip_comments(text: str, protect_envs: Iterable[str] = PROTECT_ENVS_DEFAULT) -> str:
    """
    라인 코멘트 제거:
    - 라인 내 첫 % 이후 삭제
    - 단, \\% 는 보존(임시 토큰)
    - \\verb, 보호 환경 내 %는 건드리지 않음
    """
    text = normalize_newlines(text)
    text, masked_verbs = _mask_inline_verbs(text)
    text, masked_envs  = _mask_protect_envs(text, protect_envs)

    PCT = "§§PERCENT_ESC§§"
    text = text.replace(r"\%", PCT)

    out_lines: list[str] = []
    for line in text.split("\n"):
        if "%" not in line:
            out_lines.append(line)
        else:
            out_lines.append(line.split("%", 1)[0])
    text = "\n".join(out_lines)

    text = text.replace(PCT, r"\%")
    text = _unmask(text, masked_envs)
    text = _unmask(text, masked_verbs)
    return text

def drop_envs(text: str, envs: Iterable[str]) -> str:
    """지정된 LaTeX 환경을 통째로 삭제."""
    out = text
    for env in envs:
        out = re.sub(rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}", "", out, flags=re.S)
    return out

TODO_CMDS_DEFAULT = (r"\todo", r"\marginpar")

def drop_inline_commands(text: str, commands: Iterable[str] = TODO_CMDS_DEFAULT) -> str:
    """\\todo{...} 같은 인라인 명령 삭제. 단순 중괄호 1단계만 처리."""
    out = text
    for cmd in commands:
        out = re.sub(rf"{re.escape(cmd)}\{{[^{{}}]*\}}", "", out)
    # 조건부 주석 블록
    out = re.sub(r"\\iffalse.*?\\fi", "", out, flags=re.S)
    return out


def drop_after_markers(text: str, patterns: list[str]) -> str:
    """
    patterns 중 첫 매치 지점부터 끝까지 버린다.
    예: ["\\appendix\\b", "\\section\\*?\\{Generations\\}", "\\section\\*?\\{Bias\\}"]
    """
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            return text[:m.start()]
    return text


def drop_noise_commands(text: str) -> str:
    # \looseness=-1 같은 할당형까지 제거
    text = re.sub(r"\\looseness\s*=?\s*-?\d+", "", text)

    # 나머지 레이아웃 보조 명령
    names = ["maketitle", "vspace", "phantom"]
    out = text
    for n in names:
        out = re.sub(rf"\\{n}\*?(?:\[[^\]]*\])?(?:\{{[^{{}}]*\}})?", "", out)

    # 지우고 남은 공백 정리(선택)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out



def clean_text(
    text: str,
    drop_env_list: Iterable[str] = PROTECT_ENVS_DEFAULT,
    also_drop_inline_todos: bool = True,
) -> str:
    """
    코멘트/환경/잡명령 제거(본문 전용 정리).
    preclean_for_body() 이후에 쓰는 걸 권장.
    """
    s = strip_comments(text, protect_envs=drop_env_list)
    s = drop_envs(s, drop_env_list)
    if also_drop_inline_todos:
        s = drop_inline_commands(s)
    return s

# -------- 파이프라인 헬퍼 --------

def preclean_for_body(text: str) -> str:
    """
    프리앰블 쓰레기 걷어내고 본문만 남기는 사전 정리.
    expander 이후에 호출해라.
    """
    s = extract_document_body(text)
    s = drop_setup_blocks(s)
    s = drop_noise_commands(s)
    return s
