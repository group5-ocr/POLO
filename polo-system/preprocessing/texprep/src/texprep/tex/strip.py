# -*- coding: utf-8 -*-
"""
코멘트/불필요 환경 제거 (modern typing)
- 개행 통일
- 코멘트(%) 제거. 단 \%, \verb, verbatim류 안의 %는 보존
- 지정 환경 통째 제거(tikz, minted, lstlisting, Verbatim 등)
- \todo{..}, \marginpar{..}, \iffalse...\fi 같은 잡음 제거 옵션
"""
from __future__ import annotations
from typing import Iterable
import re

PROTECT_ENVS_DEFAULT = ("verbatim", "Verbatim", "lstlisting", "lstlisting*", "minted", "tikzpicture")
VERB_INLINE_RE = re.compile(r"""\\verb\*?(?P<delim>[^A-Za-z0-9\s])(?P<body>.*?)(?P=delim)""")
_PROTECT_BLOCK = "§§PROTECT_BLOCK_{}§§"
_PROTECT_VERB = "§§PROTECT_VERB_{}§§"

def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")

def _mask_inline_verbs(text: str) -> tuple[str, list[tuple[str, str]]]:
    """\verb 구간은 %가 나와도 텍스트. 토큰으로 보호."""
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
    코멘트 제거:
    - 라인 내 % 이후 삭제
    - \% 는 보존(임시 토큰)
    - \verb, 보호 환경 내 %는 건드리지 않음
    """
    text = normalize_newlines(text)
    text, masked_verbs = _mask_inline_verbs(text)
    text, masked_envs = _mask_protect_envs(text, protect_envs)

    PCT = "§§PERCENT_ESC§§"
    text = text.replace(r"\%", PCT)

    out_lines: list[str] = []
    for line in text.split("\n"):
        if "%" not in line:
            out_lines.append(line)
            continue
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
    """\todo{...} 같은 인라인 명령 삭제. 단순 중괄호 1단계만 처리."""
    out = text
    for cmd in commands:
        out = re.sub(rf"{re.escape(cmd)}\{{[^{{}}]*\}}", "", out)
    out = re.sub(r"\\iffalse.*?\\fi", "", out, flags=re.S)
    return out

def clean_text(
    text: str,
    drop_env_list: Iterable[str] = PROTECT_ENVS_DEFAULT,
    also_drop_inline_todos: bool = True,
) -> str:
    """확장 결과를 요약/파싱용으로 정리."""
    s = strip_comments(text, protect_envs=drop_env_list)
    s = drop_envs(s, drop_env_list)
    if also_drop_inline_todos:
        s = drop_inline_commands(s)
    return s
