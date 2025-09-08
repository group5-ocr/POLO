# src/texprep/tex/math.py
# -*- coding: utf-8 -*-
"""
수식 추출기 (메인 수식 선별 포함)
- 디스플레이: equation/align/gather/... + $$...$$ + \[...\]
- 인라인: $...$, \(...\)
- '메인' 수식: 디스플레이 + 줄 전체를 차지하는 인라인 승격
- 본문엔 ⟨EQ:{id}⟩(디스플레이/승격), ⟨eq:{id}⟩(순수 인라인) 치환
"""
from __future__ import annotations
import re
from typing import Any

DISPLAY_ENVS = (
    "equation","equation*","align","align*","gather","gather*",
    "multline","multline*","flalign","flalign*","eqnarray","eqnarray*"
)

_CAP_LABEL = re.compile(r"\\label\{(?P<label>[^}]+)\}")
_TAG       = re.compile(r"\\tag\{[^}]*\}")

# 문단 분리(빈 줄 기준)
def _split_paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]

def _clean_inner(tex: str) -> str:
    tex = _CAP_LABEL.sub("", tex)
    tex = _TAG.sub("", tex)
    return tex.strip()

def extract_display(text: str) -> tuple[str, list[dict[str, Any]]]:
    """
    디스플레이 수식을 ⟨EQ:{id}⟩로 치환하고 목록 반환.
    env: 환경명 또는 "$$" 또는 "\\[\\]"
    """
    out: list[dict[str, Any]] = []
    idx = 0

    # \begin{env}...\end{env}
    pat_env = re.compile(
        r"\\begin\{(?P<env>" + "|".join(map(re.escape, DISPLAY_ENVS)) + r")\}"
        r"(?P<body>.*?)\\end\{(?P=env)\}",
        re.S
    )
    def repl_env(m: re.Match) -> str:
        nonlocal idx
        env = m.group("env")
        body = m.group("body")
        lab = _CAP_LABEL.search(body)
        inner = _clean_inner(body)
        idx += 1
        eid = lab.group("label") if lab else f"eq_{idx}"
        out.append({"id": eid, "env": env, "tex": inner, "display": True, "promoted": False})
        return f"⟨EQ:{eid}⟩"
    text = pat_env.sub(repl_env, text)

    # $$...$$
    pat_dollars = re.compile(r"\$\$(?P<body>.+?)\$\$", re.S)
    def repl_dollars(m: re.Match) -> str:
        nonlocal idx
        idx += 1
        eid = f"eq_{idx}"
        out.append({"id": eid, "env": "$$", "tex": _clean_inner(m.group("body")), "display": True, "promoted": False})
        return f"⟨EQ:{eid}⟩"
    text = pat_dollars.sub(repl_dollars, text)

    # \[...\]
    pat_bracket = re.compile(r"\\\[(?P<body>.+?)\\\]", re.S)
    def repl_bracket(m: re.Match) -> str:
        nonlocal idx
        idx += 1
        eid = f"eq_{idx}"
        out.append({"id": eid, "env": "\\[\\]", "tex": _clean_inner(m.group("body")), "display": True, "promoted": False})
        return f"⟨EQ:{eid}⟩"
    text = pat_bracket.sub(repl_bracket, text)

    return text, out

def extract_inline(text: str) -> tuple[str, list[dict[str, str]]]:
    """
    인라인 수식을 ⟨eq:{id}⟩로 치환하고 목록 반환.
    """
    inlines: list[dict[str, str]] = []
    text = text.replace("$$", "§§DOLLARS§§")  # 보호

    def repl_inline(m: re.Match) -> str:
        iid = f"ieq_{len(inlines)+1}"
        inlines.append({"id": iid, "tex": m.group(1)})
        return f"⟨eq:{iid}⟩"

    # $...$
    text = re.sub(r"\$([^\$]+?)\$", repl_inline, text)
    # \(...\)
    text = re.sub(r"\\\((.+?)\\\)", repl_inline, text)

    text = text.replace("§§DOLLARS§§", "$$")
    return text, inlines

# ---- 메인 수식 선별(인라인 승격) ----

_LINE_ONLY_INLINE = re.compile(r"^\s*(?:⟨eq:(?P<iid>ieq_\d+)⟩)\s*\.?\s*$")
# 줄이 거의 수식 뿐이고 앞뒤 텍스트가 아주 짧은 경우도 허용(잡음 단어 수 0~2)
_TOKENY_LINE = re.compile(r"^\s*(?:[,(]?\s*)?⟨eq:(?P<iid>ieq_\d+)⟩(?:\s*[.,;:]?)\s*$")

def _promote_inline_blocks(body: str, inline_map: dict[str, dict[str, str]]) -> tuple[str, list[dict[str, Any]], list[dict[str, str]]]:
    """
    본문을 문단/라인 단위로 보고,
    줄 전체가 인라인 수식 플레이스홀더인 경우 '메인'으로 승격.
    - 승격되면 ⟨EQ:id⟩로 대체
    - 남은 인라인은 그대로 유지
    """
    paras = _split_paragraphs(body)
    main_eqs: list[dict[str, Any]] = []
    kept_inlines: dict[str, dict[str, str]] = dict(inline_map)  # 얕은 복사

    # 재구성된 본문
    new_paras: list[str] = []

    for p_idx, para in enumerate(paras):
        lines = para.split("\n")
        new_lines: list[str] = []
        for l_idx, line in enumerate(lines):
            m = _LINE_ONLY_INLINE.match(line) or _TOKENY_LINE.match(line)
            if m:
                iid = m.group("iid")
                meta = kept_inlines.pop(iid, None)
                if meta:
                    # 새 eq id 만들고 승격
                    eq_id = meta["id"].replace("ieq_", "eq_")
                    main_eqs.append({
                        "id": eq_id,
                        "env": "INLINE_PROMOTED",
                        "tex": meta["tex"],
                        "display": True,
                        "promoted": True,
                        "location": {"para_index": p_idx, "line_index": l_idx}
                    })
                    new_lines.append(f"⟨EQ:{eq_id}⟩")
                    continue
            new_lines.append(line)
        new_paras.append("\n".join(new_lines))

    new_body = "\n\n".join(new_paras)
    # 남은 인라인 목록으로 환원
    remaining_inline = list(kept_inlines.values())
    return new_body, main_eqs, remaining_inline

def extract_math_with_promotion(text: str) -> tuple[str, list[dict[str, Any]], list[dict[str, str]]]:
    """
    통합 추출:
      1) 디스플레이 수식 추출
      2) 인라인 수식 추출
      3) 줄-전용 인라인 승격 → 메인 수식에 합쳐서 반환
    """
    body, displays = extract_display(text)
    body, inlines = extract_inline(body)
    inline_map = {d["id"]: d for d in inlines}
    body, promoted, remaining_inlines = _promote_inline_blocks(body, inline_map)

    # 위치 정보는 디스플레이에선 생략(필요하면 후처리로 붙여라)
    equations = displays + promoted
    return body, equations, remaining_inlines
