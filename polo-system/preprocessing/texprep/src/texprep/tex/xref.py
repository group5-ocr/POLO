# -*- coding: utf-8 -*-
"""
교차참조(xref) 도구
- 문단 분리
- \ref / \eqref / \autoref / \cref / \Cref / \Ref 수집
- 라벨 → 언급(문단/문장) 매핑 생성
- 플레이스홀더(⟨FIG:…⟩, ⟨TAB:…⟩, ⟨EQ:…⟩, ⟨eq:…⟩) 위치 인덱싱
"""

from __future__ import annotations
import re
from typing import Any

# ---- 문단/문장 유틸 ----

_PARA_SPLIT = re.compile(r"\n\s*\n", re.S)
_SENT_SPLIT = re.compile(r"(?<!\b(?:e|i)\.g)(?<!\bvs)\.(?:\s+|$)|[!?](?:\s+|$)")

def split_paragraphs(text: str) -> list[str]:
    """빈 줄 기준 문단 분리. 공백 문단 제거."""
    return [p.strip() for p in _PARA_SPLIT.split(text) if p and p.strip()]

def split_sentences(para: str) -> list[str]:
    """문장 분리. 얄팍한 휴리스틱이지만 논문엔 충분."""
    parts = _SENT_SPLIT.split(para)
    out = []
    for s in parts:
        s = (s or "").strip()
        if s:
            out.append(s)
    return out

# ---- 참조 탐지 ----

# \ref{label}, \eqref{label}, \autoref{label}, \cref{label}, \Cref{label}, \Ref{label}
_REF_FUNCS = ("ref","eqref","autoref","cref","Cref","Ref")
_REF_RE = re.compile(r"\\(?P<cmd>" + "|".join(map(re.escape, _REF_FUNCS)) + r")\{(?P<label>[^}]+)\}")

def find_refs(text: str) -> list[dict[str, str]]:
    """
    텍스트에서 참조 호출 전부 나열.
    반환: [{"cmd":"ref","label":"fig:foo"}, ...]
    """
    return [{"cmd": m.group("cmd"), "label": m.group("label")} for m in _REF_RE.finditer(text)]

def build_mentions_map(plain_text: str, with_sentences: bool = True, max_snippet_len: int = 600) -> dict[str, list[dict[str, Any]]]:
    """
    문단을 훑어 라벨 → 언급 리스트 생성.
    언급에는 문단/문장 인덱스, 스니펫 텍스트 포함.
    """
    mentions: dict[str, list[dict[str, Any]]] = {}
    paras = split_paragraphs(plain_text)

    for p_idx, para in enumerate(paras):
        refs = find_refs(para)
        if not refs:
            continue

        # 스니펫: 문장 단위로 줄여서 저장
        if with_sentences:
            sens = split_sentences(para)
        else:
            sens = [para]

        # 라벨별로 등장 문장 나눠 담기
        for r in refs:
            label = r["label"]
            # 해당 라벨이 들어간 문장만 추출
            matched_sents = [s for s in sens if f"\\{r['cmd']}{"{" + label + "}"}" in para and label in s] or sens
            for s_idx, s in enumerate(matched_sents[:3]):  # 과한 중복 방지로 최대 3개
                snippet = re.sub(r"\s+", " ", s).strip()
                if len(snippet) > max_snippet_len:
                    snippet = snippet[:max_snippet_len] + "…"
                mentions.setdefault(label, []).append({
                    "para_index": p_idx,
                    "sent_index": s_idx,
                    "cmd": r["cmd"],
                    "text": snippet,
                })
    return mentions

# ---- 플레이스홀더 인덱싱 ----

_PH_EQ_DISPLAY = re.compile(r"⟨EQ:(?P<id>[^>]+)⟩")
_PH_EQ_INLINE  = re.compile(r"⟨eq:(?P<id>[^>]+)⟩")
_PH_FIG        = re.compile(r"⟨FIG:(?P<id>[^>]+)⟩")
_PH_TAB        = re.compile(r"⟨TAB:(?P<id>[^>]+)⟩")

def index_placeholders(plain_text: str) -> dict[str, list[dict[str, Any]]]:
    """
    본문에서 플레이스홀더 위치를 수집.
    반환 키: eq_display, eq_inline, figure, table
    각 항목: {"id": "eq_3", "para_index": 10, "offset": 123}
    """
    indices = {
        "eq_display": [],
        "eq_inline": [],
        "figure": [],
        "table": [],
    }
    paras = split_paragraphs(plain_text)
    for p_idx, para in enumerate(paras):
        for m in _PH_EQ_DISPLAY.finditer(para):
            indices["eq_display"].append({"id": m.group("id"), "para_index": p_idx, "offset": m.start()})
        for m in _PH_EQ_INLINE.finditer(para):
            indices["eq_inline"].append({"id": m.group("id"), "para_index": p_idx, "offset": m.start()})
        for m in _PH_FIG.finditer(para):
            indices["figure"].append({"id": m.group("id"), "para_index": p_idx, "offset": m.start()})
        for m in _PH_TAB.finditer(para):
            indices["table"].append({"id": m.group("id"), "para_index": p_idx, "offset": m.start()})
    return indices

# ---- 자산에 mentions 붙이기 헬퍼 ----

def attach_mentions_to_assets(assets: list[dict[str, Any]], mentions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """
    assets 목록에 mentions를 병합.
    - 자산 라벨이 있으면 라벨 기준
    - 라벨이 없으면 빈 리스트
    """
    out: list[dict[str, Any]] = []
    for a in assets:
        label = a.get("label")
        # 내보낼 id는 이미 blocks.extract_assets에서 정해짐(fig_1/tab_1 등)
        a2 = dict(a)
        a2["mentions"] = mentions.get(label, []) if label else []
        out.append(a2)
    return out

# ---- xref 에지 후보 만들기(선택) ----

def propose_xref_edges(plain_text: str) -> list[dict[str, Any]]:
    """
    DB용 에지 후보.
    src_block을 아직 모를 때는 para_index로 보관해두고,
    적재 시 문단→block 매핑 후 real block_id로 변환하라.
    """
    edges: list[dict[str, Any]] = []
    paras = split_paragraphs(plain_text)
    for p_idx, para in enumerate(paras):
        for m in _REF_RE.finditer(para):
            edges.append({
                "src_para_index": p_idx,
                "cmd": m.group("cmd"),
                "target_label": m.group("label"),
            })
    return edges
