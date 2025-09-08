
# -*- coding: utf-8 -*-
"""
switch.py — 시각화 라벨 한/영 병기 스위치 유틸
- target_lang(기본: "ko")에서 라벨을 우선 사용
- ko 라벨이 없을 경우 en으로 자동 폴백
- bilingual 모드 스위치: "off" | "missing" | "always"
    - off: 타겟 라벨만 사용(없으면 폴백 언어만 사용)
    - missing: 타겟 라벨이 없을 때만 "폴백(EN)" 같이 힌트를 병기
    - always: 타겟 라벨이 있어도 항상 "타겟 (폴백)" 병기
사용 예:
    from switch import make_opts, resolve_label
    opts = make_opts(target_lang="ko", bilingual="missing")
    title = resolve_label(spec.get("labels", {}), opts)
"""
from dataclasses import dataclass

@dataclass
class BilingualOptions:
    target_lang: str = "ko"
    fallback_lang: str = "en"
    bilingual: str = "missing"  # "off", "missing", "always"
    join_fmt_always: str = "{primary} ({fallback})"
    join_fmt_missing: str = "{fallback} (EN)"
    strip: bool = True

def make_opts(target_lang="ko", bilingual="missing", fallback_lang="en"):
    b = (bilingual or "missing").lower()
    if b not in {"off","missing","always"}: b = "missing"
    return BilingualOptions(target_lang, fallback_lang, b)

def _clean(s: str, do_strip: bool) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip() if do_strip else s

def resolve_label(labels: dict, opts: BilingualOptions) -> str:
    """labels 예: {"ko": "정확도", "en": "Accuracy"}"""
    primary = _clean(labels.get(opts.target_lang, ""), opts.strip)
    fallback = _clean(labels.get(opts.fallback_lang, ""), opts.strip)

    # bilingual 스위치별 처리
    if opts.bilingual == "off":
        return primary or fallback

    if opts.bilingual == "always":
        if primary and fallback and primary.lower() != fallback.lower():
            return opts.join_fmt_always.format(primary=primary, fallback=fallback)
        # 둘 중 하나만 있으면 있는 것만
        return primary or fallback

    # opts.bilingual == "missing"
    if primary:
        return primary
    # 타겟 라벨이 없고 폴백만 있을 때
    if fallback:
        return opts.join_fmt_missing.format(fallback=fallback)
    return ""

def merge_caption(caption_labels: dict, opts: BilingualOptions) -> str:
    """
    캡션에도 동일 정책 적용.
    caption_labels 예: {"ko": "정확도 비교", "en": "Accuracy Comparison"}
    """
    return resolve_label(caption_labels or {}, opts)
