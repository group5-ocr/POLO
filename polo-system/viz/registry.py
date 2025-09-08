# -*- coding: utf-8 -*-
# 도형 문법(Grammar)을 등록/조회하는 레지스트리
from typing import Dict, Any, Callable

class Grammar:
    # 문법의 메타데이터를 보관하는 간단한 클래스
    def __init__(self, id: str, needs: list, optional: list, renderer: Callable[[dict, str], None]):
        self.id = id                  # 문법 ID (예: "curve_generic")
        self.needs = needs            # 필수 입력 키 목록
        self.optional = optional      # 선택 입력 키 목록
        self.renderer = renderer      # 렌더 함수 (inputs, out_path)

# 전역 레지스트리 (id -> Grammar)
GRAMMARS: Dict[str, Grammar] = {}

def register(grammar: Grammar):
    # 새 문법을 레지스트리에 등록
    GRAMMARS[grammar.id] = grammar

def get(id_: str) -> Grammar:
    # 문법 ID로 Grammar 조회 (없으면 에러 발생)
    if id_ not in GRAMMARS:
        raise KeyError(f"Grammar '{id_}' is not registered")
    return GRAMMARS[id_]
