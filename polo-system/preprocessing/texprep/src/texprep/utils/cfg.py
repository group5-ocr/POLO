# src/texprep/utils/cfg.py
# YAML 읽어서 파이프라인이 쓸 수 있는 파이썬 dict로 변환. 필요하면 환경변수로 덮어쓰기.

import os, yaml

# YAML에 값이 없거나 환경변수도 없을 때 쓸 기본값들
DEFAULTS = {
  "root_dir": ".",                  # TeX 소스가 있는 기본 디렉토리
  "out_dir": "./server/data/out",          # 전처리 산출물을 저장할 디렉토리
  "drop_envs": [                    # 파서가 무시할 LaTeX 환경 목록
    "tikzpicture","minted","lstlisting","verbatim","Verbatim"
  ],
  "chunk": {                        # 청킹 기본 옵션
    "max_chars": 3800,              # 한 청크당 최대 글자 수
    "overlap": 1,                   # 청크 사이 겹치는 문단 수
  },
}

def _merge(a, b):
    """
    두 dict를 병합하는 간단 유틸.
    b 값이 있으면 덮어쓰고,
    dict 안에 dict가 있으면 재귀적으로 병합한다.
    """
    out = {**a}
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

def load_cfg(path: str) -> dict:
    """
    설정 로더:
    1. YAML 파일을 읽어 dict로 변환한다.
    2. 기본값(DEFAULTS)과 병합한다.
    3. 환경변수(ENV)로 다시 덮어쓴다.
    4. 최소한의 검증을 하고 최종 config dict 반환.
    """
    # 1. YAML 파일 읽기
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    
    # 2. 기본값과 병합
    cfg = _merge(DEFAULTS, user)

    # 3. ENV 덮어쓰기 (있으면 우선 적용)
    cfg["root_dir"] = os.getenv("ROOT_DIR", cfg["root_dir"])
    cfg["out_dir"]  = os.getenv("OUT_DIR", cfg["out_dir"])
    if env := os.getenv("DROP_ENVS"):  # 쉼표로 구분된 문자열 예: "tikzpicture,minted"
        cfg["drop_envs"] = [s.strip() for s in env.split(",") if s.strip()]

    # 4. 최소 검증: 값이 비정상적일 때 기본으로 되돌림
    if not cfg["drop_envs"]:
        cfg["drop_envs"] = DEFAULTS["drop_envs"]
    if (not isinstance(cfg["chunk"]["max_chars"], int)
        or cfg["chunk"]["max_chars"] < 500):
        cfg["chunk"]["max_chars"] = DEFAULTS["chunk"]["max_chars"]

    return cfg
