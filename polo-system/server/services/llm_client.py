from __future__ import annotations

import os
import logging
from typing import Dict, Any, Optional

import httpx

# -----------------------------------------------------------------------------
# Env & Logger
# -----------------------------------------------------------------------------
EASY_MODEL_URL = os.getenv("EASY_MODEL_URL", "http://localhost:5003")

# 읽기 타임아웃을 None으로 (장시간 생성 대응), 연결만 30초
TIMEOUT = httpx.Timeout(connect=30.0, read=None, write=None, pool=None)

# 윈도우 소켓 종료 시 커넥션 재사용 이슈를 피하려고 keep-alive 끔
DEFAULT_LIMITS = httpx.Limits(max_keepalive_connections=0, max_connections=20)
DEFAULT_HEADERS = {"Connection": "close"}

logger = logging.getLogger("easy_client")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _url(path: str) -> str:
    return f"{EASY_MODEL_URL.rstrip('/')}/{path.lstrip('/')}"

def _payload_style(style: Optional[str]) -> Optional[str]:
    # 서버 기본이 three_para_ko라 명시적으로 넘겨주는 게 안전
    return style or "three_para_ko"

def _text_style(style: Optional[str]) -> str:
    return style or "default"

def _json(resp: httpx.Response) -> Dict[str, Any]:
    resp.raise_for_status()
    return resp.json()  # type: ignore[return-value]

def load_assets_metadata(source_dir: str) -> Dict[str, Any]:
    """assets.jsonl에서 이미지 메타데이터 로드"""
    from pathlib import Path
    import json
    
    assets_file = Path(source_dir) / "assets.jsonl"
    if not assets_file.exists():
        return {}
    
    assets = {}
    try:
        with open(assets_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    asset = json.loads(line)
                    if asset.get("kind") == "figure" and asset.get("graphics"):
                        assets[asset["id"]] = asset
    except Exception as e:
        logger.warning(f"[LLM_CLIENT] assets.jsonl 로드 실패: {e}")
    
    return assets


# 공용 AsyncClient (HTTP/2 끔, keep-alive 끔, 읽기 무제한)
def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=TIMEOUT,
        http2=False,
        limits=DEFAULT_LIMITS,
        headers=DEFAULT_HEADERS,
        follow_redirects=False,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
async def run_batch(
    paper_id: str,
    jsonl_path: str,
    output_dir: str,
    *,
    style: Optional[str] = "three_para_ko",
    assets_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    /batch 호출: 섹션별 재서술(+옵션 viz) → 결과 저장.
    - jsonl_path: 파일 경로(또는 JSONL 문자열) 그대로 서버에 전달.
    - assets_metadata: 이미지 메타데이터 (선택사항)
    """
    url = _url("/batch")
    payload = {
        "paper_id": str(paper_id),
        "chunks_jsonl": jsonl_path,
        "output_dir": output_dir,
        "style": _payload_style(style),
    }
    if assets_metadata:
        payload["assets_metadata"] = assets_metadata
    
    async with _client() as client:
        res = await client.post(url, json=payload)
        data = _json(res)
        logger.info(
            "✅ Easy /batch 완료 | paper_id=%s | success=%s | failed=%s | timed_out=%s",
            paper_id, data.get("success"), data.get("failed"), data.get("ok") is False
        )
        return data


async def run_from_transport(
    paper_id: str,
    transport_path: str,
    *,
    output_dir: Optional[str] = None,
    style: Optional[str] = "three_para_ko",
) -> Dict[str, Any]:
    """
    /from-transport 호출: .jsonl / .jsonl.gz / .tex 파일을 서버가 읽어 /batch로 위임.
    """
    url = _url("/from-transport")
    payload = {
        "paper_id": str(paper_id),
        "transport_path": transport_path,
        "output_dir": output_dir,
        "style": _payload_style(style),
    }
    async with _client() as client:
        res = await client.post(url, json=payload)
        data = _json(res)
        logger.info(
            "✅ Easy /from-transport 완료 | paper_id=%s | success=%s | failed=%s",
            paper_id, data.get("success"), data.get("failed")
        )
        return data


async def simplify_text(
    text: str,
    *,
    style: Optional[str] = "default",
) -> Dict[str, Any]:
    """
    /easy 호출: 단일 텍스트를 쉬운 버전으로 변환.
    """
    url = _url("/easy")
    payload = {
        "text": text,
        "style": _text_style(style),
    }
    async with _client() as client:
        res = await client.post(url, json=payload)
        data = _json(res)
        logger.info("✅ Easy /easy 완료 | chars=%s", len(text))
        return data


async def generate_json(text: str) -> Dict[str, Any]:
    """
    /generate 호출: 원문에서 섹션 추출 → Grounded JSON 산출.
    """
    url = _url("/generate")
    payload = {"text": text}
    async with _client() as client:
        res = await client.post(url, json=payload)
        data = _json(res)
        logger.info("✅ Easy /generate 완료 | sections_filled=ok")
        return data


async def health() -> Dict[str, Any]:
    """
    /health 호출: 서버 상태 확인.
    """
    url = _url("/health")
    async with _client() as client:
        res = await client.get(url)
        data = _json(res)
        logger.info(
            "ℹ️ health | status=%s | gpu_available=%s | model=%s",
            data.get("status"), data.get("gpu_available"), data.get("model_name")
        )
        return data