# server/services/viz_client.py
from __future__ import annotations

import os
import httpx
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("viz_client")
logger.setLevel(logging.INFO)

VIZ_MODEL_URL = os.getenv("VIZ_MODEL_URL", "http://localhost:5005")
# 기본은 구 generate 네임스페이스. 현재 라우터에서 비활성(410)일 수 있으니 예외 처리함.
CALLBACK_URL_DEFAULT = os.getenv("CALLBACK_URL", "http://localhost:8000").rstrip("/")
DEFAULT_CALLBACK_PATH = "/generate/viz-callback"
TIMEOUT = int(os.getenv("VIZ_TIMEOUT", "180"))


def _normalize_callback_url(callback_url: Optional[str]) -> Optional[str]:
    """
    None 이면 None 유지(콜백 생략).
    값이 base URL이면 /generate/viz-callback 을 붙여서 절대경로로 만듦.
    이미 경로가 포함돼 있으면 그대로 사용.
    """
    if callback_url is None:
        return None
    url = callback_url.strip().rstrip("/")
    if not url:
        return None
    if "/generate/" not in url:
        url = f"{url}{DEFAULT_CALLBACK_PATH}"
    return url


async def generate(
    paper_id: str,
    index: int,
    rewritten_text: str,
    callback_url: Optional[str] = f"{CALLBACK_URL_DEFAULT}{DEFAULT_CALLBACK_PATH}",
) -> Dict[str, Any]:
    """
    시각화 모델 호출 → (옵션) 콜백 전송
    - VIZ POST {VIZ_MODEL_URL}/viz
      body = {paper_id, index, rewritten_text, target_lang:"ko", bilingual:"missing"}
      resp = {image_path, success, ...}
    - callback_url 이 주어지면 콜백 POST.
      (410 Gone 이면 무시하고 지나감: 구 라우터 비활성화 대응)
    """
    api = f"{VIZ_MODEL_URL.rstrip('/')}/viz"
    cb = _normalize_callback_url(callback_url)

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                api,
                json={
                    "paper_id": paper_id,
                    "index": index,
                    "rewritten_text": rewritten_text,
                    "target_lang": "ko",
                    "bilingual": "missing",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error("❌ Viz 모델 호출 실패: %s", e)
        return {"ok": False, "paper_id": paper_id, "index": index, "error": str(e)}

    image_path = data.get("image_path")
    ok = bool(image_path)

    # 콜백 (선택)
    if cb:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    cb,
                    json={
                        "paper_id": paper_id,
                        "index": index,
                        "image_path": image_path,
                    },
                )
                # 구 라우터가 비활성(410)일 때는 무시
                if r.status_code == 410:
                    logger.info("ℹ️ Viz 콜백 라우터(410) 비활성: %s", cb)
                else:
                    r.raise_for_status()
            logger.info("✅ Viz 콜백 전송 완료: paper_id=%s index=%s", paper_id, index)
        except Exception as e:
            # 콜백 실패는 전체 파이프라인을 막지 않음
            logger.warning("⚠️ Viz 콜백 전송 실패: %s", e)

    return {"ok": ok, "paper_id": paper_id, "index": index, "image_path": image_path, "raw": data}
