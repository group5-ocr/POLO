import os
import httpx
import logging

logger = logging.getLogger("math_client")
logger.setLevel(logging.INFO)

MATH_MODEL_URL = os.getenv("MATH_MODEL_URL", "http://localhost:5004")
CALLBACK_URL = os.getenv("CALLBACK_URL", "http://localhost:8000/generate/math-callback")
TIMEOUT = int(os.getenv("MATH_TIMEOUT", "300"))


async def ingest_text(paper_id: str, math_text_path: str):
    """
    Math 모델 API 호출 → 완료되면 콜백 전송.
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # 서버 Math API는 POST /math, 바디는 {"path": ...}
            resp = await client.post(
                f"{MATH_MODEL_URL}/math",
                json={"path": math_text_path},
            )
            resp.raise_for_status()
            data = resp.json()

        # 결과 경로 추출(보고서 .tex 우선, 없으면 json)
        outputs = data.get("outputs") or {}
        result_path = outputs.get("report_tex") or outputs.get("json")

        # 콜백 URL이 베이스인 경우 자동 보정
        callback_url = CALLBACK_URL
        if "/generate/" not in callback_url:
            if callback_url.endswith("/"):
                callback_url = callback_url[:-1]
            callback_url = f"{callback_url}/generate/math-callback"

        # 콜백 전송
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                callback_url,
                json={
                    "paper_id": paper_id,
                    "math_result_path": result_path,
                    "sections": None,
                },
            )
        logger.info(f"✅ Math 콜백 전송 완료: paper_id={paper_id}")

    except Exception as e:
        logger.error(f"❌ Math 모델 호출 실패: {e}")