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
            resp = await client.post(
                f"{MATH_MODEL_URL}/generate",
                json={"paper_id": paper_id, "math_text_path": math_text_path},
            )
            resp.raise_for_status()
            data = resp.json()

        # 콜백 전송
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                CALLBACK_URL,
                json={
                    "paper_id": paper_id,
                    "math_result_path": data.get("result_path"),
                    "sections": data.get("sections"),
                },
            )
        logger.info(f"✅ Math 콜백 전송 완료: paper_id={paper_id}")

    except Exception as e:
        logger.error(f"❌ Math 모델 호출 실패: {e}")