import os
import httpx
import logging

logger = logging.getLogger("viz_client")
logger.setLevel(logging.INFO)

VIZ_MODEL_URL = os.getenv("VIZ_MODEL_URL", "http://localhost:5005")
CALLBACK_URL = os.getenv("CALLBACK_URL", "http://localhost:8000/generate/viz-callback")
TIMEOUT = int(os.getenv("VIZ_TIMEOUT", "180"))


async def generate(paper_id: str, index: int, rewritten_text: str):
    """
    시각화 모델 호출 → 완료되면 /generate/viz-callback 으로 콜백
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{VIZ_MODEL_URL}/generate",
                json={
                    "paper_id": paper_id,
                    "index": index,
                    "rewritten_text": rewritten_text,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # 콜백 전송
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                CALLBACK_URL,
                json={
                    "paper_id": paper_id,
                    "index": index,
                    "image_path": data.get("image_path"),
                },
            )
        logger.info(f"✅ Viz 콜백 전송 완료: paper_id={paper_id}, index={index}")

    except Exception as e:
        logger.error(f"❌ Viz 모델 호출 실패: {e}")