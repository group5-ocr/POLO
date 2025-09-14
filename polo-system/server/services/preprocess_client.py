import os
import httpx
import logging

logger = logging.getLogger("preprocess_client")
logger.setLevel(logging.INFO)

PREPROCESS_URL = os.getenv("PREPROCESS_URL", "http://localhost:5002")
TIMEOUT = int(os.getenv("PREPROCESS_TIMEOUT", "300"))


async def run(paper_id: str, source_dir: str, callback: str):
    """
    Preprocess 서비스 호출 → 완료되면 /generate/preprocess/callback 으로 콜백
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{PREPROCESS_URL}/preprocess",
                json={
                    "input_path": source_dir,
                    "output_dir": f"data/outputs/{paper_id}",
                    "config_path": "configs/default.yaml"
                },
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"✅ 전처리 완료: paper_id={paper_id}, out_dir={result.get('out_dir')}")
            return result
    except Exception as e:
        logger.error(f"❌ 전처리 요청 실패: {e}")
        raise