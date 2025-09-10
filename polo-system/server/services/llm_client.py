# server/services/llm_client.py
import os
import requests
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def _env(name: str, default: str) -> str:
    return os.getenv(name, default)

class EasyLLMClient:
    """파인튜닝된 Easy LLM 서비스 클라이언트"""

    def __init__(
        self,
        base_url: str = None,
        generate_path: str = None,
        default_timeout: int = None,
        max_retries: int = None,
    ):
        self.base_url = base_url or _env("EASY_LLM_BASE_URL", "http://localhost:5003")
        # 모델 서버가 /generate가 아닌 /easy/generate인 경우 환경변수로 바꿔줄 수 있음
        self.generate_path = generate_path or _env("EASY_LLM_GENERATE_PATH", "/generate")
        self.session = requests.Session()
        self.default_timeout = int(default_timeout or _env("EASY_LLM_TIMEOUT", "900"))
        self.max_retries = int(max_retries or _env("EASY_LLM_MAX_RETRIES", "2"))

    def health_check(self) -> bool:
        """모델 서비스 상태 확인: /health -> /healthz -> / 순서로 시도"""
        for path in ("/health", "/healthz", "/"):
            try:
                r = self.session.get(f"{self.base_url}{path}", timeout=5)
                if r.ok:
                    # 일부 서버는 단순 텍스트를 반환하므로 상태코드로만 판단
                    return True
            except Exception as e:
                logger.debug(f"health check {path} failed: {e}")
        logger.error(f"Easy LLM 서비스 연결 실패(base_url={self.base_url})")
        return False

    def _post_json(self, path: str, payload: Dict[str, Any], timeout: int):
        url = f"{self.base_url}{path}"
        r = self.session.post(url, json=payload, timeout=timeout, headers={"Content-Type": "application/json"})
        return r

    def generate(
        self,
        text: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Optional[dict]:
        """
        논문 텍스트를 모델 서비스로 보내 easy_json을 받는다.
        - 반환이 문자열(JSON str)이어도 처리
        - 4xx/5xx 재시도(최대 max_retries)
        """
        payload = {
            "text": text,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
        }
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 2):  # 1회+재시도
            try:
                resp = self._post_json(self.generate_path, payload, self.default_timeout)
                if resp.status_code == 200:
                    # JSON 파싱 보강
                    try:
                        return resp.json()
                    except Exception as je:
                        # 서버가 text/plain JSON 문자열을 보낼 수 있음
                        txt = resp.text.strip()
                        try:
                            import json as _json
                            return _json.loads(txt)
                        except Exception as j2:
                            last_error = RuntimeError(f"JSON 파싱 실패: {je}; text[:200]={txt[:200]}")
                            logger.error(str(last_error))
                else:
                    # 4xx/5xx
                    preview = resp.text[:200].replace("\n", " ")
                    logger.error(f"[{attempt}/{self.max_retries+1}] 모델 응답 오류: {resp.status_code} - {preview}")
                    last_error = RuntimeError(f"HTTP {resp.status_code}")
            except Exception as e:
                last_error = e
                logger.error(f"[{attempt}/{self.max_retries+1}] 요청 실패: {e}")

            if attempt <= self.max_retries:
                time.sleep(min(5 * attempt, 15))

        logger.error(f"텍스트 생성 최종 실패: {last_error}")
        return None

    def get_model_info(self) -> Optional[dict]:
        """모델 정보 조회: / -> /health -> /healthz 순으로 시도"""
        for path in ("/", "/health", "/healthz"):
            try:
                r = self.session.get(f"{self.base_url}{path}", timeout=5)
                if r.ok:
                    try:
                        return r.json()
                    except Exception:
                        return {"text": r.text}
            except Exception as e:
                logger.debug(f"model info {path} failed: {e}")
        logger.error("모델 정보 조회 실패")
        return None

# 전역 인스턴스
easy_llm = EasyLLMClient()