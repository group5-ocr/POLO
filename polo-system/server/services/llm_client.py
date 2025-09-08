import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class EasyLLMClient:
    """파인튜닝된 Easy LLM 서비스 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:5003"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """모델 서비스 상태 확인"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Easy LLM 서비스 연결 실패: {e}")
            return False
    
    def generate(self, text: str, max_length: int = 512, temperature: float = 0.7) -> Optional[dict]:
        """논문 텍스트를 /generate로 보내 JSON 변환 결과를 받는다"""
        try:
            payload = {
                "text": text
            }

            response = self.session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=120,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"텍스트 생성 실패: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"텍스트 생성 요청 실패: {e}")
            return None
    
    def get_model_info(self) -> Optional[dict]:
        """모델 정보 조회: 루트(/) 또는 헬스(/health)에서 정보 수집"""
        try:
            # 우선 루트 엔드포인트 시도 ("POLO Easy Model API", model 등)
            response = self.session.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                return response.json()

            # 실패 시 /health로 대체
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"모델 정보 조회 실패: {e}")
            return None

# 전역 클라이언트 인스턴스
easy_llm = EasyLLMClient()