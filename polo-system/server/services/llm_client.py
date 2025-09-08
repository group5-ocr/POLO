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
    
    def generate(self, text: str, max_length: int = 512, temperature: float = 0.7) -> Optional[str]:
        """텍스트 생성"""
        try:
            payload = {
                "text": text,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": 0.9
            }
            
            response = self.session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("generated_text", "")
            else:
                logger.error(f"텍스트 생성 실패: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"텍스트 생성 요청 실패: {e}")
            return None
    
    def get_model_info(self) -> Optional[dict]:
        """모델 정보 조회"""
        try:
            response = self.session.get(f"{self.base_url}/model_info", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"모델 정보 조회 실패: {e}")
            return None

# 전역 클라이언트 인스턴스
easy_llm = EasyLLMClient()