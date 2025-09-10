# server/services/environment.py
import os
import socket
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class EnvironmentManager:
    """환경 감지 및 DB/로컬 파일 시스템 선택 관리"""
    
    def __init__(self):
        self.is_db_available = False
        self.is_academy_network = False
        self._check_environment()
    
    def _check_environment(self):
        """환경 상태 확인"""
        # 1. 학원 네트워크 감지 (IP 범위 기반)
        self.is_academy_network = self._detect_academy_network()
        
        # 2. DB 연결 가능 여부 확인
        self.is_db_available = self._test_db_connection()
        
        logger.info(f"환경 감지 완료 - 학원 네트워크: {self.is_academy_network}, DB 사용 가능: {self.is_db_available}")
    
    def _detect_academy_network(self) -> bool:
        """학원 네트워크 감지 (IP 범위 기반)"""
        try:
            # 학원 네트워크 IP 범위 (예: 192.168.1.x, 10.0.0.x 등)
            academy_ip_ranges = [
                "192.168.1.",  # 일반적인 학원 네트워크
                "10.0.0.",     # 일반적인 학원 네트워크
                "172.16.",     # 일반적인 학원 네트워크
                # 필요에 따라 추가 IP 범위 설정
            ]
            
            # 현재 IP 주소 확인
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            # 학원 네트워크 IP 범위에 속하는지 확인
            for ip_range in academy_ip_ranges:
                if local_ip.startswith(ip_range):
                    logger.info(f"학원 네트워크 감지됨: {local_ip}")
                    return True
            
            logger.info(f"집 네트워크 감지됨: {local_ip}")
            return False
            
        except Exception as e:
            logger.warning(f"네트워크 감지 실패: {e}")
            return False
    
    def _test_db_connection(self) -> bool:
        """DB 연결 가능 여부 테스트"""
        try:
            from .db import db_manager
            return db_manager.test_connection()
        except Exception as e:
            logger.warning(f"DB 연결 테스트 실패: {e}")
            return False
    
    def should_use_database(self) -> bool:
        """DB 사용 여부 결정"""
        # 학원 네트워크이고 DB가 사용 가능하면 DB 사용
        if self.is_academy_network and self.is_db_available:
            return True
        
        # 그 외의 경우 로컬 파일 시스템 사용
        return False
    
    def get_storage_info(self) -> Dict[str, Any]:
        """현재 저장소 정보 반환"""
        return {
            "use_database": self.should_use_database(),
            "is_academy_network": self.is_academy_network,
            "is_db_available": self.is_db_available,
            "storage_type": "database" if self.should_use_database() else "local_filesystem"
        }

# 전역 환경 관리자 인스턴스
env_manager = EnvironmentManager()


