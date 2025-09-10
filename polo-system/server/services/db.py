# server/services/db.py
import os
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import asyncio
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
from psycopg2.extras import RealDictCursor

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConfig:
    """데이터베이스 설정 클래스"""
    
    def __init__(self):
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = os.getenv("POSTGRES_PORT", "5432")
        self.database = os.getenv("POSTGRES_DB", "polo_db")
        self.username = os.getenv("POSTGRES_USER", "polo_user")
        self.password = os.getenv("POSTGRES_PASSWORD", "polo_password")
        
    @property
    def database_url(self) -> str:
        """SQLAlchemy용 데이터베이스 URL 생성"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def psycopg2_url(self) -> str:
        """psycopg2용 데이터베이스 URL 생성"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class DatabaseManager:
    """PostgreSQL 데이터베이스 관리 클래스"""
    
    def __init__(self):
        self.config = DatabaseConfig()
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """SQLAlchemy 엔진 초기화"""
        try:
            self.engine = create_engine(
                self.config.database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False  # SQL 쿼리 로깅 (개발 시 True로 설정)
            )
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info("데이터베이스 엔진이 성공적으로 초기화되었습니다.")
        except Exception as e:
            logger.error(f"데이터베이스 엔진 초기화 실패: {e}")
            raise
    
    def get_session(self) -> Session:
        """데이터베이스 세션 반환"""
        if not self.SessionLocal:
            raise RuntimeError("데이터베이스가 초기화되지 않았습니다.")
        return self.SessionLocal()
    
    @asynccontextmanager
    async def get_async_session(self):
        """비동기 컨텍스트 매니저로 세션 관리"""
        session = self.get_session()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"데이터베이스 세션 오류: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """데이터베이스 연결 테스트"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                return result.fetchone()[0] == 1
        except Exception as e:
            logger.error(f"데이터베이스 연결 테스트 실패: {e}")
            return False
    

class PostgreSQLClient:
    """psycopg2를 사용한 직접 PostgreSQL 클라이언트"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
    
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.connection = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                cursor_factory=RealDictCursor
            )
            logger.info("PostgreSQL에 성공적으로 연결되었습니다.")
        except Exception as e:
            logger.error(f"PostgreSQL 연결 실패: {e}")
            raise
    
    def disconnect(self):
        """데이터베이스 연결 해제"""
        if self.connection:
            self.connection.close()
            logger.info("PostgreSQL 연결이 해제되었습니다.")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """쿼리 실행 및 결과 반환"""
        if not self.connection:
            self.connect()
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    return cursor.fetchall()
                return []
        except Exception as e:
            logger.error(f"쿼리 실행 실패: {e}")
            self.connection.rollback()
            raise

# 전역 데이터베이스 매니저 인스턴스
db_manager = DatabaseManager()

# 의존성 주입을 위한 함수
def get_database():
    """FastAPI 의존성 주입용 데이터베이스 세션 반환"""
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()

def get_postgres_client():
    """FastAPI 의존성 주입용 PostgreSQL 클라이언트 반환"""
    client = PostgreSQLClient(db_manager.config)
    client.connect()
    try:
        yield client
    finally:
        client.disconnect()

