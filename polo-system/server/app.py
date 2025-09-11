from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from sqlalchemy import text

# 환경 변수 로드
load_dotenv()

# 모듈 alias로 불러 충돌 방지
from routes import upload, results, generate as easy_generate
from services.database.db import DB

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행되는 함수"""
    # 시작 시
    print("🚀 POLO 서버 시작 중...")
    
    # 데이터베이스 초기화 및 연결 테스트
    await DB.init()
    print(f"✅ 데이터베이스 연결 성공 (모드: {DB.mode})")
    
    yield
    
    # 종료 시
    print("🛑 POLO 서버 종료 중...")

# FastAPI 앱 생성
app = FastAPI(
    title="POLO Easy Inference API", 
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ✅ easy 전용 prefix
app.include_router(upload.router,        prefix="/easy")
app.include_router(results.router,       prefix="/easy")
app.include_router(easy_generate.router, prefix="/generate")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
    
    @app.get("/db/health")
    async def db_health():
        """데이터베이스 연결 상태 확인"""
        try:
            # 간단한 쿼리로 연결 테스트
            async with DB.session() as session:
                await session.execute(text("SELECT 1"))
            return {
                "status": "ok",
                "database": "connected",
                "mode": DB.mode
            }
        except Exception as e:
            return {
                "status": "error", 
                "database": "disconnected",
                "error": str(e)
            }
