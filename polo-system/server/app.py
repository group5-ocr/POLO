from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 모듈 alias로 불러 충돌 방지
from routes import convert, upload, results, generate as easy_generate, database, files
from services.db import db_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행되는 함수"""
    # 시작 시
    print("🚀 POLO 서버 시작 중...")
    
    # 데이터베이스 연결 테스트
    if db_manager.test_connection():
        print("✅ 데이터베이스 연결 성공")
    else:
        print("❌ 데이터베이스 연결 실패")
    
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
app.include_router(convert.router,       prefix="/easy")
app.include_router(results.router,       prefix="/easy")
app.include_router(easy_generate.router, prefix="/easy")
    
# ✅ 데이터베이스 관련 라우트
app.include_router(database.router,      prefix="/db")
    
# ✅ 파일 관리 라우트
app.include_router(files.router,         prefix="/api")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
    
@app.get("/db/health")
def db_health():
    """데이터베이스 연결 상태 확인"""
    is_connected = db_manager.test_connection()
    return {
        "status": "ok" if is_connected else "error",
        "database": "connected" if is_connected else "disconnected"
    }
