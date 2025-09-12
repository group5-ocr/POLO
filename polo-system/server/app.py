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
    import os
    import subprocess
    import time
    from pathlib import Path
    
    port = os.getenv("SERVER_PORT", "8000")
    print("🚀 POLO 서버 시작 중...")
    print(f"📊 포트: {port}")
    
    # 데이터베이스 초기화 및 연결 테스트
    await DB.init()
    print(f"✅ 데이터베이스 연결 성공 (모드: {DB.mode})")
    
    # 서비스 시작 대기
    print("⏳ 다른 서비스들이 준비될 때까지 대기 중...")
    time.sleep(3)
    
    # Viz와 Preprocess 서비스 시작 (백그라운드)
    try:
        root_dir = Path(__file__).parent.parent
        viz_dir = root_dir / "viz"
        preprocess_dir = root_dir / "preprocessing" / "texprep"
        
        # 환경변수 설정
        env_vars = os.environ.copy()
        env_vars["VIZ_PORT"] = "5005"
        env_vars["PREPROCESS_PORT"] = "5002"
        temp_dir = os.environ.get("TEMP", os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "Temp"))
        env_vars["HF_HOME"] = os.path.join(temp_dir, "hf_cache")
        env_vars["TRANSFORMERS_CACHE"] = os.path.join(temp_dir, "hf_cache")
        
        # Viz 서비스 시작
        if viz_dir.exists() and (viz_dir / "app.py").exists():
            print("🎨 Viz 서비스 시작 (포트: 5005)")
            try:
                subprocess.Popen(["python", "app.py"], cwd=str(viz_dir), env=env_vars)
                print("✅ Viz 서비스 시작 완료")
            except Exception as viz_error:
                print(f"❌ Viz 서비스 시작 실패: {viz_error}")
        else:
            print("⚠️ Viz 디렉토리 또는 app.py를 찾을 수 없습니다")
        
        # Preprocess 서비스 시작
        if preprocess_dir.exists() and (preprocess_dir / "app.py").exists():
            print("🔧 Preprocess 서비스 시작 (포트: 5002)")
            try:
                subprocess.Popen(["python", "app.py"], cwd=str(preprocess_dir), env=env_vars)
                print("✅ Preprocess 서비스 시작 완료")
            except Exception as preprocess_error:
                print(f"❌ Preprocess 서비스 시작 실패: {preprocess_error}")
        else:
            print("⚠️ Preprocess 디렉토리 또는 app.py를 찾을 수 없습니다")
            
    except Exception as e:
        print(f"⚠️ 내부 서비스 시작 실패: {e}")
        import traceback
        traceback.print_exc()
    
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
