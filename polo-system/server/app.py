# server/app.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.database.db import DB

from routes import upload, generate, results, math_generate, database

# .env 파일 로드
load_dotenv()

app = FastAPI(title="POLO Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    await DB.init()
    print(f"현재 DB 모드: {DB.mode}")

@app.on_event("shutdown")
async def shutdown():
    close_fn = getattr(DB, "close", None)
    if callable(close_fn):
        await close_fn()

app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(generate.router, prefix="/generate", tags=["callbacks"])
app.include_router(results.router, prefix="/results", tags=["results"])
app.include_router(math_generate.router, prefix="/math", tags=["math"])
app.include_router(database.router, prefix="/db", tags=["database"])

# API 엔드포인트 추가
app.include_router(upload.router, prefix="/api", tags=["api"])
app.include_router(generate.router, prefix="/api", tags=["api"])
app.include_router(results.router, prefix="/api", tags=["api"])
app.include_router(math_generate.router, prefix="/api", tags=["api"])