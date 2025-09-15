# server/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.database.db import DB

from routes import upload, generate, results, math_generate

app = FastAPI(title="POLO Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy", "db_mode": DB.mode}

@app.on_event("startup")
async def startup():
    await DB.init()
    print(f"현재 DB 모드: {DB.mode}")

@app.on_event("shutdown")
async def shutdown():
    close_fn = getattr(DB, "close", None)
    if callable(close_fn):
        await close_fn()

# API 엔드포인트
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(generate.router, prefix="/api", tags=["generate"])
app.include_router(results.router, prefix="/api", tags=["results"])
app.include_router(math_generate.router, prefix="/api", tags=["math"])