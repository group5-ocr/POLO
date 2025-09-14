# server/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.database import db as DB
from routes import upload, generate, results

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

app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(generate.router, prefix="/generate", tags=["callbacks"])
app.include_router(results.router, prefix="/results", tags=["results"])