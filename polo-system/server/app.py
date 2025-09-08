from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import convert

app = FastAPI(title="POLO Inference API")

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우트 등록
app.include_router(convert.router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "ok"}