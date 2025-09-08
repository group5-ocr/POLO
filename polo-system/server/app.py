from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import convert

app = FastAPI(title="POLO Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.include_router(convert.router, prefix="/api")

@app.get("/health")
def health(): return {"status": "ok"}
