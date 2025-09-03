from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class InferenceRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: InferenceRequest):
    # 여기에 HuggingFace 파인튜닝된 모델 로딩 & 추론 로직 넣으면 됨
    return {"summary": f"[요약결과]\n{request.text[:300]}..."}
