import httpx

MODEL_ENDPOINTS = {
    "math": "http://math-llm:5001/predict",
    "summary": "http://summary-llm:5002/predict",
    "easy": "http://easy-llm:5003/predict"
}

class RemoteLLM:
    def __init__(self, model_type: str):
        self.url = MODEL_ENDPOINTS[model_type]
        self.model_type = model_type

    async def summarize(self, content: bytes) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(self.url, json={
                "text": content.decode("utf-8")
            })
            if response.status_code != 200:
                return f"[{self.model_type.upper()} 실패]"
            return response.json().get("summary", f"[{self.model_type.upper()} 응답 없음]")

def load_math_llm():
    return RemoteLLM("math")

def load_summary_llm():
    return RemoteLLM("summary")

def load_easy_llm():
    return RemoteLLM("easy")