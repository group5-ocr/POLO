# OpenAI 기반
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def embed_text(text: str) -> list:
    """
    OpenAI API를 이용해 단일 텍스트를 임베딩합니다.
    :param text: 문자열
    :return: 임베딩 벡터 (list of floats)
    """
    try:
        response = openai.Embedding.create(
            model="text-embedding-3-small",  # 또는 text-embedding-ada-002
            input=text
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"[Embedding Error]: {e}")
        return []