# ChromaDB를 이용한 벡터 저장 및 관리
import chromadb
from chromadb.config import Settings

# ChromaDB 클라이언트 설정
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chromadb_store"
))

collection = client.get_or_create_collection(name="papers")

def store_vector(user_id: int, original_text: str, embedding: list, metadata: dict):
    """
    벡터 DB에 임베딩 벡터를 저장합니다.
    :param user_id: 사용자 ID
    :param original_text: 원본 텍스트
    :param embedding: 임베딩 벡터 (리스트)
    :param metadata: dict 형식의 메타데이터 (예: {"filename": "논문.pdf"})
    """
    vector_id = f"{user_id}_{metadata.get('filename', 'no_filename')}"
    try:
        collection.add(
            documents=[original_text],
            embeddings=[embedding],
            ids=[vector_id],
            metadatas=[metadata]
        )
    except Exception as e:
        print(f"[ChromaDB Error]: {e}")
