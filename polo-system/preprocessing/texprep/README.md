# README.md (전처리 모듈)

## 위치

- 경로: `polo-system/preprocessing/texprep/`
- 이 모듈은 문서(TeX 기반 논문 등)를 전처리하여 검색 친화적인 산출물을 생성한다.

---

## 실행 방식

- 각자 전처리된 논문으로 처리 확인(테스트)할 때 아래와 같이 해주시면 됩니다.

### 로컬 테스트 (단독 실행)

```bash
python app.py --input <파일경로> --output <저장경로>
```

- `--input`: 처리할 파일 경로
- `--output`: 결과 저장 디렉토리

### 서비스 환경

- 실제 서비스에서는 \*\*`polo-system/server/services/preprocess_client.py`\*\*가 이 모듈을 호출한다.
- 따라서 운영 시 `app.py`는 직접 실행되지 않고, **테스트·디버깅 용도**로만 사용한다.
- 산출물 저장 경로는 `preprocess_client.py`에서 지정되며, 도커 볼륨 경로로 매핑된다.

---

## 산출물

산출물은 `<output>/<doc_id>/` 디렉토리에 저장된다.

- **chunks.jsonl**

  - 문서 청크 단위 분할 결과
  - **활용도**: RAG 검색 인덱싱 핵심

- **merged_body.tex**

  - 병합된 TeX 원문
  - **활용도**: 원문 재가공, 청크 생성 근거

- **xref_edges.jsonl / xref_mentions.jsonl**

  - 본문 내 참조 관계(Fig.1, Eq.(2) 등)
  - **활용도**: 레퍼런스 검색, 본문-참조 연결 유지

- **equations.jsonl**

  - 추출된 수식
  - **활용도**: 수학 특화 검색(RAG 확장 시)

- **assets.jsonl / transport.json**

  - 이미지·표 등 부가 자원 메타데이터
  - **활용도**: 멀티모달 검색(현재는 사용 안 함)

---

## 페이로드 구조 (예시: chunks.jsonl)

```json
{
  "doc_id": "BERT_2018",
  "chunk_id": 12,
  "text": "청크 본문 텍스트",
  "para_index": 43,
  "sent_index": 0,
  "tags": ["abstract", "introduction"]
}
```

- `doc_id`: 문서 ID
- `chunk_id`: 청크 고유 ID
- `text`: 본문 텍스트
- `para_index`, `sent_index`: 원문 내 위치
- `tags`: 의미적 구분(섹션 등)

---

## 정리

- **핵심 산출물**: `chunks.jsonl`, `merged_body.tex`, `xref_*`
- **보류 산출물**: `equations.jsonl`, `assets.jsonl`, `transport.json` → 향후 내용 검색 확장 시 활용
- **책임 경계**: 본 모듈은 전처리만 담당, 서비스 병합 및 운영은 `preprocess_client.py`에서 처리

---
