# AI Grammars (한국어 주석 풀버전)

## 폴더 구성

- `glossary_hybrid.json` — 영어 용어 사전 (지표·리소스 등 키워드 + 정규식)
- `text_to_spec.py` — 사전을 읽어 텍스트를 스펙(JSON)으로 변환
- `registry.py` — 문법(템플릿) 레지스트리
- `render.py` — 스펙을 받아 PNG로 렌더
- `grammars/` — 10개 도형 문법 모듈
- `out/` — 샘플 출력 이미지가 생성되는 폴더

## 실행

```bash
python render.py
```
