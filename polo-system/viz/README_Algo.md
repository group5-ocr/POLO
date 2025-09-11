# AI Grammars

## 파일 구성

- `glossary_hybrid.json` — 한글, 영어 용어 사전 (지표·리소스 등 키워드 + 정규식)
- `text_to_spec.py` — 사전을 읽어 텍스트를 스펙(JSON)으로 변환
- `registry.py` — 문법(템플릿) 레지스트리
- `render.py` — 스펙을 받아 PNG로 렌더 -> 실행파일
- `grammars/` — 도형 문법 모듈
- `charts/` — 샘플 출력 이미지가 생성되는 폴더
- `switch.py` — 시각화 라벨 한/영 병기 스위치 유틸
- `generic_rules.py` — 개념/예시 도식 빌더 문법 모듈과 연결
- `patch_glossary.py` — 용어 사전 보강 패처(패치 끝나면 지워도 됨)

## 평가지표

분류/랭킹: accuracy, precision, recall, f1, auroc, (주의) auprc

검출/세분화: map_detection (mAP@[.5:.95]), average_recall (AR), miou, dice, pq

생성/복원: fid, kid, lpips, inception_score (IS는 데이터셋 의존 큼)

음성/문자 인식: wer, cer

## 실행

```bash
python app.py
```
