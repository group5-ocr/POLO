# Papago 번역 설정 가이드

## 🚀 빠른 시작

### 1. Papago API 키 발급
1. [네이버 개발자 센터](https://developers.naver.com/apps/#/myapps) 접속
2. 로그인 후 "애플리케이션 등록" 클릭
3. 애플리케이션 이름 입력 (예: "POLO 번역")
4. 서비스 환경: "WEB" 선택
5. 서비스 URL: `http://localhost:5003` 입력
6. 등록 후 "Client ID"와 "Client Secret" 복사

### 2. 환경 변수 설정
```bash
# Windows PowerShell
$env:PAPAGO_CLIENT_ID="your_client_id_here"
$env:PAPAGO_CLIENT_SECRET="your_client_secret_here"

# 단일 번역 (영어 → 한국어)
$env:PAPAGO_USE_CHAIN="false"

# 체인 번역 (영어 → 일본어 → 한국어)
$env:PAPAGO_USE_CHAIN="true"
```

### 3. 자동 설정 스크립트 사용
```bash
python setup_papago.py
```

## 🔧 번역 방법 선택

### 단일 번역 (기본)
- **방법**: 영어 → 한국어
- **장점**: 빠름, API 호출 1회
- **단점**: 직접 번역으로 자연스러움 부족

### 체인 번역 (고급)
- **방법**: 영어 → 일본어 → 한국어
- **장점**: 더 자연스러운 한국어
- **단점**: 느림, API 호출 2회

## 🧪 테스트

### 간단한 테스트
```bash
python test_easy_simple.py
```

### 전체 논문 처리
```bash
python run_easy_processing.py
```

## 📊 결과 확인

### HTML 결과
- `server/data/outputs/transformer/easy_outputs_user/user_processing/easy_results.html`
- 브라우저에서 열어서 확인

### JSON 결과
- `server/data/outputs/transformer/easy_outputs_user/user_processing/easy_results.json`
- 원본 이미지와 번역 결과 포함

## 🔍 문제 해결

### API 키 오류
```
❌ Papago API 키가 설정되지 않음 → LLM 번역 사용
```
**해결**: 환경 변수 설정 확인

### 번역 실패
```
❌ Papago 번역 실패 → LLM 번역 사용
```
**해결**: API 키 유효성 확인, 네트워크 연결 확인

### 체인 번역 오류
```
❌ Papago 체인 번역 실패
```
**해결**: API 할당량 확인, 단일 번역으로 변경

## 💡 팁

1. **처음 사용**: 단일 번역으로 시작
2. **품질 중시**: 체인 번역 사용
3. **속도 중시**: 단일 번역 사용
4. **API 할당량**: 월 10,000자 무료 (네이버)

## 🔄 번역 방법 변경

```bash
# 단일 번역으로 변경
$env:PAPAGO_USE_CHAIN="false"

# 체인 번역으로 변경
$env:PAPAGO_USE_CHAIN="true"
```

변경 후 Easy 모델을 다시 실행하면 새로운 번역 방법이 적용됩니다.
