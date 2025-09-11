# PostgreSQL 데이터베이스 사용 가이드

이 문서는 POLO 프로젝트에서 PostgreSQL 데이터베이스를 사용하는 방법을 설명합니다.

## 📋 목차

1. [설치 및 설정](#설치-및-설정)
2. [환경 변수 구성](#환경-변수-구성)
3. [데이터베이스 연결 방법](#데이터베이스-연결-방법)
4. [API 사용 예제](#api-사용-예제)
5. [보안 및 인증](#보안-및-인증)
6. [테스트 및 디버깅](#테스트-및-디버깅)

## 🚀 설치 및 설정

### 1. 의존성 설치

```bash
cd polo-system/server
pip install -r requirements.api.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하여 데이터베이스 설정을 구성하세요:

```env
# PostgreSQL Database Configuration (외부 데이터베이스 사용)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=polo_db
POSTGRES_USER=polo_user
POSTGRES_PASSWORD=polo_password

# Hugging Face Token
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

**참고**: 이 프로젝트는 외부 PostgreSQL 데이터베이스를 사용합니다. Docker는 AI 모델만 실행합니다.

## 🔌 데이터베이스 연결 방법

### 1. SQLAlchemy ORM 사용

```python
from server.services.db import get_database
from sqlalchemy.orm import Session

def my_function(db: Session = Depends(get_database)):
    # 데이터베이스 작업 수행
    result = db.execute(text("SELECT * FROM users"))
    return result.fetchall()
```

### 2. psycopg2 직접 사용

```python
from server.services.db import get_postgres_client, PostgreSQLClient

def my_function(pg_client: PostgreSQLClient = Depends(get_postgres_client)):
    # 직접 SQL 쿼리 실행
    result = pg_client.execute_query("SELECT * FROM users")
    return result
```

### 3. 비동기 컨텍스트 매니저 사용

```python
from server.services.db import db_manager

async def my_async_function():
    async with db_manager.get_async_session() as session:
        # 비동기 데이터베이스 작업
        result = await session.execute(text("SELECT * FROM users"))
        return result.fetchall()
```

## 📡 API 사용 예제

### 1. 서버 실행

```bash
cd polo-system/server
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. API 엔드포인트

#### 데이터베이스 상태 확인

```bash
curl http://localhost:8000/db/health
```

#### 사용자 관리

```bash
# 사용자 목록 조회
curl http://localhost:8000/db/users

# 새 사용자 생성 (비밀번호 자동 해시화)
curl -X POST "http://localhost:8000/db/users" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123"
  }'

# 사용자 로그인 (비밀번호 검증)
curl -X POST "http://localhost:8000/db/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "password123"
  }'
```

#### 통계 정보 조회

```bash
curl http://localhost:8000/db/stats
```

## 🔐 보안 및 인증

### 1. 비밀번호 해시화

모든 사용자 비밀번호는 bcrypt를 사용하여 안전하게 해시화됩니다:

```python
from server.utils.password import hash_password, verify_password

# 비밀번호 해시화
hashed = hash_password("plain_password")

# 비밀번호 검증
is_valid = verify_password("plain_password", hashed)
```

### 2. 기존 사용자 비밀번호 해시화

기존 사용자들의 비밀번호는 회원가입 시 자동으로 해시화됩니다.
만약 기존 평문 비밀번호가 있다면, 해당 사용자가 다음 로그인 시 비밀번호를 재설정하도록 안내하세요.

### 3. 보안 모범 사례

- 비밀번호는 절대 평문으로 저장하지 않음
- bcrypt를 사용한 안전한 해시화
- 로그인 시 해시화된 비밀번호와 비교
- API 응답에서 비밀번호 필드 제외

## 🧪 테스트 및 디버깅

### 1. 데이터베이스 연결 테스트

```bash
cd polo-system/server
python -c "from services.db import db_manager; print('DB 연결 테스트:', db_manager.test_connection())"
```

### 2. 데이터베이스 직접 접속

```bash
# 로컬 PostgreSQL 클라이언트 사용
psql -h localhost -p 5432 -U polo_user -d polo_db
```

### 3. 서버 로그 확인

```bash
# 서버 실행 시 로그 확인
cd polo-system/server
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 📊 데이터베이스 스키마

### 주요 테이블

1. **users** - 사용자 정보

   - user_id (PK), email, password (해시화됨), nickname, job, create_at

2. **origin_file** - 원본 문서 정보

   - 파일 업로드 및 관리용 테이블

### 보안 고려사항

- **password** 필드는 bcrypt로 해시화되어 저장
- API 응답에서 비밀번호 필드는 제외
- 로그인 시 해시화된 비밀번호와 비교하여 인증

## 🔧 고급 설정

### 1. 연결 풀 설정

`server/services/db.py`에서 연결 풀 설정을 조정할 수 있습니다:

```python
self.engine = create_engine(
    self.config.database_url,
    poolclass=QueuePool,
    pool_size=10,        # 기본 연결 수
    max_overflow=20,     # 최대 추가 연결 수
    pool_pre_ping=True,  # 연결 상태 확인
    pool_recycle=3600,   # 연결 재사용 시간 (초)
)
```

### 2. 환경별 설정

개발/운영 환경에 따라 다른 설정을 사용할 수 있습니다:

```python
# 개발 환경
ENVIRONMENT=development

# 운영 환경
ENVIRONMENT=production
```

### 3. 백업 및 복원

```bash
# 데이터베이스 백업
pg_dump -h localhost -p 5432 -U polo_user polo_db > backup.sql

# 데이터베이스 복원
psql -h localhost -p 5432 -U polo_user polo_db < backup.sql
```

## 🚨 문제 해결

### 1. 연결 실패

- 환경 변수 설정 확인
- PostgreSQL 서비스 상태 확인
- 방화벽 설정 확인

### 2. 권한 오류

- 사용자 권한 확인
- 데이터베이스 접근 권한 확인

### 3. 비밀번호 해시화 오류

- bcrypt 패키지 설치 확인: `pip install bcrypt==4.1.2`
- 기존 사용자 비밀번호 해시화: `python hash_existing_passwords.py`

### 4. 성능 문제

- 연결 풀 설정 조정
- 인덱스 최적화
- 쿼리 성능 분석

## 📚 추가 자료

- [PostgreSQL 공식 문서](https://www.postgresql.org/docs/)
- [SQLAlchemy 문서](https://docs.sqlalchemy.org/)
- [FastAPI 데이터베이스 가이드](https://fastapi.tiangolo.com/tutorial/sql-databases/)
