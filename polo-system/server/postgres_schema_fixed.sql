-- PostgreSQL 스키마 수정 (SQLite와 동일하게)
-- 기존 테이블 삭제 (의존성 역순으로)
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS math_file CASCADE;
DROP TABLE IF EXISTS easy_file CASCADE;
DROP TABLE IF EXISTS tex CASCADE;
DROP TABLE IF EXISTS origin_file CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- 1) 사용자 테이블
CREATE TABLE users (
    user_id   SERIAL PRIMARY KEY,
    email     VARCHAR(255) NOT NULL UNIQUE,
    password  VARCHAR(255) NOT NULL,
    nickname  VARCHAR(100) NOT NULL,
    job       VARCHAR(100),
    create_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now()
);

-- 2) 원본 파일 테이블 (ORM 모델과 일치)
CREATE TABLE origin_file (
    origin_id SERIAL PRIMARY KEY,
    user_id   INT NOT NULL,
    filename  VARCHAR(255) NOT NULL,
    create_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now(),
    id        INT,

    CONSTRAINT fk_origin_user
      FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX idx_origin_user ON origin_file(user_id);

-- 3) tex 테이블 (ORM 모델과 일치)
CREATE TABLE tex (
    tex_id      SERIAL PRIMARY KEY,
    origin_id   INT NOT NULL,
    file_addr   VARCHAR(500),
    total_chunks INT NOT NULL DEFAULT 0,
    easy_done   INT NOT NULL DEFAULT 0,
    viz_done    INT NOT NULL DEFAULT 0,
    math_done   BOOLEAN NOT NULL DEFAULT FALSE,

    CONSTRAINT fk_tex_origin
      FOREIGN KEY (origin_id) REFERENCES origin_file(origin_id) ON DELETE CASCADE
);
CREATE INDEX idx_tex_origin ON tex(origin_id);

-- 4) easy_file 테이블 (ORM 모델과 일치)
CREATE TABLE easy_file (
    easy_id   SERIAL PRIMARY KEY,
    tex_id    INT NOT NULL,
    origin_id INT NOT NULL,
    filename  VARCHAR(255),
    file_addr VARCHAR(500),

    CONSTRAINT fk_easy_tex
      FOREIGN KEY (tex_id) REFERENCES tex(tex_id) ON DELETE CASCADE
);
CREATE INDEX idx_easy_tex ON easy_file(tex_id);

-- 5) math_file 테이블 (ORM 모델과 일치)
CREATE TABLE math_file (
    math_id   SERIAL PRIMARY KEY,
    tex_id    INT NOT NULL,
    origin_id INT NOT NULL,
    filename  VARCHAR(255),
    file_addr VARCHAR(500),

    CONSTRAINT fk_math_tex
      FOREIGN KEY (tex_id) REFERENCES tex(tex_id) ON DELETE CASCADE
);
CREATE INDEX idx_math_tex ON math_file(tex_id);

-- 6) chunks 테이블 (ORM 모델과 일치)
CREATE TABLE chunks (
    key            VARCHAR(255) PRIMARY KEY,
    paper_id       VARCHAR(255) NOT NULL,
    index          INT NOT NULL,
    text           TEXT,
    rewritten_text TEXT,
    image_path     VARCHAR(500)
);
CREATE INDEX idx_chunks_paper_id ON chunks(paper_id);

