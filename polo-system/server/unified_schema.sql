-- 통합 스키마 (PostgreSQL과 SQLite 모두 지원)
-- 기존 테이블 삭제 (의존성 역순으로)
DROP TABLE IF EXISTS visualizations CASCADE;
DROP TABLE IF EXISTS math_equations CASCADE;
DROP TABLE IF EXISTS easy_paragraphs CASCADE;
DROP TABLE IF EXISTS easy_sections CASCADE;
DROP TABLE IF EXISTS integrated_results CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS math_file CASCADE;
DROP TABLE IF EXISTS easy_file CASCADE;
DROP TABLE IF EXISTS tex CASCADE;
DROP TABLE IF EXISTS origin_file CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- PostgreSQL 확장 활성화 (SQLite에서는 무시됨)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1) 사용자 테이블
CREATE TABLE users (
    user_id   SERIAL PRIMARY KEY,
    email     VARCHAR(255) NOT NULL UNIQUE,
    password  VARCHAR(255) NOT NULL,
    nickname  VARCHAR(100) NOT NULL,
    job       VARCHAR(100),
    create_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2) 원본 파일 테이블
CREATE TABLE origin_file (
    origin_id SERIAL PRIMARY KEY,
    user_id   INTEGER NOT NULL,
    filename  VARCHAR(255) NOT NULL,
    create_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    id        INTEGER,

    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
CREATE INDEX idx_origin_user ON origin_file(user_id);

-- 3) tex 테이블 (통합 결과 지원)
CREATE TABLE tex (
    tex_id      SERIAL PRIMARY KEY,
    origin_id   INTEGER NOT NULL,
    file_addr   VARCHAR(500),
    total_chunks INTEGER NOT NULL DEFAULT 0,
    easy_done   INTEGER NOT NULL DEFAULT 0,
    viz_done    INTEGER NOT NULL DEFAULT 0,
    math_done   BOOLEAN NOT NULL DEFAULT FALSE,
    integrated_done BOOLEAN NOT NULL DEFAULT FALSE,
    integrated_result_id INTEGER,

    FOREIGN KEY (origin_id) REFERENCES origin_file(origin_id) ON DELETE CASCADE
);
CREATE INDEX idx_tex_origin ON tex(origin_id);
CREATE INDEX idx_tex_integrated ON tex(integrated_done);

-- 4) easy_file 테이블
CREATE TABLE easy_file (
    easy_id   SERIAL PRIMARY KEY,
    tex_id    INTEGER NOT NULL,
    origin_id INTEGER NOT NULL,
    filename  VARCHAR(255),
    file_addr VARCHAR(500),

    FOREIGN KEY (tex_id) REFERENCES tex(tex_id) ON DELETE CASCADE,
    FOREIGN KEY (origin_id) REFERENCES origin_file(origin_id) ON DELETE CASCADE
);
CREATE INDEX idx_easy_tex ON easy_file(tex_id);
CREATE INDEX idx_easy_origin ON easy_file(origin_id);

-- 5) math_file 테이블
CREATE TABLE math_file (
    math_id   SERIAL PRIMARY KEY,
    tex_id    INTEGER NOT NULL,
    origin_id INTEGER NOT NULL,
    filename  VARCHAR(255),
    file_addr VARCHAR(500),

    FOREIGN KEY (tex_id) REFERENCES tex(tex_id) ON DELETE CASCADE,
    FOREIGN KEY (origin_id) REFERENCES origin_file(origin_id) ON DELETE CASCADE
);
CREATE INDEX idx_math_tex ON math_file(tex_id);
CREATE INDEX idx_math_origin ON math_file(origin_id);

-- 6) 통합 결과 테이블
CREATE TABLE integrated_results (
    result_id SERIAL PRIMARY KEY,
    tex_id INTEGER NOT NULL,
    origin_id INTEGER NOT NULL,
    paper_id VARCHAR(255) NOT NULL,
    paper_title VARCHAR(500),
    paper_authors TEXT,
    paper_venue VARCHAR(255),
    total_sections INTEGER DEFAULT 0,
    total_equations INTEGER DEFAULT 0,
    total_visualizations INTEGER DEFAULT 0,
    processing_status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (tex_id) REFERENCES tex(tex_id) ON DELETE CASCADE,
    FOREIGN KEY (origin_id) REFERENCES origin_file(origin_id) ON DELETE CASCADE
);
CREATE INDEX idx_integrated_paper_id ON integrated_results(paper_id);
CREATE INDEX idx_integrated_tex ON integrated_results(tex_id);
CREATE INDEX idx_integrated_status ON integrated_results(processing_status);

-- 7) Easy 섹션 테이블
CREATE TABLE easy_sections (
    section_id SERIAL PRIMARY KEY,
    integrated_result_id INTEGER NOT NULL,
    easy_section_id VARCHAR(255) NOT NULL,
    easy_section_title VARCHAR(500),
    easy_section_type VARCHAR(50),
    easy_section_order INTEGER,
    easy_section_level INTEGER,
    easy_section_parent VARCHAR(255),
    easy_content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (integrated_result_id) REFERENCES integrated_results(result_id) ON DELETE CASCADE
);
CREATE INDEX idx_easy_sections_result ON easy_sections(integrated_result_id);
CREATE INDEX idx_easy_sections_order ON easy_sections(easy_section_order);

-- 8) Easy 문단 테이블
CREATE TABLE easy_paragraphs (
    paragraph_id SERIAL PRIMARY KEY,
    section_id INTEGER NOT NULL,
    easy_paragraph_id VARCHAR(255) NOT NULL,
    easy_paragraph_text TEXT,
    easy_visualization_trigger BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (section_id) REFERENCES easy_sections(section_id) ON DELETE CASCADE
);
CREATE INDEX idx_easy_paragraphs_section ON easy_paragraphs(section_id);
CREATE INDEX idx_easy_paragraphs_trigger ON easy_paragraphs(easy_visualization_trigger);

-- 9) Math 수식 테이블
CREATE TABLE math_equations (
    equation_id SERIAL PRIMARY KEY,
    integrated_result_id INTEGER NOT NULL,
    math_equation_id VARCHAR(255) NOT NULL,
    math_equation_latex TEXT,
    math_equation_explanation TEXT,
    math_equation_section_ref VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (integrated_result_id) REFERENCES integrated_results(result_id) ON DELETE CASCADE
);
CREATE INDEX idx_math_equations_result ON math_equations(integrated_result_id);
CREATE INDEX idx_math_equations_section ON math_equations(math_equation_section_ref);

-- 10) 시각화 테이블 (JSON 타입 통일)
CREATE TABLE visualizations (
    viz_id SERIAL PRIMARY KEY,
    integrated_result_id INTEGER NOT NULL,
    section_id INTEGER,
    paragraph_id INTEGER,
    viz_type VARCHAR(50),
    viz_title VARCHAR(500),
    viz_description TEXT,
    viz_image_path VARCHAR(500),
    viz_metadata TEXT,  -- PostgreSQL과 SQLite 모두에서 TEXT로 통일
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (integrated_result_id) REFERENCES integrated_results(result_id) ON DELETE CASCADE,
    FOREIGN KEY (section_id) REFERENCES easy_sections(section_id) ON DELETE SET NULL,
    FOREIGN KEY (paragraph_id) REFERENCES easy_paragraphs(paragraph_id) ON DELETE SET NULL
);
CREATE INDEX idx_viz_result ON visualizations(integrated_result_id);
CREATE INDEX idx_viz_type ON visualizations(viz_type);
CREATE INDEX idx_viz_section ON visualizations(section_id);

-- 11) chunks 테이블 (easy_sections 이후에 생성)
CREATE TABLE chunks (
    key            VARCHAR(255) PRIMARY KEY,
    paper_id       VARCHAR(255) NOT NULL,
    index          INTEGER NOT NULL,
    text           TEXT,
    rewritten_text TEXT,
    image_path     VARCHAR(500),
    section_id     INTEGER,

    FOREIGN KEY (section_id) REFERENCES easy_sections(section_id) ON DELETE SET NULL
);
CREATE INDEX idx_chunks_paper_id ON chunks(paper_id);
CREATE INDEX idx_chunks_index ON chunks(index);
CREATE INDEX idx_chunks_section ON chunks(section_id);

-- 완료 메시지
SELECT '통합 스키마 생성 완료!' as status;
