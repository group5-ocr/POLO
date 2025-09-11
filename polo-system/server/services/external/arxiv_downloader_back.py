#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
arXiv PDF/소스 일괄 처리 유틸리티

기능 개요
1) (선택) 기존 PDF의 왼쪽 수직 여백/좌측영역/전체 텍스트에서 arXiv ID 추출 (버전 제거)
2) arXiv PDF와 e-print(tar.gz) 다운로드
   - 우선 arxiv 라이브러리 이용(urllib 기반, 전역 SSL 컨텍스트 주입)
   - 실패 시 requests+certifi로 폴백
3) e-print(tar.gz) 안전 추출 → .tex 스캔 → 메인 .tex 추정 → 상위 N줄 미리보기

사용 예시
- ID 직접 지정:
    python arxiv_downloader_tester.py --id 1706.03762
- 기존 PDF에서 ID 추출하여 처리:
    python arxiv_downloader_tester.py --pdf "C:\\requirement\\YOLO_v4.pdf"
- 회사/학교 자체 CA 사용:
    python arxiv_downloader_tester.py --id 1706.03762 --corp-ca "C:\\certs\\corp-ca.pem"
"""

from __future__ import annotations
import argparse
import os
import re
import ssl
import tarfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import certifi
import requests

# PyMuPDF
import fitz  # type: ignore

# arxiv 라이브러리 (urllib 사용)
import urllib.request
import arxiv  # type: ignore


# ===== 공용 정규식: arXiv ID (버전 포함/clean 그룹) =====
ARXIV_PAT = re.compile(r"arXiv:(\d{4}\.\d{4,5})(?:v\d+)?", re.I)


# ===== 1) PDF에서 arXiv ID 추출 =====
def extract_arxiv_id_from_pdf(pdf_path: Path, left_margin_px: int = 120) -> Optional[str]:
    """
    PDF 경로에서 arXiv ID(숫자만, 버전 제거)를 추출합니다.
    우선순위: (1) 왼쪽 '수직' 텍스트 → (2) 왼쪽 여백 전체 텍스트 → (3) 페이지 전체 텍스트
    찾지 못하면 None 반환
    """
    def extract_vertical_text_from_left_margin(page, left_margin_px=120) -> str:
        data = page.get_text("dict")
        vertical_spans = []
        for block in data.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    dx, dy = span.get("dir", (1, 0))
                    sx0, _, _, _ = span.get("bbox", (0, 0, 0, 0))
                    is_left = sx0 <= left_margin_px
                    is_vertical = abs(dx) < 0.5 and abs(dy) > 0.5
                    if is_left and is_vertical:
                        vertical_spans.append(span)
        if not vertical_spans:
            return ""
        vertical_spans.sort(key=lambda s: (round(s["bbox"][0], 1), s["bbox"][1]))
        return re.sub(r"\s+", "", "".join(s["text"] for s in vertical_spans))

    doc = fitz.open(pdf_path)
    try:
        page = doc[0]

        # 1) 왼쪽 수직 텍스트
        raw_vertical = extract_vertical_text_from_left_margin(page, left_margin_px)
        m = ARXIV_PAT.search(raw_vertical)
        if m:
            return m.group(1)

        # 2) 왼쪽 여백 전체 텍스트
        left_text = page.get_text("text", clip=fitz.Rect(0, 0, left_margin_px, page.rect.height))
        m2 = ARXIV_PAT.search(left_text or "")
        if m2:
            return m2.group(1)

        # 3) 페이지 전체 텍스트
        full_text = page.get_text("text")
        m3 = ARXIV_PAT.search(full_text or "")
        if m3:
            return m3.group(1)

        return None
    finally:
        doc.close()


# ===== 2) urllib 전역 SSL 컨텍스트 교체 =====
def install_global_urllib_ssl_context(corp_ca_pem: Optional[str]) -> None:
    if corp_ca_pem and os.path.exists(corp_ca_pem):
        ctx = ssl.create_default_context(cafile=corp_ca_pem)
    else:
        ctx = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    urllib.request.install_opener(opener)


# ===== 3) 다운로드 단계 =====
def download_with_arxiv_lib(arxiv_id: str, pdf_path: Path, src_tar: Path) -> None:
    """arxiv 패키지의 download_* API 사용 (urllib 기반)"""
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(search.results())

    print("[arxiv] PDF 다운로드 중…")
    result.download_pdf(filename=str(pdf_path))
    print("  →", pdf_path)

    print("[arxiv] 원본 소스(tar.gz) 다운로드 중…")
    result.download_source(filename=str(src_tar))
    print("  →", src_tar)


def download_direct(arxiv_id: str, pdf_path: Path, src_tar: Path, corp_ca_pem: Optional[str]) -> None:
    """requests+certifi로 직접 URL에서 다운로드 (폴백 플랜)"""
    verify_arg = corp_ca_pem if (corp_ca_pem and os.path.exists(corp_ca_pem)) else certifi.where()

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    src_url = f"https://arxiv.org/e-print/{arxiv_id}"  # tar.gz

    print("[requests] PDF 다운로드 중…", pdf_url)
    with requests.get(pdf_url, stream=True, timeout=60, verify=verify_arg) as r:
        r.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in r.iter_content(1024 * 64):
                if chunk:
                    f.write(chunk)
    print("  →", pdf_path)

    print("[requests] 원본 소스(tar.gz) 다운로드 중…", src_url)
    with requests.get(src_url, stream=True, timeout=60, verify=verify_arg) as r:
        r.raise_for_status()
        with open(src_tar, "wb") as f:
            for chunk in r.iter_content(1024 * 64):
                if chunk:
                    f.write(chunk)
    print("  →", src_tar)


# ===== 4) e-print 안전 추출 & 메인 tex 추정 =====
def safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="r:*") as tar:
        def is_within_directory(directory: str, target: str) -> bool:
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

        for member in tar.getmembers():
            member_path = os.path.join(out_dir, member.name)
            if not is_within_directory(str(out_dir), member_path):
                raise RuntimeError(f"경로 탈출 의심 항목: {member.name}")
        tar.extractall(out_dir)


def find_all_tex(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.tex") if p.is_file()]


def guess_main_tex(tex_files: List[Path]) -> Optional[Path]:
    """
    메인 tex 추정 규칙:
    1) 이름 우선: main/ms/paper/arxiv/root.tex
    2) 내용 가산점: \documentclass(-2), \begin{document}(-1)
    3) 경로가 짧을수록 선호
    """
    if not tex_files:
        return None

    priority = ["main.tex", "ms.tex", "paper.tex", "arxiv.tex", "root.tex"]
    name_rank = {n: i for i, n in enumerate(priority)}

    best: Tuple[int, int, int, Path] = (999, 999, 10**9, tex_files[0])
    for p in tex_files:
        name_score = name_rank.get(p.name.lower(), 999)

        content_score = 50  # 기본 패널티
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            score = 0
            if re.search(r"\\documentclass", text):
                score -= 2
            if re.search(r"\\begin\{document\}", text):
                score -= 1
            content_score = score if score != 0 else 10
        except Exception:
            pass

        path_len = len(str(p))
        cand = (name_score, content_score, path_len, p)
        if cand < best:
            best = cand

    return best[3]


# ===== 5) 메인 실행 루틴 =====
def run(
    arxiv_id: Optional[str],
    pdf_to_extract_id: Optional[Path],
    out_root: Path,
    corp_ca_pem: Optional[str],
    left_margin_px: int,
    preview_lines: int,
    clean_extract_dir: bool,
) -> None:
    # 0) arXiv ID 결정
    if not arxiv_id:
        if not pdf_to_extract_id:
            raise ValueError("arXiv ID를 직접 지정하시거나(--id), 기존 PDF에서 추출할 경로를 제공해 주세요(--pdf).")
        print(f"[INFO] PDF에서 arXiv ID 추출 시도: {pdf_to_extract_id}")
        arxiv_id = extract_arxiv_id_from_pdf(pdf_to_extract_id, left_margin_px)
        if not arxiv_id:
            raise RuntimeError("PDF에서 arXiv ID를 찾지 못했습니다.")
        print(f"[OK] 추출된 ID: {arxiv_id}")

    # 1) 출력 경로 준비
    out_dir = out_root / arxiv_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{arxiv_id}.pdf"
    src_tar = out_dir / f"{arxiv_id}_source.tar.gz"

    print(f"[INFO] 대상 ID   : {arxiv_id}")
    print(f"[INFO] 대상 폴더 : {out_dir.resolve()}")

    # 2) urllib 전역 SSL 컨텍스트 설치
    install_global_urllib_ssl_context(corp_ca_pem)

    # 3) 다운로드: arxiv 라이브러리 → requests 폴백
    try:
        download_with_arxiv_lib(arxiv_id, pdf_path, src_tar)
    except Exception as e:
        print("⚠️ arxiv 라이브러리 방식 실패:", repr(e))
        print("→ requests+certifi로 직접 다운로드를 시도합니다.")
        download_direct(arxiv_id, pdf_path, src_tar, corp_ca_pem)

    print("\n✅ 다운로드 완료")
    print("PDF :", pdf_path.resolve())
    print("SRC :", src_tar.resolve())

    # 4) e-print 해제 및 .tex 탐색
    src_out = out_dir / "source"
    if src_out.exists() and clean_extract_dir:
        shutil.rmtree(src_out)

    print("\n[INFO] 압축 해제 중…")
    safe_extract_tar(src_tar, src_out)
    print(f"[OK]  추출 완료: {src_out.resolve()}")

    tex_list = find_all_tex(src_out)
    print(f"[INFO] 찾은 .tex 개수: {len(tex_list)}")
    if not tex_list:
        print("⚠️ .tex 파일이 보이지 않습니다. 압축 구조를 확인해 주세요.")
        return

    # (참고) 경로가 짧은 순으로 샘플 10개 표시
    for i, p in enumerate(sorted(tex_list, key=lambda x: len(str(x)))[:10], 1):
        try:
            rel = p.relative_to(src_out)
        except ValueError:
            rel = p
        print(f"  - 샘플{i}: {rel}")

    main_tex = guess_main_tex(tex_list)
    if main_tex:
        print(f"\n✅ 메인 .tex 추정: {main_tex.resolve()}")
        try:
            lines = main_tex.read_text(encoding="utf-8", errors="ignore").splitlines()
            preview = "\n".join(lines[:preview_lines])
            print(f"\n--- 메인 .tex 미리보기 (상위 {preview_lines}줄) ---")
            print(preview)
            print("-----------------------------------------------")
        except Exception as e:
            print(f"(미리보기 실패: {e})")
    else:
        print("⚠️ 메인 .tex를 추정하지 못했습니다. 파일명을 확인해 주세요.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="arXiv PDF/소스 다운로드 및 e-print 해제·메인 .tex 추정 유틸리티")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--id", help="arXiv ID (예: 1706.03762)")
    src.add_argument("--pdf", type=Path, help="기존 PDF 경로에서 arXiv ID 추출")

    p.add_argument("--out-root", type=Path, default=Path("./arxiv_files"),
                   help="출력 루트 폴더 (기본: ./arxiv_files)")
    p.add_argument("--corp-ca", type=Path, default=None,
                   help="회사/학교 자체 CA PEM 경로 (선택)")
    p.add_argument("--left-margin", type=int, default=120,
                   help="PDF 왼쪽 여백(px) 판정값 (기본: 120)")
    p.add_argument("--preview-lines", type=int, default=40,
                   help="메인 .tex 미리보기 줄 수 (기본: 40)")
    p.add_argument("--keep-extract", action="store_true",
                   help="이미 존재하는 source 폴더 유지(기본은 재생성)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(
            arxiv_id=args.id,
            pdf_to_extract_id=args.pdf,
            out_root=args.out_root,
            corp_ca_pem=str(args.corp_ca) if args.corp_ca else None,
            left_margin_px=args.left_margin,
            preview_lines=args.preview_lines,
            clean_extract_dir=not args.keep_extract,
        )
    except Exception as e:
        print("❌ 오류:", e)
        raise

# python arxiv_downloader_tester.py --id 1706.03762