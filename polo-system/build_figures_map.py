# -*- coding: utf-8 -*-
"""
PDF → PNG 렌더링 및 사이드카 맵 생성
통합 JSON 구조를 건드리지 않고 별도 figures_map.json으로 Figure 정보 제공
"""
from __future__ import annotations
import json
import hashlib
import shutil
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️ PyMuPDF not available. Install with: pip install pymupdf")

# ==== [환경 경로 설정] ========================================================
ASSETS_JSONL = Path(r"C:\POLO\POLO\polo-system\server\data\out\source\assets.jsonl")
SOURCE_DIR   = Path(r"C:\POLO\POLO\polo-system\server\data\arxiv\1506.02640\source")
# PNG와 정적 서빙 기준 루트(= /static 에 마운트할 루트)
STATIC_ROOT  = Path(r"C:\POLO\POLO\polo-system\server\data\outputs")

# 결과물 저장 위치
FIG_ROOT     = STATIC_ROOT / "viz" / "figures"            # PNG 보관 디렉터리
OUT_INDEX    = STATIC_ROOT / "viz" / "figures_map.json"   # 사이드카 맵(JSON)

# (선택) 안내 파일
NOTE_PATH    = STATIC_ROOT / "viz" / "README_figures.txt"
# ============================================================================


def md5_10(s: str) -> str:
    """문자열을 10자리 MD5 해시로 변환"""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def render_pdf_to_pngs(pdf_path: Path, out_dir: Path, dpi: int = 200) -> list[Path]:
    """
    PDF 파일을 페이지별로 PNG로 렌더링 (중복 방지)
    
    Args:
        pdf_path: PDF 파일 경로
        out_dir: 출력 디렉터리
        dpi: 렌더링 해상도
        
    Returns:
        생성된 PNG 파일 경로 리스트
    """
    if not PYMUPDF_AVAILABLE:
        print(f"⚠️ PyMuPDF 없음, PDF 렌더링 스킵: {pdf_path}")
        return []
    
    if not pdf_path.exists():
        print(f"⚠️ PDF 파일 없음: {pdf_path}")
        return []
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        doc = fitz.open(str(pdf_path))
        outs: list[Path] = []
        
        for i, page in enumerate(doc):
            out_path = out_dir / f"{pdf_path.stem}_p{i+1}.png"
            
            # 이미 존재하면 재생성하지 않음 (중복 방지)
            if not out_path.exists():
                try:
                    pix = page.get_pixmap(dpi=dpi)
                    pix.save(str(out_path))
                    print(f"✅ [PDF] 페이지 렌더링: {out_path.name}")
                except Exception as e:
                    print(f"⚠️ [PDF] 페이지 렌더링 실패 {i+1}: {e}")
                    continue
            else:
                print(f"✅ [PDF] 파일 이미 존재, 재생성 스킵: {out_path.name}")
            
            outs.append(out_path)
        
        doc.close()
        return outs
        
    except Exception as e:
        print(f"❌ [PDF] 문서 열기 실패: {pdf_path} - {e}")
        return []


def copy_if_needed(src: Path, dst_dir: Path) -> Path:
    """
    이미지 파일을 출력 디렉터리로 복사 (중복 방지)
    
    Args:
        src: 원본 파일 경로
        dst_dir: 출력 디렉터리
        
    Returns:
        복사된 파일 경로
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    
    if not dst.exists():
        try:
            shutil.copyfile(src, dst)
            print(f"✅ [IMG] 이미지 복사: {dst.name}")
        except Exception as e:
            print(f"⚠️ [IMG] 이미지 복사 실패: {e}")
    else:
        print(f"✅ [IMG] 파일 이미 존재, 재복사 스킵: {dst.name}")
    
    return dst


def to_web_url(p: Path) -> str:
    """
    Windows 경로를 /static 기준 웹 URL로 변환
    
    Args:
        p: 파일 경로
        
    Returns:
        웹 접근 가능한 URL
    """
    p = p.resolve()
    try:
        rel = p.relative_to(STATIC_ROOT.resolve())
        return "/static/" + str(rel).replace("\\", "/")
    except ValueError:
        # STATIC_ROOT와 관계없는 경로인 경우
        return str(p).replace("\\", "/")


def find_graphics_file(graphics: str, source_dir: Path) -> Path | None:
    """
    LaTeX graphics 파일명으로 실제 파일 찾기 (확장자 생략 대응)
    
    Args:
        graphics: LaTeX에서 참조한 그래픽 파일명
        source_dir: 소스 디렉터리
        
    Returns:
        찾은 파일 경로 또는 None
    """
    # 직접 경로로 시도
    src = (source_dir / graphics).resolve()
    if src.exists():
        return src
    
    # 확장자 생략된 경우 검색 (PDF 우선)
    stem = Path(graphics).stem
    extensions = [".pdf", ".png", ".jpg", ".jpeg", ".eps", ".ps"]
    
    for ext in extensions:
        candidate = source_dir / f"{stem}{ext}"
        if candidate.exists():
            print(f"✅ [FIND] 그래픽 파일 발견: {graphics} → {candidate.name}")
            return candidate
    
    # 와일드카드 검색
    candidates = list(source_dir.glob(f"{stem}.*"))
    if candidates:
        print(f"✅ [FIND] 그래픽 파일 발견 (glob): {graphics} → {candidates[0].name}")
        return candidates[0]
    
    print(f"⚠️ [FIND] 그래픽 파일 없음: {graphics}")
    return None


def main():
    """메인 실행 함수"""
    # 입력 파일 검증
    if not ASSETS_JSONL.exists():
        print(f"❌ assets.jsonl 없음: {ASSETS_JSONL}")
        return
    
    if not SOURCE_DIR.exists():
        print(f"❌ source 폴더 없음: {SOURCE_DIR}")
        return
    
    # 출력 디렉터리 생성
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    
    figures = []
    order = 0
    
    print(f"📖 [START] assets.jsonl 파싱: {ASSETS_JSONL}")
    
    try:
        with ASSETS_JSONL.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON 파싱 오류 (line {line_num}): {e}")
                    continue
                
                # figure 환경만 처리
                env = str(obj.get("env", "")).lower()
                if env not in ("figure", "figure*"):
                    continue
                
                graphics = obj.get("graphics")
                if not graphics:
                    print(f"⚠️ graphics 필드 없음 (line {line_num})")
                    continue
                
                print(f"🔍 [PROCESS] Figure 처리: {graphics}")
                
                # 그래픽 원본 파일 찾기
                src = find_graphics_file(graphics, SOURCE_DIR)
                if not src:
                    continue
                
                # 출력 디렉터리: figures/<basename>/
                out_dir = FIG_ROOT / Path(graphics).stem
                
                # PDF → PNG(s) / 이미지면 복사
                png_paths: list[Path]
                if src.suffix.lower() == ".pdf":
                    png_paths = render_pdf_to_pngs(src, out_dir, dpi=220)
                else:
                    single = copy_if_needed(src, out_dir)
                    png_paths = [single] if single.exists() else []
                
                if not png_paths:
                    print(f"⚠️ PNG 생성/복사 실패: {graphics}")
                    continue
                
                # 중복 방지용 버전 해시 (label+caption+src 기준)
                key = obj.get("label") or Path(graphics).stem
                cap = obj.get("caption") or ""
                h = md5_10(f"{key}|{cap}|{src.as_posix()}")
                
                # 대표 이미지는 첫 번째 페이지 (멀티페이지는 all_pages에 전부 포함)
                main_png = png_paths[0]
                order += 1
                
                item = {
                    "order": order,
                    "label": obj.get("label"),
                    "caption": cap,
                    "graphics": graphics,
                    "src_file": str(src),
                    "image_path": to_web_url(main_png) + f"?v={h}",
                    "all_pages": [to_web_url(p) + f"?v={h}" for p in png_paths],
                    "hash": h
                }
                figures.append(item)
                
                print(f"✅ [ADDED] Figure {order}: {graphics} ({len(png_paths)} pages)")
    
    except Exception as e:
        print(f"❌ [ERROR] assets.jsonl 처리 실패: {e}")
        return
    
    # 사이드카 맵 저장 (문서 등장 순서대로)
    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    
    sidecar_data = {
        "figures": figures,
        "metadata": {
            "total_count": len(figures),
            "generated_at": "auto",
            "source_assets": str(ASSETS_JSONL),
            "source_dir": str(SOURCE_DIR),
            "static_root": str(STATIC_ROOT)
        }
    }
    
    OUT_INDEX.write_text(
        json.dumps(sidecar_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    # 안내 파일 생성
    NOTE_PATH.write_text(
        "이 파일은 figures_map.json의 사용법 안내입니다.\n"
        "- /static 경로에 정적 마운트되어야 프론트에서 접근 가능\n"
        f"- figures_map.json: {OUT_INDEX}\n"
        f"- PNG root:        {FIG_ROOT}\n"
        f"- 총 figure 개수:   {len(figures)}\n"
        "\n"
        "사용법:\n"
        "1. 서버에서 /static 마운트 확인\n"
        "2. 프론트에서 /static/viz/figures_map.json 로드\n"
        "3. [Figure] 토큰을 순서대로 교체\n",
        encoding="utf-8"
    )
    
    print(f"\n🎉 [COMPLETE] 사이드카 맵 생성 완료!")
    print(f"📄 figures_map.json: {OUT_INDEX}")
    print(f"📁 PNG root:         {FIG_ROOT}")
    print(f"📊 총 figure 개수:    {len(figures)}")
    
    if figures:
        print(f"🔗 예시 URL:         {figures[0]['image_path']}")
        print(f"🏷️  첫 번째 라벨:     {figures[0].get('label', 'N/A')}")


if __name__ == "__main__":
    main()
