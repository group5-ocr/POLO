# -*- coding: utf-8 -*-
"""
Figure 사이드카 맵 생성 도구 (기존 파이프라인에 추가만)
assets.jsonl → PDF→PNG 렌더링 → figures_map.json 생성
"""
from __future__ import annotations
import json
import hashlib
import shutil
from pathlib import Path

try:
    import fitz  # pip install pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️ PyMuPDF 없음. 설치: pip install pymupdf")

# 경로 설정 (실제 환경에 맞게 수정)
ASSETS_JSONL = Path(r"C:\POLO\POLO\polo-system\server\data\out\source\assets.jsonl")
SOURCE_DIR   = Path(r"C:\POLO\POLO\polo-system\server\data\arxiv\1506.02640\source")
STATIC_ROOT  = Path(r"C:\POLO\POLO\polo-system\server\data\outputs")

FIG_ROOT     = STATIC_ROOT / "viz" / "figures"            # PNG 보관
OUT_INDEX    = STATIC_ROOT / "viz" / "figures_map.json"   # 사이드카 맵


def md5_10(s: str) -> str:
    """10자리 MD5 해시"""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def to_web_url(p: Path) -> str:
    """Windows 경로를 /static 웹 URL로 변환"""
    try:
        rel = p.resolve().relative_to(STATIC_ROOT.resolve())
        return "/static/" + str(rel).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def render_pdf_to_pngs(pdf: Path, out_dir: Path, dpi: int = 220) -> list[Path]:
    """PDF → PNG 렌더링 (중복 방지)"""
    if not PYMUPDF_AVAILABLE:
        print(f"⚠️ PyMuPDF 없음, PDF 렌더링 스킵: {pdf}")
        return []
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        doc = fitz.open(str(pdf))
        outs = []
        
        for i, page in enumerate(doc):
            out_path = out_dir / f"{pdf.stem}_p{i+1}.png"
            if not out_path.exists():
                try:
                    page.get_pixmap(dpi=dpi).save(str(out_path))
                    print(f"✅ [PDF] 렌더링: {out_path.name}")
                except Exception as e:
                    print(f"⚠️ [PDF] 렌더링 실패 {i+1}: {e}")
                    continue
            else:
                print(f"✅ [PDF] 재사용: {out_path.name}")
            outs.append(out_path)
        
        doc.close()
        return outs
        
    except Exception as e:
        print(f"❌ [PDF] 열기 실패: {pdf} - {e}")
        return []


def copy_if_needed(src: Path, dst_dir: Path) -> Path:
    """이미지 파일 복사 (중복 방지)"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    
    if not dst.exists():
        try:
            shutil.copyfile(src, dst)
            print(f"✅ [IMG] 복사: {dst.name}")
        except Exception as e:
            print(f"⚠️ [IMG] 복사 실패: {e}")
    else:
        print(f"✅ [IMG] 재사용: {dst.name}")
    
    return dst


def main():
    """메인 실행"""
    # 입력 검증
    if not ASSETS_JSONL.exists():
        print(f"❌ assets.jsonl 없음: {ASSETS_JSONL}")
        return
    
    if not SOURCE_DIR.exists():
        print(f"❌ source 폴더 없음: {SOURCE_DIR}")
        return
    
    print(f"📖 [START] Figure 맵 생성")
    print(f"  - assets.jsonl: {ASSETS_JSONL}")
    print(f"  - source:       {SOURCE_DIR}")
    print(f"  - output:       {OUT_INDEX}")
    
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    
    figures = []
    order = 0
    
    # assets.jsonl 파싱
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
            
            graphics_list = obj.get("graphics", [])
            if not graphics_list or len(graphics_list) == 0:
                print(f"⚠️ graphics 필드 없음 (line {line_num})")
                continue
            
            # graphics가 리스트인 경우 첫 번째 항목 사용
            graphics = graphics_list[0] if isinstance(graphics_list, list) else graphics_list
            if not graphics:
                print(f"⚠️ graphics 값 없음 (line {line_num})")
                continue
            
            print(f"🔍 [PROCESS] {graphics}")
            
            # 원본 파일 찾기 (확장자 생략 대응)
            src = SOURCE_DIR / graphics
            if not src.exists():
                stem = Path(graphics).stem
                # PDF 우선, 없으면 다른 확장자
                candidates = list(SOURCE_DIR.glob(f"{stem}.pdf")) or list(SOURCE_DIR.glob(f"{stem}.*"))
                if candidates:
                    src = candidates[0]
                    print(f"  → 발견: {src.name}")
            
            if not src.exists():
                print(f"  ❌ 원본 없음: {graphics}")
                continue
            
            # 출력 디렉터리
            out_dir = FIG_ROOT / Path(graphics).stem
            
            # PDF → PNG 또는 이미지 복사
            if src.suffix.lower() == ".pdf":
                pngs = render_pdf_to_pngs(src, out_dir)
            else:
                png = copy_if_needed(src, out_dir)
                pngs = [png] if png.exists() else []
            
            if not pngs:
                print(f"  ❌ PNG 생성 실패")
                continue
            
            # 메타데이터 생성
            key = obj.get("label") or Path(graphics).stem
            cap = obj.get("caption") or ""
            ver = md5_10(f"{key}|{cap}|{src.as_posix()}")
            main_png = pngs[0]
            order += 1
            
            figure_item = {
                "order": order,
                "label": obj.get("label"),
                "caption": cap,
                "graphics": graphics,
                "src_file": str(src),
                "image_path": to_web_url(main_png) + f"?v={ver}",
                "all_pages": [to_web_url(p) + f"?v={ver}" for p in pngs]
            }
            
            figures.append(figure_item)
            print(f"  ✅ 추가: {graphics} ({len(pngs)} pages)")
    
    # 사이드카 맵 저장
    sidecar_data = {
        "figures": figures,
        "metadata": {
            "total_count": len(figures),
            "generated_at": "auto",
            "source_assets": str(ASSETS_JSONL),
            "source_dir": str(SOURCE_DIR)
        }
    }
    
    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_INDEX.write_text(
        json.dumps(sidecar_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    print(f"\n🎉 [COMPLETE] figures_map.json 생성!")
    print(f"  📄 출력: {OUT_INDEX}")
    print(f"  📊 총 {len(figures)}개 figures")
    
    if figures:
        print(f"  🔗 예시: {figures[0]['image_path']}")


if __name__ == "__main__":
    main()
