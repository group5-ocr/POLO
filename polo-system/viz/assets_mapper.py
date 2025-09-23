"""
PDF 원본 파일을 PNG로 렌더링하고 에셋 인덱스 구축
- 중복 방지를 위한 해시 기반 파일명
- PyMuPDF를 사용한 PDF → PNG 변환
"""

from pathlib import Path
import json
import hashlib
import shutil
from typing import List, Dict, Any, Optional

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️ PyMuPDF not available. PDF rendering will be skipped.")


def _hash(s: str) -> str:
    """문자열을 해시해서 10자리 해시 문자열 생성"""
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:10]


def render_pdf_to_pngs(pdf_path: Path, out_dir: Path, dpi: int = 200) -> List[Path]:
    """
    PDF 파일을 페이지별로 PNG로 렌더링
    
    Args:
        pdf_path: PDF 파일 경로
        out_dir: 출력 디렉토리
        dpi: 렌더링 해상도 (기본값: 200)
        
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
        outs = []
        
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


def copy_image_file(src_path: Path, out_dir: Path) -> Optional[Path]:
    """
    이미지 파일을 출력 디렉토리로 복사
    
    Args:
        src_path: 원본 이미지 파일 경로
        out_dir: 출력 디렉토리
        
    Returns:
        복사된 파일 경로 또는 None
    """
    if not src_path.exists():
        return None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_path = out_dir / src_path.name
    
    # 이미 존재하면 재복사하지 않음
    if not dst_path.exists():
        try:
            shutil.copyfile(src_path, dst_path)
            print(f"✅ [IMG] 이미지 복사: {dst_path.name}")
        except Exception as e:
            print(f"⚠️ [IMG] 이미지 복사 실패: {e}")
            return None
    else:
        print(f"✅ [IMG] 파일 이미 존재, 재복사 스킵: {dst_path.name}")
    
    return dst_path


def find_graphics_file(graphics_name: str, source_dir: Path) -> Optional[Path]:
    """
    LaTeX graphics 파일명으로 실제 파일 찾기
    확장자가 생략된 경우 PDF 우선으로 검색
    
    Args:
        graphics_name: LaTeX에서 참조한 그래픽 파일명
        source_dir: 소스 디렉토리
        
    Returns:
        찾은 파일 경로 또는 None
    """
    # 직접 경로로 시도
    direct_path = (source_dir / graphics_name).resolve()
    if direct_path.exists():
        return direct_path
    
    # 확장자 생략된 경우 검색 (PDF 우선)
    stem = Path(graphics_name).stem
    extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.eps', '.ps']
    
    for ext in extensions:
        candidate = source_dir / f"{stem}{ext}"
        if candidate.exists():
            print(f"✅ [FIND] 그래픽 파일 발견: {graphics_name} → {candidate.name}")
            return candidate
    
    # 와일드카드 검색
    candidates = list(source_dir.glob(f"{stem}.*"))
    if candidates:
        print(f"✅ [FIND] 그래픽 파일 발견 (glob): {graphics_name} → {candidates[0].name}")
        return candidates[0]
    
    print(f"⚠️ [FIND] 그래픽 파일 없음: {graphics_name}")
    return None


def build_figure_index(
    assets_jsonl: Path,
    source_dir: Path,
    png_root: Path
) -> List[Dict[str, Any]]:
    """
    assets.jsonl에서 figure 정보를 읽어 PNG 인덱스 구축
    
    Args:
        assets_jsonl: assets.jsonl 파일 경로
        source_dir: 소스 디렉토리 (PDF/이미지 파일들)
        png_root: PNG 출력 루트 디렉토리
        
    Returns:
        figure 정보 리스트
        [{order, label, caption, src_file, pngs, key, rel_url}]
    """
    if not assets_jsonl.exists():
        print(f"⚠️ assets.jsonl 파일 없음: {assets_jsonl}")
        return []
    
    figures = []
    
    try:
        with assets_jsonl.open('r', encoding='utf-8') as f:
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
                env = str(obj.get('env', '')).lower()
                if env not in ('figure', 'figure*'):
                    continue
                
                graphics_name = obj.get('graphics')
                if not graphics_name:
                    print(f"⚠️ graphics 필드 없음 (line {line_num})")
                    continue
                
                # 실제 파일 찾기
                src_file = find_graphics_file(graphics_name, source_dir)
                if not src_file:
                    continue
                
                # 출력 디렉토리 설정
                fig_out_dir = png_root / "viz" / "figures" / Path(graphics_name).stem
                
                # PDF면 렌더링, 이미지면 복사
                pngs = []
                if src_file.suffix.lower() == '.pdf':
                    pngs = render_pdf_to_pngs(src_file, fig_out_dir)
                else:
                    # 이미지 파일 복사
                    copied = copy_image_file(src_file, fig_out_dir)
                    if copied:
                        pngs = [copied]
                
                if not pngs:
                    print(f"⚠️ PNG 생성 실패: {graphics_name}")
                    continue
                
                # 메타데이터 구성
                key = obj.get('label') or Path(graphics_name).stem
                caption = obj.get('caption', '')
                
                # 내용 해시로 버전 생성 (중복 방지)
                content_hash = _hash(f"{key}|{caption}|{src_file.as_posix()}")
                
                # 첫 번째 페이지를 기본으로 사용
                primary_png = pngs[0]
                rel_url = f"viz/figures/{Path(graphics_name).stem}/{primary_png.name}?v={content_hash}"
                
                figure_info = {
                    "order": len(figures) + 1,
                    "label": obj.get('label'),
                    "caption": caption,
                    "src_file": str(src_file),
                    "pngs": [str(p) for p in pngs],
                    "rel_url": rel_url,
                    "key": key,
                    "hash": content_hash
                }
                
                figures.append(figure_info)
                print(f"✅ [FIG] 인덱스 구축: {key} ({len(pngs)} pages)")
                
    except Exception as e:
        print(f"❌ [FIG] 인덱스 구축 실패: {e}")
        return []
    
    print(f"✅ [FIG] 총 {len(figures)}개 figure 인덱스 구축 완료")
    return figures


def get_figure_web_paths(figures: List[Dict[str, Any]], static_prefix: str = "/static") -> List[Dict[str, Any]]:
    """
    figure 정보에 웹 접근 가능한 경로 추가
    
    Args:
        figures: build_figure_index 결과
        static_prefix: 정적 파일 URL 프리픽스
        
    Returns:
        웹 경로가 추가된 figure 정보 리스트
    """
    for fig in figures:
        # 기본 이미지 경로
        fig["web_url"] = f"{static_prefix}/{fig['rel_url']}"
        
        # 모든 페이지 경로
        fig["all_web_pages"] = []
        for png_path in fig.get("pngs", []):
            png_path_obj = Path(png_path)
            # PNG 경로에서 상대 경로 추출
            try:
                # png_root를 기준으로 상대 경로 계산
                rel_path = png_path_obj.name  # 파일명만 사용
                fig_stem = Path(fig["key"]).stem
                web_path = f"{static_prefix}/viz/figures/{fig_stem}/{rel_path}?v={fig['hash']}"
                fig["all_web_pages"].append(web_path)
            except Exception as e:
                print(f"⚠️ 웹 경로 생성 실패: {png_path} - {e}")
    
    return figures
