"""
PNG 시각화 파일 저장 유틸리티
- 중복 생성 방지를 위한 내용 기반 해시 파일명
- Windows → Web 경로 정규화
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Union


def spec_hash(spec: dict) -> str:
    """
    생성 스펙(JSON)을 해시해서 내용 기반 파일명 생성
    
    Args:
        spec: 시각화 생성 스펙 딕셔너리
        
    Returns:
        10자리 해시 문자열
    """
    data = json.dumps(spec, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.md5(data).hexdigest()[:10]


def save_png_unique(img, out_dir: Path, base: str, spec: dict) -> Path:
    """
    내용 기반 해시 파일명으로 PNG 저장 (중복 방지)
    
    Args:
        img: PIL Image 객체 또는 저장할 이미지
        out_dir: 출력 디렉토리
        base: 기본 파일명 (확장자 제외)
        spec: 시각화 생성 스펙
        
    Returns:
        저장된 파일의 Path 객체
        
    Examples:
        >>> save_png_unique(img, Path("outputs/viz"), "flow_arch", spec)
        Path("outputs/viz/flow_arch__a1b2c3d4e5.png")
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h = spec_hash(spec)
    out_path = out_dir / f"{base}__{h}.png"
    
    # 이미 존재하면 재생성하지 않음
    if out_path.exists():
        print(f"✅ [VIZ] 파일 이미 존재, 재생성 스킵: {out_path.name}")
        return out_path
    
    # 새로 생성
    if hasattr(img, 'save'):
        # PIL Image 객체인 경우
        img.save(str(out_path))
    else:
        # 파일 경로나 바이트 데이터인 경우
        if isinstance(img, (str, Path)):
            # 파일 경로인 경우 복사
            import shutil
            shutil.copy2(img, out_path)
        elif isinstance(img, bytes):
            # 바이트 데이터인 경우 직접 저장
            with open(out_path, 'wb') as f:
                f.write(img)
        else:
            raise ValueError(f"지원하지 않는 이미지 타입: {type(img)}")
    
    print(f"✅ [VIZ] 새 파일 생성: {out_path.name}")
    return out_path


def save_base64_png_unique(base64_data: str, out_dir: Path, base: str, spec: dict) -> Path:
    """
    Base64 데이터를 내용 기반 해시 파일명으로 PNG 저장
    
    Args:
        base64_data: Base64 인코딩된 이미지 데이터
        out_dir: 출력 디렉토리
        base: 기본 파일명 (확장자 제외)
        spec: 시각화 생성 스펙
        
    Returns:
        저장된 파일의 Path 객체
    """
    import base64
    
    out_dir.mkdir(parents=True, exist_ok=True)
    h = spec_hash(spec)
    out_path = out_dir / f"{base}__{h}.png"
    
    # 이미 존재하면 재생성하지 않음
    if out_path.exists():
        print(f"✅ [VIZ] 파일 이미 존재, 재생성 스킵: {out_path.name}")
        return out_path
    
    # Base64 디코딩 후 저장
    img_bytes = base64.b64decode(base64_data)
    with open(out_path, 'wb') as f:
        f.write(img_bytes)
    
    print(f"✅ [VIZ] 새 파일 생성: {out_path.name}")
    return out_path


# 정적 파일 루트 설정 (환경에 따라 조정)
STATIC_ROOT = Path(__file__).resolve().parent.parent / "server" / "data" / "outputs"


def to_web_path(file_path: Union[str, Path]) -> str:
    """
    파일 경로를 웹 접근 가능한 URL 경로로 변환
    Windows 역슬래시를 슬래시로 정규화
    
    Args:
        file_path: 파일 경로 (절대 경로 또는 Path 객체)
        
    Returns:
        웹 접근 가능한 URL 경로 (예: "/static/viz/section_1/flow_arch__a1b2c3.png")
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    try:
        # STATIC_ROOT 기준 상대 경로 계산
        rel_path = file_path.relative_to(STATIC_ROOT)
        # Windows 역슬래시를 슬래시로 변환
        web_path = "/static/" + str(rel_path).replace("\\", "/")
        return web_path
    except ValueError:
        # STATIC_ROOT와 관계없는 경로인 경우 그대로 반환
        return str(file_path).replace("\\", "/")


def normalize_image_path(raw_path: str, static_root: Path = None) -> str:
    """
    이미지 경로를 웹 표준 형식으로 정규화
    
    Args:
        raw_path: 원본 경로
        static_root: 정적 파일 루트 (기본값: STATIC_ROOT)
        
    Returns:
        정규화된 웹 경로
    """
    if not raw_path:
        return ""
    
    if static_root is None:
        static_root = STATIC_ROOT
    
    # 이미 웹 경로 형식인 경우
    if raw_path.startswith(('/static/', 'http://', 'https://')):
        return raw_path
    
    # 파일 경로인 경우 웹 경로로 변환
    try:
        path = Path(raw_path)
        if path.is_absolute():
            return to_web_path(path)
        else:
            # 상대 경로인 경우 static 경로로 변환
            return "/static/" + raw_path.replace("\\", "/")
    except Exception:
        # 변환 실패 시 원본 반환 (역슬래시만 정규화)
        return raw_path.replace("\\", "/")


def get_unique_filename(base: str, spec: dict, extension: str = "png") -> str:
    """
    내용 기반 해시를 포함한 고유 파일명 생성
    
    Args:
        base: 기본 파일명
        spec: 생성 스펙
        extension: 파일 확장자 (기본값: "png")
        
    Returns:
        해시가 포함된 고유 파일명
        
    Examples:
        >>> get_unique_filename("flow_arch", spec)
        "flow_arch__a1b2c3d4e5.png"
    """
    h = spec_hash(spec)
    return f"{base}__{h}.{extension}"
