# services/arxiv_client.py
from __future__ import annotations
import os, re
from pathlib import Path
from typing import Optional, TypedDict
import anyio

# 벤더(외부 스크립트)
from services.external import arxiv_downloader_back as vendor

ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")

class ArxivResult(TypedDict):
    arxiv_id: str
    out_dir: str
    pdf_path: str
    src_tar: str
    source_dir: str
    main_tex: Optional[str]

def _get_out_root(out_root: Optional[str]) -> Path:
    # .env 가 있다면 여기서 읽어도 됨: os.getenv("ARXIV_OUT_ROOT", "server/data/arxiv")
    base = out_root or os.getenv("ARXIV_OUT_ROOT", "server/data/arxiv")
    p = Path(base).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def _get_corp_ca() -> Optional[str]:
    v = os.getenv("CORP_CA_PEM")
    return v if v else None

async def fetch_and_extract(
    arxiv_id: str,
    out_root: Optional[str] = None,
    corp_ca_pem: Optional[str] = None,
    left_margin_px: int = 120,
    preview_lines: int = 40,
) -> ArxivResult:
    """arXiv ID로 PDF/e-print를 내려받고, 안전 추출 + main.tex 추정까지 수행."""
    if not ARXIV_ID_RE.match(arxiv_id):
        raise ValueError(f"Invalid arXiv id format: {arxiv_id}")

    out_root_p = _get_out_root(out_root)
    corp_ca = corp_ca_pem if corp_ca_pem is not None else _get_corp_ca()

    def _run():
        vendor.run(
            arxiv_id=arxiv_id,
            pdf_to_extract_id=None,
            out_root=out_root_p,
            corp_ca_pem=corp_ca,
            left_margin_px=left_margin_px,
            preview_lines=preview_lines,
            clean_extract_dir=True,
        )
        out_dir = out_root_p / arxiv_id
        pdf = out_dir / f"{arxiv_id}.pdf"
        tar = out_dir / f"{arxiv_id}_source.tar.gz"
        source = out_dir / "source"

        tex_list = vendor.find_all_tex(source)
        main_tex = vendor.guess_main_tex(tex_list) if tex_list else None

        return (
            out_dir.resolve(),
            pdf.resolve(),
            tar.resolve(),
            source.resolve(),
            (str(main_tex.resolve()) if main_tex else None),
        )

    out_dir, pdf, tar, source, main_tex = await anyio.to_thread.run_sync(_run)

    return {
        "arxiv_id": arxiv_id,
        "out_dir": str(out_dir),
        "pdf_path": str(pdf),
        "src_tar": str(tar),
        "source_dir": str(source),
        "main_tex": main_tex,
    }
