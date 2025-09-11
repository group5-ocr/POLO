from pathlib import Path
from typing import Optional, TypedDict
import anyio
from services.external import arxiv_downloader_back as vendor

class ArxivResult(TypedDict):
    arxiv_id: str
    out_dir: str
    pdf_path: str
    src_tar: str
    source_dir: str
    main_tex: Optional[str]


async def fetch_and_extract(
    arxiv_id: str,
    out_root: str = "data/arxiv",
    corp_ca_pem: Optional[str] = None,
    left_margin_px: int = 120,
    preview_lines: int = 40,
) -> ArxivResult:
    out_root_p = Path(out_root)

    def _run():
        vendor.run(
            arxiv_id=arxiv_id,
            pdf_to_extract_id=None,
            out_root=out_root_p,
            corp_ca_pem=corp_ca_pem,
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

        return out_dir, pdf, tar, source, (str(main_tex) if main_tex else None)

    out_dir, pdf, tar, source, main_tex = await anyio.to_thread.run_sync(_run)

    return {
        "arxiv_id": arxiv_id,
        "out_dir": str(out_dir),
        "pdf_path": str(pdf),
        "src_tar": str(tar),
        "source_dir": str(source),
        "main_tex": main_tex,
    }