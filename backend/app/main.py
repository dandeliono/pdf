import io
import logging
import math
from pathlib import Path
from typing import Dict, Set

import fitz  # type: ignore
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

if not FRONTEND_DIR.exists():
    raise RuntimeError(
        f"Frontend assets directory not found: {FRONTEND_DIR}. "
        "Ensure the project is run from the repository root or that the "
        "frontend assets are copied alongside the backend."
    )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PDF Background Watermark Remover", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/remove-watermark")
async def remove_watermark(
    file: UploadFile = File(...),
    coverage_ratio: float = Form(0.5),
    page_ratio: float = Form(0.8),
):
    """Remove repeated background watermarks from a PDF file."""
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    _ensure_ratio(coverage_ratio, "coverage_ratio")
    _ensure_ratio(page_ratio, "page_ratio")
    logger.info(
        "Removing background watermarks using coverage_ratio=%s, page_ratio=%s",
        coverage_ratio,
        page_ratio,
    )

    try:
        cleaned_pdf = _remove_watermark_bytes(data, coverage_ratio, page_ratio)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - logged and reported
        logger.exception("Failed to remove watermark")
        raise HTTPException(status_code=500, detail="Failed to process PDF") from exc

    headers = {"Content-Disposition": f"attachment; filename=cleaned-{file.filename}"}
    return Response(content=cleaned_pdf, media_type="application/pdf", headers=headers)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")


def _remove_watermark_bytes(data: bytes, coverage_ratio: float, page_ratio: float) -> bytes:
    doc = fitz.open(stream=data, filetype="pdf")

    try:
        if doc.page_count == 0:
            raise ValueError("The uploaded PDF contains no pages")

        background_xrefs = _identify_background_images(doc, coverage_ratio, page_ratio)
        logger.info("Identified %s background image candidates", len(background_xrefs))

        pages_cleaned = 0
        for page in doc:
            removed = False
            for xref in background_xrefs:
                if page.get_image_rects(xref):
                    page.delete_image(xref)
                    removed = True
            if removed:
                page.clean_contents()
                pages_cleaned += 1

        logger.info(
            "Background watermark removal completed. Pages cleaned: %s", pages_cleaned
        )

        output = io.BytesIO()
        doc.save(output, incremental=False, deflate=True, garbage=4)
        return output.getvalue()
    finally:
        doc.close()


def _identify_background_images(
    doc: "fitz.Document", coverage_ratio: float, page_ratio: float
) -> Set[int]:
    """Identify image xrefs that match the background watermark pattern."""

    num_pages = len(doc)
    min_pages_required = max(1, math.ceil(page_ratio * num_pages))
    usage: Dict[int, Set[int]] = {}

    for page in doc:
        page_area = abs(page.rect)
        if page_area == 0:
            continue

        for image in page.get_images(full=True):
            xref = image[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue

            coverage = sum(abs(rect) for rect in rects) / page_area
            if coverage < coverage_ratio:
                continue

            pages = usage.setdefault(xref, set())
            pages.add(page.number)

    background_xrefs = {
        xref for xref, pages in usage.items() if len(pages) >= min_pages_required
    }

    logger.info(
        "Coverage threshold %.0f%%, background images selected: %s",
        coverage_ratio * 100,
        background_xrefs,
    )
    return background_xrefs


def _ensure_ratio(value: float, name: str) -> None:
    if not (0 < value <= 1):
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be between 0 and 1 (exclusive of 0)",
        )


__all__ = ["app"]
