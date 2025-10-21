import io
import logging
from typing import List, Optional

import fitz  # type: ignore
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PDF Watermark Remover", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalise_keywords(keywords: Optional[str]) -> List[str]:
    """Normalise a comma separated keyword string into a list."""
    if not keywords:
        return ["watermark", "confidential", "draft"]

    cleaned: List[str] = []
    for part in keywords.split(","):
        keyword = part.strip().lower()
        if keyword:
            cleaned.append(keyword)
    return cleaned or ["watermark", "confidential", "draft"]


@app.post("/api/remove-watermark")
async def remove_watermark(
    file: UploadFile = File(...),
    keywords: Optional[str] = Form(None),
):
    """Remove watermarks that match supplied keywords from a PDF file."""
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    keyword_list = _normalise_keywords(keywords)
    logger.info("Removing watermarks using keywords: %s", keyword_list)

    try:
        cleaned_pdf = _remove_watermark_bytes(data, keyword_list)
    except Exception as exc:  # pragma: no cover - logged and reported
        logger.exception("Failed to remove watermark")
        raise HTTPException(status_code=500, detail="Failed to process PDF") from exc

    headers = {"Content-Disposition": f"attachment; filename=cleaned-{file.filename}"}
    return Response(content=cleaned_pdf, media_type="application/pdf", headers=headers)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="frontend", html=True), name="static")


def _remove_watermark_bytes(data: bytes, keywords: List[str]) -> bytes:
    doc = fitz.open(stream=data, filetype="pdf")

    keyword_set = {k.lower() for k in keywords}
    annotations_removed_total = 0
    redactions_applied_total = 0

    for page in doc:
        # Remove annotations that contain keywords
        annotations = page.annots()
        if annotations:
            to_delete = []
            for annot in annotations:
                contents = (annot.info or {}).get("content", "")
                if any(keyword in contents.lower() for keyword in keyword_set):
                    to_delete.append(annot)
            for annot in to_delete:
                annotations_removed_total += 1
                page.delete_annot(annot)

        text_dict = page.get_text("dict")
        page_redactions = 0
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text:
                        continue
                    lowered = text.lower()
                    if any(keyword in lowered for keyword in keyword_set):
                        rect = fitz.Rect(span["bbox"])
                        page.add_redact_annot(rect, fill=(1, 1, 1))
                        page_redactions += 1

        if page_redactions:
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            redactions_applied_total += page_redactions

    logger.info(
        "Watermark removal completed. %s annotations removed, %s redactions applied",
        annotations_removed_total,
        redactions_applied_total,
    )

    output = io.BytesIO()
    doc.save(output, incremental=False, deflate=True)
    doc.close()
    return output.getvalue()


__all__ = ["app"]
