import asyncio
import base64
import binascii
import io
import logging
import math
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Optional, Set, Tuple, cast
from collections import defaultdict

import fitz  # type: ignore
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from starlette.websockets import WebSocketState
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PDF Watermark Remover", version="1.0.0")

_APP_DIR = Path(__file__).resolve().parent
_FRONTEND_DIR_CANDIDATES = [
    _APP_DIR.parent / "frontend",
    _APP_DIR.parent.parent / "frontend",
]
for _candidate in _FRONTEND_DIR_CANDIDATES:
    if (_candidate / "index.html").is_file():
        FRONTEND_DIR = _candidate
        break
else:  # pragma: no cover - fails fast during startup
    raise RuntimeError("Unable to locate frontend directory for static files")

UPLOAD_ROOT = Path(tempfile.gettempdir()) / "pdf_watermark_uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

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


_ALLOWED_MODES = {"auto", "vector", "raster"}


def _normalise_mode(mode: Optional[str]) -> str:
    if mode is None:
        return "auto"
    value = str(mode).strip().lower()
    if not value:
        return "auto"
    if value not in _ALLOWED_MODES:
        raise ValueError(
            "无效的处理模式，可选值为: auto, vector, raster"
        )
    return value


def _safe_mode(mode: Optional[str]) -> str:
    try:
        return _normalise_mode(mode)
    except ValueError:
        logger.warning("Unknown processing mode '%s', falling back to auto", mode)
        return "auto"


@app.post("/api/remove-watermark")
async def remove_watermark(
    file: Optional[UploadFile] = File(None),
    upload_id: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
):
    """Remove watermarks that match supplied keywords from a PDF file."""
    keyword_list = _normalise_keywords(keywords)
    try:
        request_mode = _normalise_mode(mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("Removing watermarks using keywords: %s", keyword_list)

    if upload_id and file:
        raise HTTPException(
            status_code=400, detail="Provide either upload_id or file, not both"
        )

    if upload_id:
        try:
            assembled_path, metadata = _assemble_upload(upload_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        original_filename = str(metadata.get("filename") or "uploaded.pdf")
        processing_mode = _safe_mode(metadata.get("mode"))
        logger.info("Processing upload %s using mode: %s", upload_id, processing_mode)
        try:
            cleaned_pdf = _remove_watermark_from_path(
                assembled_path, keyword_list, mode=processing_mode
            )
        except Exception as exc:  # pragma: no cover - logged and reported
            logger.exception("Failed to remove watermark for upload %s", upload_id)
            raise HTTPException(status_code=500, detail="Failed to process PDF") from exc
        finally:
            _cleanup_upload(upload_id)
    else:
        if not file:
            raise HTTPException(status_code=400, detail="No PDF file supplied")

        if file.content_type not in {"application/pdf", "application/x-pdf"}:
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        original_filename = file.filename or "uploaded.pdf"
        logger.info("Processing immediate upload using mode: %s", request_mode)
        try:
            cleaned_pdf = _remove_watermark_from_bytes(
                data, keyword_list, mode=request_mode
            )
        except Exception as exc:  # pragma: no cover - logged and reported
            logger.exception("Failed to remove watermark")
            raise HTTPException(status_code=500, detail="Failed to process PDF") from exc

    safe_filename = Path(original_filename).name or "uploaded.pdf"
    headers = {"Content-Disposition": f"attachment; filename=cleaned-{safe_filename}"}
    return Response(content=cleaned_pdf, media_type="application/pdf", headers=headers)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/", response_class=FileResponse)
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=False), name="static")


ProgressCallback = Callable[[str, float, str], None]


class UploadInitRequest(BaseModel):
    file_name: str = Field(..., alias="fileName")
    total_chunks: int = Field(..., alias="totalChunks")
    mode: Optional[str] = "auto"
    model_config = ConfigDict(populate_by_name=True)


class UploadInitResponse(BaseModel):
    upload_id: str = Field(..., alias="uploadId")
    model_config = ConfigDict(populate_by_name=True)


def _metadata_path(upload_dir: Path) -> Path:
    return upload_dir / "meta.json"


def _write_metadata(upload_dir: Path, metadata: Dict[str, object]) -> None:
    meta_path = _metadata_path(upload_dir)
    tmp_path = meta_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(metadata, ensure_ascii=False))
    tmp_path.replace(meta_path)


def _read_metadata(upload_dir: Path) -> Dict[str, object]:
    meta_path = _metadata_path(upload_dir)
    if not meta_path.exists():
        raise ValueError("Upload session metadata missing")
    data = json.loads(meta_path.read_text())
    uploaded = data.get("uploaded")
    if isinstance(uploaded, list):
        data["uploaded"] = [int(index) for index in uploaded]
    else:
        data["uploaded"] = []
    data["total_chunks"] = int(data.get("total_chunks", 0))
    data["mode"] = _safe_mode(data.get("mode"))
    return data


def _create_upload_session(file_name: str, total_chunks: int, mode: str) -> str:
    if total_chunks <= 0:
        raise ValueError("total_chunks must be greater than zero")
    upload_id = uuid.uuid4().hex
    upload_dir = UPLOAD_ROOT / upload_id
    upload_dir.mkdir(parents=True, exist_ok=False)
    metadata: Dict[str, object] = {
        "filename": Path(file_name).name or "uploaded.pdf",
        "total_chunks": total_chunks,
        "uploaded": [],
        "mode": mode,
    }
    _write_metadata(upload_dir, metadata)
    return upload_id


def _get_upload_dir(upload_id: str) -> Path:
    upload_dir = UPLOAD_ROOT / upload_id
    if not upload_dir.exists():
        raise ValueError("Upload session not found")
    return upload_dir


def _save_upload_chunk(
    upload_id: str, chunk_index: int, total_chunks_claim: int, data: bytes
) -> Tuple[int, int]:
    if chunk_index < 0:
        raise ValueError("chunk_index must be zero or positive")
    upload_dir = _get_upload_dir(upload_id)
    metadata = _read_metadata(upload_dir)
    total_chunks = metadata["total_chunks"]
    if total_chunks_claim != total_chunks:
        raise ValueError("total_chunks mismatch for upload session")
    if chunk_index >= total_chunks:
        raise ValueError("chunk_index exceeds declared total_chunks")

    chunk_path = upload_dir / f"{chunk_index:06d}.chunk"
    with open(chunk_path, "wb") as chunk_file:
        chunk_file.write(data)

    uploaded_indices = set(int(idx) for idx in metadata.get("uploaded", []))
    uploaded_indices.add(chunk_index)
    metadata["uploaded"] = sorted(uploaded_indices)
    _write_metadata(upload_dir, metadata)

    return len(uploaded_indices), total_chunks


def _assemble_upload(upload_id: str) -> Tuple[Path, Dict[str, object]]:
    upload_dir = _get_upload_dir(upload_id)
    metadata = _read_metadata(upload_dir)
    total_chunks = metadata["total_chunks"]
    if total_chunks <= 0:
        raise ValueError("Upload session has no chunks")

    missing = [
        index
        for index in range(total_chunks)
        if not (upload_dir / f"{index:06d}.chunk").exists()
    ]
    if missing:
        raise ValueError(f"Missing chunks: {missing}")

    assembled_path = upload_dir / "assembled.pdf"
    with open(assembled_path, "wb") as target:
        for index in range(total_chunks):
            chunk_path = upload_dir / f"{index:06d}.chunk"
            with open(chunk_path, "rb") as source:
                shutil.copyfileobj(source, target)

    return assembled_path, metadata


def _cleanup_upload(upload_id: str) -> None:
    upload_dir = UPLOAD_ROOT / upload_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)


@app.post("/api/upload/init", response_model=UploadInitResponse)
async def init_chunked_upload(payload: UploadInitRequest):
    file_name = payload.file_name or "uploaded.pdf"
    try:
        processing_mode = _normalise_mode(payload.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        upload_id = _create_upload_session(file_name, payload.total_chunks, processing_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "Initialised upload session %s for file %s with %s chunks (mode=%s)",
        upload_id,
        file_name,
        payload.total_chunks,
        processing_mode,
    )
    return UploadInitResponse(upload_id=upload_id)


@app.post("/api/upload/chunk")
async def receive_upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    chunk: UploadFile = File(...),
):
    try:
        data = await chunk.read()
    except Exception as exc:  # pragma: no cover - FastAPI handles file IO
        raise HTTPException(status_code=400, detail="Failed to read uploaded chunk") from exc

    if not data:
        raise HTTPException(status_code=400, detail="Chunk must not be empty")

    try:
        received, expected_total = _save_upload_chunk(upload_id, chunk_index, total_chunks, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.debug(
        "Received chunk %s/%s for upload %s (%s bytes)",
        chunk_index + 1,
        expected_total,
        upload_id,
        len(data),
    )
    return {
        "status": "ok",
        "receivedChunks": received,
        "totalChunks": expected_total,
    }


@app.websocket("/ws/remove-watermark")
async def remove_watermark_ws(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    upload_id: Optional[str] = None
    total_chunks: Optional[int] = None
    received_chunks = 0

    async def _send_status(message: str, progress: float, stage: str = "status", **extra) -> None:
        if websocket.application_state != WebSocketState.CONNECTED:
            return
        payload = {
            "type": "status",
            "stage": stage,
            "progress": round(max(0.0, min(progress, 100.0)), 2),
            "message": message,
        }
        payload.update(extra)
        await websocket.send_json(payload)

    async def _send_error(message: str, close_code: int = 1011) -> None:
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"type": "error", "message": message})
            finally:
                await websocket.close(code=close_code)
        if upload_id:
            _cleanup_upload(upload_id)

    try:
        initial_message = await websocket.receive_json()
    except WebSocketDisconnect:
        return
    except Exception:
        await _send_error("无法解析请求，请重试", 1003)
        return

    if not isinstance(initial_message, dict) or initial_message.get("type") != "start":
        await _send_error("无效的请求类型", 1003)
        return

    file_name = initial_message.get("fileName") or "uploaded.pdf"
    keywords_input = initial_message.get("keywords") or ""
    total_chunks = initial_message.get("totalChunks")
    mode_input = initial_message.get("mode")

    if not isinstance(total_chunks, int) or total_chunks <= 0:
        await _send_error("totalChunks 参数无效", 1003)
        return

    keyword_list = _normalise_keywords(keywords_input)

    try:
        processing_mode = _normalise_mode(mode_input)
    except ValueError as exc:
        await _send_error(str(exc), 1003)
        return

    try:
        upload_id = _create_upload_session(file_name, total_chunks, processing_mode)
    except ValueError as exc:
        await _send_error(str(exc), 1003)
        return

    await _send_status("上传会话已建立，准备接收分片", 0.0, stage="init", uploadId=upload_id)

    try:
        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                await _send_error("收到未知格式的数据", 1003)
                return

            message_type = message.get("type")
            if message_type == "chunk":
                if total_chunks is None:
                    await _send_error("上传会话尚未初始化", 1003)
                    return
                if "index" not in message or "data" not in message:
                    await _send_error("分片数据缺少必要字段", 1003)
                    return
                chunk_index = message["index"]
                chunk_data_b64 = message["data"]

                if not isinstance(chunk_index, int):
                    await _send_error("分片索引必须为整数", 1003)
                    return
                if not isinstance(chunk_data_b64, str):
                    await _send_error("分片数据必须为字符串", 1003)
                    return

                try:
                    chunk_bytes = base64.b64decode(chunk_data_b64)
                except binascii.Error:
                    await _send_error("分片数据解码失败", 1003)
                    return

                try:
                    received_chunks, _ = _save_upload_chunk(
                        upload_id, chunk_index, total_chunks, chunk_bytes
                    )
                except ValueError as exc:
                    await _send_error(str(exc), 1003)
                    return

                upload_progress = (received_chunks / total_chunks) * 30.0
                await _send_status(
                    f"已接收分片 {received_chunks}/{total_chunks}",
                    upload_progress,
                    stage="upload",
                )
            elif message_type == "end":
                break
            else:
                await _send_error("收到未知类型的消息", 1003)
                return
    except WebSocketDisconnect:
        if upload_id:
            _cleanup_upload(upload_id)
        return
    except Exception:
        await _send_error("处理分片时发生错误", 1011)
        return

    if total_chunks is None or received_chunks < total_chunks:
        await _send_error("分片接收未完成", 1003)
        return

    try:
        assembled_path, metadata = _assemble_upload(upload_id)
    except ValueError as exc:
        await _send_error(str(exc), 1003)
        return

    await _send_status("分片上传完成，开始分析 PDF", 32.0, stage="process")

    original_filename = Path(str(metadata.get("filename") or "uploaded.pdf")).name
    processing_mode = _safe_mode(metadata.get("mode"))

    def _report(stage: str, progress: float, message: str) -> None:
        if websocket.application_state != WebSocketState.CONNECTED:
            return
        scaled_progress = 30.0 + max(0.0, min(progress, 1.0)) * 70.0
        future = asyncio.run_coroutine_threadsafe(
            websocket.send_json(
                {
                    "type": "status",
                    "stage": stage,
                    "progress": round(scaled_progress, 2),
                    "message": message,
                }
            ),
            loop,
        )

        def _safely_consume(fut):
            try:
                fut.result()
            except Exception:
                logger.debug("Ignored websocket send failure", exc_info=True)

        future.add_done_callback(_safely_consume)

    try:
        processed_bytes = await loop.run_in_executor(
            None,
            lambda: _remove_watermark_from_path(
                assembled_path, keyword_list, progress_callback=_report, mode=processing_mode
            ),
        )
    except Exception:
        logger.exception("Failed to process PDF via websocket upload %s", upload_id)
        await _send_error("处理 PDF 失败，请稍后再试", 1011)
        return
    finally:
        if upload_id:
            _cleanup_upload(upload_id)

    encoded_result = base64.b64encode(processed_bytes).decode("ascii")
    await websocket.send_json(
        {
            "type": "complete",
            "message": "处理完成，准备下载",
            "fileName": f"cleaned-{original_filename}",
            "progress": 100.0,
            "fileData": encoded_result,
        }
    )
    await websocket.close()


def _remove_watermark_from_bytes(
    data: bytes,
    keywords: List[str],
    progress_callback: Optional[ProgressCallback] = None,
    mode: str = "auto",
) -> bytes:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return _process_document(doc, keywords, progress_callback, mode=_safe_mode(mode))
    finally:
        doc.close()


def _remove_watermark_from_path(
    path: Path,
    keywords: List[str],
    progress_callback: Optional[ProgressCallback] = None,
    mode: str = "auto",
) -> bytes:
    doc = fitz.open(str(path))
    try:
        return _process_document(doc, keywords, progress_callback, mode=_safe_mode(mode))
    finally:
        doc.close()


def _process_document(
    doc: fitz.Document,
    keywords: List[str],
    progress_callback: Optional[ProgressCallback] = None,
    mode: str = "auto",
) -> bytes:
    processing_mode = _safe_mode(mode)
    keyword_set = {k.lower() for k in keywords}
    annotations_removed_total = 0
    redactions_applied_total = 0
    background_images_removed_total = 0
    repeated_text_redactions_total = 0
    rasterized_pages_total = 0
    total_pages = len(doc)
    mask_rects: Dict[int, List[fitz.Rect]] = defaultdict(list)

    if progress_callback:
        progress_callback("start", 0.0, "开始分析 PDF")

    background_image_xrefs: Set[int] = set()
    if processing_mode != "raster":
        background_image_xrefs = _identify_background_images(
            doc, progress_callback=progress_callback, progress_span=(0.05, 0.35)
        )
        if background_image_xrefs:
            logger.info(
                "Detected %s candidate background images with significant coverage",
                len(background_image_xrefs),
            )
            if progress_callback:
                progress_callback(
                    "scan_background",
                    0.37,
                    f"检测到 {len(background_image_xrefs)} 个疑似背景水印图像",
                )
    elif progress_callback:
        progress_callback("scan_background", 0.35, "图像模式将跳过背景图像拆分")

    repeated_text_candidates = _identify_repeated_text_watermarks(
        doc, progress_callback=progress_callback, progress_span=(0.35, 0.45)
    )
    if repeated_text_candidates:
        logger.info(
            "Detected %s repeated text watermark candidates",
            len(repeated_text_candidates),
        )
        if progress_callback:
            progress_callback(
                "scan_text",
                0.46,
                f"检测到 {len(repeated_text_candidates)} 个重复文本水印候选",
            )

    processing_span = (0.45, 0.85 if processing_mode == "raster" else 0.95)
    for index, page in enumerate(doc, start=1):
        page_bounds = page.rect

        if background_image_xrefs and processing_mode != "raster":
            removed_on_page = 0
            cleaned = False
            for xref in background_image_xrefs:
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                if not cleaned:
                    page.clean_contents()
                    cleaned = True
                removed_on_page += len(rects)
                page.delete_image(xref)
            if removed_on_page:
                background_images_removed_total += removed_on_page

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

        page_redactions = 0
        text_dict = page.get_text("dict")

        if repeated_text_candidates:
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    line_text = "".join(span.get("text", "") for span in spans)
                    normalised_line = "".join(line_text.split())
                    if not normalised_line or normalised_line not in repeated_text_candidates:
                        continue
                    rect = _union_rects([fitz.Rect(span["bbox"]) for span in spans])
                    mask_rect = _register_mask_rect(mask_rects, index - 1, rect, page_bounds)
                    if mask_rect is None:
                        continue
                    if processing_mode != "raster":
                        page.add_redact_annot(mask_rect, fill=(1, 1, 1))
                        page_redactions += 1
                        repeated_text_redactions_total += 1

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text:
                        continue
                    lowered = text.lower()
                    if not any(keyword in lowered for keyword in keyword_set):
                        continue
                    rect = fitz.Rect(span["bbox"])
                    mask_rect = _register_mask_rect(mask_rects, index - 1, rect, page_bounds)
                    if mask_rect is None:
                        continue
                    if processing_mode != "raster":
                        page.add_redact_annot(mask_rect, fill=(1, 1, 1))
                        page_redactions += 1

        if page_redactions and processing_mode != "raster":
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            redactions_applied_total += page_redactions

        if progress_callback and total_pages:
            span_start, span_end = processing_span
            span_delta = max(0.0, span_end - span_start)
            fraction = index / total_pages
            progress_value = span_start + span_delta * fraction
            progress_callback(
                "process_pages",
                min(progress_value, span_end),
                f"处理中：第 {index}/{total_pages} 页",
            )

    total_masks = sum(len(rects) for rects in mask_rects.values())
    vector_changes = (
        annotations_removed_total
        + redactions_applied_total
        + background_images_removed_total
        + repeated_text_redactions_total
    )

    if processing_mode == "raster":
        if total_masks:
            rasterized_pages_total = _apply_raster_overlays(
                doc,
                mask_rects,
                progress_callback=progress_callback,
                progress_span=(0.85, 0.95),
            )
    elif processing_mode == "auto" and vector_changes == 0 and total_masks:
        if progress_callback:
            progress_callback(
                "rasterize",
                0.6,
                "未检测到可直接移除的水印，尝试图像覆盖模式",
            )
        rasterized_pages_total = _apply_raster_overlays(
            doc,
            mask_rects,
            progress_callback=progress_callback,
            progress_span=(0.6, 0.9),
        )

    logger.info(
        (
            "Watermark removal completed. %s annotations removed, %s keyword redactions applied, "
            "%s background images deleted, %s repeated text redactions applied, %s pages rasterized"
        ),
        annotations_removed_total,
        redactions_applied_total,
        background_images_removed_total,
        repeated_text_redactions_total,
        rasterized_pages_total,
    )
    if progress_callback:
        progress_callback("finalize", 0.97, "正在生成结果文件")

    output = io.BytesIO()
    doc.save(output, incremental=False, deflate=True)
    if progress_callback:
        progress_callback("complete", 1.0, "PDF 处理完成")
    return output.getvalue()


def _register_mask_rect(
    rect_map: Dict[int, List[fitz.Rect]],
    page_index: int,
    rect: fitz.Rect,
    bounds: fitz.Rect,
    padding: float = 1.5,
) -> Optional[fitz.Rect]:
    if rect.x1 <= rect.x0 or rect.y1 <= rect.y0:
        return None
    candidate = fitz.Rect(rect)
    candidate.x0 = max(bounds.x0, candidate.x0 - padding)
    candidate.y0 = max(bounds.y0, candidate.y0 - padding)
    candidate.x1 = min(bounds.x1, candidate.x1 + padding)
    candidate.y1 = min(bounds.y1, candidate.y1 + padding)
    if candidate.x1 <= candidate.x0 or candidate.y1 <= candidate.y0:
        return None
    target_list = rect_map[page_index]
    for existing in target_list:
        if (
            abs(existing.x0 - candidate.x0) < 0.5
            and abs(existing.y0 - candidate.y0) < 0.5
            and abs(existing.x1 - candidate.x1) < 0.5
            and abs(existing.y1 - candidate.y1) < 0.5
        ):
            return existing
    target_list.append(candidate)
    return candidate


def _apply_raster_overlays(
    doc: fitz.Document,
    rect_map: Dict[int, List[fitz.Rect]],
    progress_callback: Optional[ProgressCallback] = None,
    progress_span: Tuple[float, float] = (0.6, 0.9),
    scale: float = 2.0,
) -> int:
    pages = [index for index, rects in rect_map.items() if rects]
    if not pages:
        return 0

    span_start, span_end = progress_span
    span_delta = max(0.0, span_end - span_start)
    processed = 0

    for idx, page_index in enumerate(sorted(pages), start=1):
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        mode = "RGBA" if pix.n == 4 else "RGB"
        image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        draw = ImageDraw.Draw(image)
        scale_x = pix.width / page.rect.width
        scale_y = pix.height / page.rect.height
        fill_color = (255, 255, 255, 255) if mode == "RGBA" else (255, 255, 255)

        for rect in rect_map[page_index]:
            x0 = max(0, int(math.floor(rect.x0 * scale_x)) - 3)
            y0 = max(0, int(math.floor(rect.y0 * scale_y)) - 3)
            x1 = min(pix.width, int(math.ceil(rect.x1 * scale_x)) + 3)
            y1 = min(pix.height, int(math.ceil(rect.y1 * scale_y)) + 3)
            draw.rectangle([x0, y0, x1, y1], fill=fill_color)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        page.clean_contents()
        page.insert_image(page.rect, stream=buffer.getvalue(), keep_proportion=False, overlay=False)
        processed += 1

        if progress_callback:
            fraction = idx / max(len(pages), 1)
            progress_value = span_start + span_delta * fraction
            progress_callback(
                "rasterize",
                min(progress_value, span_end),
                f"图像模式处理第 {page_index + 1}/{len(doc)} 页",
            )

    return processed


__all__ = ["app"]


def _identify_repeated_text_watermarks(
    doc: fitz.Document,
    coverage_threshold: float = 0.08,
    frequency_threshold: float = 0.6,
    progress_callback: Optional[ProgressCallback] = None,
    progress_span: Tuple[float, float] = (0.3, 0.4),
) -> Set[str]:
    if len(doc) < 2:
        return set()

    span_start, span_end = progress_span
    span_delta = max(0.0, span_end - span_start)

    text_usage: Dict[str, Dict[str, object]] = {}

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        page_area = abs(page.rect) or 1.0
        page_dict = page.get_text("dict")
        processed_on_page: Set[str] = set()

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                raw_text = "".join(span.get("text", "") for span in spans)
                normalised = "".join(raw_text.split())
                if len(normalised) < 6:
                    continue
                rect = _union_rects([fitz.Rect(span["bbox"]) for span in spans if "bbox" in span])
                if rect is None:
                    continue
                coverage = abs(rect) / page_area
                if coverage < coverage_threshold:
                    continue
                if normalised in processed_on_page:
                    continue

                entry = text_usage.setdefault(
                    normalised,
                    {"pages": set(), "coverage": [], "colors": set()},
                )
                pages = cast(Set[int], entry["pages"])
                pages.add(page.number)
                coverages = entry["coverage"]
                coverages.append(coverage)
                colors = entry["colors"]
                for span in spans:
                    color = span.get("color")
                    if isinstance(color, (list, tuple)) and len(color) >= 3:
                        colors.add(tuple(round(float(c), 2) for c in color[:3]))
                processed_on_page.add(normalised)

        if progress_callback:
            fraction = (page_index + 1) / len(doc)
            progress_value = span_start + span_delta * fraction
            progress_callback(
                "scan_text",
                min(progress_value, span_end),
                f"扫描文本水印候选 ({page_index + 1}/{len(doc)})",
            )

    minimum_pages = max(2, math.ceil(len(doc) * frequency_threshold))
    candidates: Set[str] = set()

    for text, data in text_usage.items():
        pages = cast(Set[int], data["pages"])
        if len(pages) < minimum_pages:
            continue
        coverages = data["coverage"]
        average_coverage = mean(coverages) if coverages else 0.0
        if average_coverage < coverage_threshold:
            continue
        candidates.add(text)

    if progress_callback:
        progress_callback(
            "scan_text",
            span_end,
            f"文本水印候选分析完成，识别到 {len(candidates)} 项",
        )

    return candidates


def _identify_background_images(
    doc: fitz.Document,
    coverage_threshold: float = 0.5,
    frequency_threshold: float = 0.8,
    progress_callback: Optional[ProgressCallback] = None,
    progress_span: Tuple[float, float] = (0.0, 0.3),
) -> List[int]:
    """Identify images that likely represent repeated watermarks."""
    if len(doc) == 0:
        return []

    image_usage: Dict[int, Dict[str, object]] = {}
    total_pages = len(doc)
    span_start, span_end = progress_span
    span_delta = max(0.0, span_end - span_start)

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        page_area = abs(page.rect)
        page_width = max(1.0, page.rect.width)
        page_height = max(1.0, page.rect.height)
        if not page_area:
            continue

        for img in page.get_images(full=True):
            xref = img[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue

            total_rect_area = sum(abs(rect) for rect in rects)
            coverage = total_rect_area / page_area

            max_width_ratio = max(rect.width / page_width for rect in rects)
            max_height_ratio = max(rect.height / page_height for rect in rects)

            if (
                coverage < coverage_threshold
                and coverage < 0.01
                and max_width_ratio < 0.35
                and max_height_ratio < 0.2
            ):
                continue

            entry = image_usage.setdefault(
                xref,
                {
                    "count": 0,
                    "pages": set(),
                    "coverages": [],
                    "max_width_ratios": [],
                    "max_height_ratios": [],
                },
            )
            pages = cast(Set[int], entry["pages"])
            if page.number not in pages:
                entry["count"] = int(entry["count"]) + 1
                pages.add(page.number)
                entry["coverages"].append(coverage)
                entry["max_width_ratios"].append(max_width_ratio)
                entry["max_height_ratios"].append(max_height_ratio)

        if progress_callback and total_pages:
            fraction = (page_index + 1) / total_pages
            progress_value = span_start + span_delta * fraction
            progress_callback(
                "scan_background",
                min(progress_value, span_end),
                f"扫描背景图像 ({page_index + 1}/{total_pages})",
            )

    if not image_usage:
        if progress_callback:
            progress_callback("scan_background", span_end, "未发现重复背景图像")
        return []

    minimum_pages = max(1, math.ceil(len(doc) * frequency_threshold))
    background_images: List[int] = []
    for xref, data in image_usage.items():
        if data["count"] < minimum_pages:
            continue

        coverages = data.get("coverages", [])
        width_ratios = data.get("max_width_ratios", [])
        height_ratios = data.get("max_height_ratios", [])
        avg_coverage = mean(coverages) if coverages else 0.0
        avg_width_ratio = mean(width_ratios) if width_ratios else 0.0
        avg_height_ratio = mean(height_ratios) if height_ratios else 0.0

        if (
            avg_coverage >= coverage_threshold
            or (avg_coverage >= 0.01 and avg_width_ratio >= 0.35)
            or (avg_coverage >= 0.008 and avg_height_ratio >= 0.5)
        ):
            background_images.append(xref)

    if progress_callback:
        progress_callback(
            "scan_background",
            span_end,
            f"背景图像扫描完成，共识别 {len(background_images)} 个候选项",
        )

    return background_images


def _union_rects(rects: List[fitz.Rect]) -> Optional[fitz.Rect]:
    valid_rects = [rect for rect in rects if rect is not None and abs(rect) > 0]
    if not valid_rects:
        return None
    union = fitz.Rect(valid_rects[0])
    for rect in valid_rects[1:]:
        union |= rect
    return union
