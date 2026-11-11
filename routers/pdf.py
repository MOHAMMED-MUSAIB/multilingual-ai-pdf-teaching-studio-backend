"""Pdf API routes."""

import io
import logging
import uuid
from pathlib import Path

import pymupdf
from PIL import Image
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from services import (
    DB_PATH, MAX_PDF_BYTES, MAX_PDF_PAGES, PDF_DIR, PDF_OCR_LANGUAGES,
    _dominant_ocr_language, _extract_pdf_page_ocr, _extract_preprocessed_ocr,
    _has_sufficient_arabic_text, _needs_pdf_ocr, database, document_or_404,
    logger, page_payload, ReadingPosition, utc_now,
)

router = APIRouter()

@router.post("/api/pdf/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict:
    original_name = Path((file.filename or "").replace("\\", "/")).name
    if not original_name or not original_name.lower().endswith(".pdf"):
        raise HTTPException(415, "Choose a PDF file.")
    content = await file.read(MAX_PDF_BYTES + 1)
    await file.close()
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(413, f"PDF is too large. Maximum size is {MAX_PDF_BYTES // (1024 * 1024)} MB.")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(415, "The selected file is not a valid PDF.")
    try:
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            if pdf.is_encrypted:
                raise HTTPException(422, "This PDF is password-protected. Unlock it before uploading.")
            total_pages = pdf.page_count
            if total_pages < 1:
                raise HTTPException(422, "This PDF has no pages.")
            if total_pages > MAX_PDF_PAGES:
                raise HTTPException(413, f"PDF has too many pages. Maximum is {MAX_PDF_PAGES}.")
            page_texts = []
            ocr_warning = None
            for page_number in range(1, total_pages + 1):
                page = pdf.load_page(page_number - 1)
                extracted_text = page.get_text("text", sort=True).strip()
                if _needs_pdf_ocr(extracted_text, original_name):
                    try:
                        ocr_text = _extract_pdf_page_ocr(page, original_name)
                    except (OSError, RuntimeError, ValueError) as exc:
                        if ocr_warning is None:
                            logger.warning(
                                "PDF OCR failed on page %s: %s", page_number, exc
                            )
                        ocr_warning = (
                            "Some pages need OCR. Check that Tesseract and the "
                            "required language data are installed. Urdu PDFs need "
                            "the Urdu (urd) language data; reopen the page to retry."
                        )
                        extracted_text = ""
                    else:
                        extracted_text = ocr_text
                page_texts.append((page_number, extracted_text))
    except HTTPException:
        raise
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError, ValueError) as exc:
        raise HTTPException(422, "Could not read this PDF. It may be damaged or unsupported.") from exc
    document_id = str(uuid.uuid4())
    stored_path = PDF_DIR / f"{document_id}.pdf"
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = stored_path.with_suffix(".tmp")
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(stored_path)
        with database() as con:
            con.execute(
                "INSERT INTO documents(id, filename, file_path, total_pages, uploaded_at, last_read_page) VALUES(?,?,?,?,?,1)",
                (document_id, original_name, str(stored_path), total_pages, utc_now()),
            )
            con.executemany(
                "INSERT INTO pdf_pages(document_id, page_number, extracted_text) VALUES(?,?,?)",
                [(document_id, page_number, text) for page_number, text in page_texts],
            )
    except Exception:
        stored_path.unlink(missing_ok=True)
        temporary_path.unlink(missing_ok=True)
        logger.exception("Failed to persist uploaded PDF")
        raise HTTPException(500, "Could not save this PDF to local storage.")
    text_pages = sum(bool(text) for _, text in page_texts)
    return {
        "id": document_id,
        "filename": original_name,
        "total_pages": total_pages,
        "last_read_page": 1,
        "text_pages": text_pages,
        "ocr_required": text_pages == 0,
        "ocr_warning": ocr_warning,
    }

@router.get("/api/pdf")
def list_documents() -> list[dict]:
    with database() as con:
        rows = con.execute(
            """SELECT d.id, d.filename, d.total_pages, d.uploaded_at, d.last_read_page,
                      COUNT(CASE WHEN length(trim(p.extracted_text)) > 0 THEN 1 END) AS text_pages
               FROM documents d LEFT JOIN pdf_pages p ON p.document_id = d.id
               GROUP BY d.id ORDER BY d.uploaded_at DESC"""
        ).fetchall()
    return [dict(row) for row in rows]

@router.get("/api/pdf/{document_id}")
def get_document(document_id: str) -> dict:
    row = document_or_404(document_id)
    with database() as con:
        text_pages = con.execute(
            "SELECT COUNT(*) AS text_pages FROM pdf_pages WHERE document_id=? AND length(trim(extracted_text)) > 0",
            (document_id,),
        ).fetchone()["text_pages"]
    return {
        "id": row["id"],
        "filename": row["filename"],
        "total_pages": row["total_pages"],
        "uploaded_at": row["uploaded_at"],
        "last_read_page": row["last_read_page"],
        "text_pages": text_pages,
        "ocr_required": text_pages == 0,
    }

@router.get("/api/pdf/{document_id}/file")
def get_document_file(document_id: str) -> FileResponse:
    row = document_or_404(document_id)
    path = Path(row["file_path"]).resolve()
    if not path.is_relative_to(PDF_DIR.resolve()) or not path.is_file():
        raise HTTPException(404, "Stored PDF file is missing.")
    return FileResponse(path, media_type="application/pdf", filename=row["filename"])

@router.get("/api/pdf/{document_id}/pages")
def get_pages(document_id: str) -> dict:
    document_or_404(document_id)
    with database() as con:
        rows = con.execute(
            "SELECT page_number, extracted_text FROM pdf_pages WHERE document_id=? ORDER BY page_number",
            (document_id,),
        ).fetchall()
    return {
        "pages": [page_payload(row) for row in rows],
        "ocr_required": not any(row["extracted_text"].strip() for row in rows),
    }

@router.get("/api/pdf/{document_id}/page/{page_number}")
def get_page(document_id: str, page_number: int) -> dict:
    document = document_or_404(document_id)
    if page_number < 1 or page_number > document["total_pages"]:
        raise HTTPException(404, "Page number is outside this document.")
    with database() as con:
        row = con.execute(
            "SELECT page_number, extracted_text FROM pdf_pages WHERE document_id=? AND page_number=?",
            (document_id, page_number),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Page text was not found.")
    payload = page_payload(row)
    if _needs_pdf_ocr(payload["text"], document["filename"]):
        pdf_path = Path(document["file_path"]).resolve()
        if not pdf_path.is_relative_to(PDF_DIR.resolve()) or not pdf_path.is_file():
            raise HTTPException(404, "Stored PDF file is missing.")
        try:
            with pymupdf.open(pdf_path) as pdf:
                extracted_text = _extract_pdf_page_ocr(
                    pdf.load_page(page_number - 1), document["filename"]
                )
            if not extracted_text:
                raise RuntimeError("OCR did not recognize readable text on this page.")
        except (OSError, pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError, ValueError) as exc:
            logger.warning(
                "PDF OCR failed on page %s of document %s: %s",
                page_number,
                document_id,
                exc,
            )
            payload["text"] = ""
            payload["has_text"] = False
            payload["message"] = (
                "OCR is needed for this page, but it could not be completed. Install "
                "Tesseract OCR and the required language data. Urdu PDFs need the "
                f"Urdu (urd) language data. Font files alone are not OCR data. ({exc})"
            )
            return payload
        with database() as con:
            con.execute(
                "UPDATE pdf_pages SET extracted_text=? WHERE document_id=? AND page_number=?",
                (extracted_text, document_id, page_number),
            )
        payload["text"] = extracted_text
        payload["has_text"] = True
    if not payload["has_text"]:
        payload["message"] = (
            "No readable text was extracted. Scanned pages and PDFs with unsupported "
            "font encodings require local Tesseract OCR with the appropriate language data."
        )
    return payload

@router.post("/api/pdf/{document_id}/page/{page_number}/ocr")
def apply_page_ocr(document_id: str, page_number: int) -> dict:
    document = document_or_404(document_id)
    if page_number < 1 or page_number > document["total_pages"]:
        raise HTTPException(404, "Page number is outside this document.")
    pdf_path = Path(document["file_path"]).resolve()
    if not pdf_path.is_relative_to(PDF_DIR.resolve()) or not pdf_path.is_file():
        raise HTTPException(404, "Stored PDF file is missing.")
    try:
        with pymupdf.open(pdf_path) as pdf:
            extracted_text = _extract_pdf_page_ocr(
                pdf.load_page(page_number - 1), document["filename"]
            )
    except (OSError, pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Requested PDF OCR failed on page %s of document %s: %s",
            page_number,
            document_id,
            exc,
        )
        raise HTTPException(
            503,
            "Could not apply OCR. Install Tesseract OCR with the required language "
            f"data; Urdu PDFs need the Urdu (urd) data. ({exc})",
        ) from exc
    if not extracted_text:
        raise HTTPException(422, "OCR did not recognize readable text on this page.")
    with database() as con:
        con.execute(
            "UPDATE pdf_pages SET extracted_text=? WHERE document_id=? AND page_number=?",
            (extracted_text, document_id, page_number),
        )
    return {
        "page_number": page_number,
        "text": extracted_text,
        "has_text": True,
    }

@router.post("/api/ocr/image")
async def extract_image_text(file: UploadFile = File(...)) -> dict:
    if not file.filename or Path(file.filename).suffix.lower() != ".png":
        raise HTTPException(415, "Choose a PNG image crop.")
    content = await file.read(15 * 1024 * 1024 + 1)
    await file.close()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(413, "The image crop is too large. Keep it under 15 MB.")
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(415, "The selected file is not a valid PNG image.")
    try:
        with Image.open(io.BytesIO(content)) as source_image:
            source_image.load()
            image = source_image.convert("L")
        languages = list(dict.fromkeys(
            language.strip()
            for language in PDF_OCR_LANGUAGES.split("+")
            if language.strip()
        ))
        language_config = "+".join(languages) or PDF_OCR_LANGUAGES
        extracted_text = _extract_preprocessed_ocr(image, language_config)
        focused_language = _dominant_ocr_language(extracted_text, languages)
        if focused_language:
            focused_text = _extract_preprocessed_ocr(image, focused_language)
            if focused_language != "urd" or _has_sufficient_arabic_text(focused_text):
                extracted_text = focused_text
    except HTTPException:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Image OCR failed: %s", exc)
        raise HTTPException(
            503,
            "Could not apply OCR. Install Tesseract OCR with the required language "
            f"data; Urdu text needs the Urdu (urd) data. ({exc})",
        ) from exc
    if not extracted_text:
        raise HTTPException(422, "OCR did not recognize readable text in this image crop.")
    return {"text": extracted_text}

@router.get("/api/pdf/{document_id}/status")
def get_document_status(document_id: str) -> dict:
    document = get_document(document_id)
    return {
        "id": document["id"],
        "total_pages": document["total_pages"],
        "text_pages": document["text_pages"],
        "ocr_required": document["ocr_required"],
    }

@router.put("/api/pdf/{document_id}/position")
def update_reading_position(document_id: str, position: ReadingPosition) -> dict:
    document = document_or_404(document_id)
    if position.page > document["total_pages"]:
        raise HTTPException(422, "Page number is outside this document.")
    with database() as con:
        con.execute(
            "UPDATE documents SET last_read_page=? WHERE id=?",
            (position.page, document_id),
        )
    return {"saved": True, "page": position.page}

@router.delete("/api/pdf/{document_id}")
def delete_document(document_id: str) -> dict:
    row = document_or_404(document_id)
    path = Path(row["file_path"]).resolve()
    with database() as con:
        con.execute("DELETE FROM conversations WHERE document_id=?", (document_id,))
        con.execute("DELETE FROM documents WHERE id=?", (document_id,))
    if path.is_relative_to(PDF_DIR.resolve()):
        path.unlink(missing_ok=True)
    return {"deleted": True}
