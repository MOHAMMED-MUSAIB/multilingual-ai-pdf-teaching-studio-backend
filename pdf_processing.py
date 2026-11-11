from __future__ import annotations

import unicodedata
from typing import Any

import pymupdf
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from fastapi import HTTPException
from sqlalchemy.engine import RowMapping

from config import PDF_OCR_LANGUAGES, logger
from database import database

def document_or_404(document_id: str) -> RowMapping:
    with database() as con:
        row = con.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Document not found.")
    return row

def page_payload(row: RowMapping) -> dict:
    text = row["extracted_text"]
    return {
        "page_number": row["page_number"],
        "text": text,
        "has_text": bool(text.strip()),
    }

def _is_urdu_pdf(filename: str | None) -> bool:
    if not filename:
        return False
    normalized_name = filename.casefold()
    return "urdu" in normalized_name or "\u0627\u0631\u062f\u0648" in normalized_name

def _has_sufficient_arabic_text(text: str) -> bool:
    alphabetic_characters = [character for character in text if character.isalpha()]
    arabic_characters = sum(
        unicodedata.name(character, "").startswith("ARABIC ")
        for character in alphabetic_characters
    )
    return arabic_characters >= max(3, len(alphabetic_characters) * 0.2)

def _dominant_ocr_language(text: str, languages: list[str]) -> str | None:
    alphabetic_characters = [character for character in text if character.isalpha()]
    if not alphabetic_characters:
        return None

    script_counts = {
        script: sum(unicodedata.name(character, "").startswith(script) for character in alphabetic_characters)
        for script in ("ARABIC", "KANNADA", "LATIN")
    }
    script_thresholds = {"ARABIC": 0.2, "KANNADA": 0.3, "LATIN": 0.6}
    for script, language in (("ARABIC", "urd"), ("KANNADA", "kan"), ("LATIN", "eng")):
        if language in languages and script_counts[script] >= max(
            5, len(alphabetic_characters) * script_thresholds[script]
        ):
            return language
    return None

def _preprocess_ocr_image(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    contrasted = ImageOps.autocontrast(grayscale, cutoff=1)
    sharpened = ImageEnhance.Sharpness(contrasted).enhance(1.2)
    return sharpened.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))

def _extract_preprocessed_ocr(image: Image.Image, language: str) -> str:
    processed_image = _preprocess_ocr_image(image)
    return pytesseract.image_to_string(
        processed_image,
        lang=language,
        config="--oem 1 --psm 6",
    ).strip()

def _extract_page_ocr(page: pymupdf.Page, language: str) -> str:
    pixmap = page.get_pixmap(dpi=400, colorspace=pymupdf.csGRAY, alpha=False)
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    return _extract_preprocessed_ocr(image, language)

def _needs_pdf_ocr(text: str, filename: str | None = None) -> bool:
    if not text.strip():
        return True

    if _is_urdu_pdf(filename) and not _has_sufficient_arabic_text(text):
        return True

    non_whitespace = [character for character in text if not character.isspace()]
    kannada_legacy_markers = {
        character
        for character in non_whitespace
        if character in "\u00c0\u00aa\u00c4\u00a3\u00b8\u00a7\u00a5\u00b5\u00ba\u00b1\u00b2"
    }
    if len(kannada_legacy_markers) >= 3 and sum(
        character in kannada_legacy_markers for character in non_whitespace
    ) >= max(6, len(non_whitespace) * 0.05):
        return True

    script_counts: dict[str, int] = {}
    for character in text:
        if character.isalpha():
            script = unicodedata.name(character, "").split(" ", 1)[0]
            if script:
                script_counts[script] = script_counts.get(script, 0) + 1
    if "ARABIC" in script_counts:
        unexpected_scripts = sum(
            count for script, count in script_counts.items()
            if script not in {"ARABIC", "LATIN"}
        )
        if len(script_counts) >= 3 and unexpected_scripts >= 5:
            return True
    return False

def _extract_pdf_page_ocr(page: pymupdf.Page, filename: str | None = None) -> str:
    languages = list(dict.fromkeys(
        language.strip()
        for language in PDF_OCR_LANGUAGES.split("+")
        if language.strip()
    ))
    language_config = "+".join(languages) or PDF_OCR_LANGUAGES
    prefer_urdu = _is_urdu_pdf(filename) and "urd" in languages
    fallback_languages = list(languages)
    if prefer_urdu:
        fallback_languages.remove("urd")
        fallback_languages.insert(0, "urd")

    def extract(language: str, *, preprocess: bool = False) -> str:
        if preprocess:
            return _extract_page_ocr(page, language)
        text_page = page.get_textpage_ocr(language=language, dpi=300, full=True)
        return page.get_text("text", textpage=text_page).strip()

    if prefer_urdu:
        return extract("urd", preprocess=True)

    try:
        extracted_text = extract(language_config)
    except (OSError, RuntimeError, ValueError) as configured_error:
        if len(languages) < 2:
            raise
        failures: list[Exception] = []
        for language in fallback_languages:
            try:
                return extract(language, preprocess=language == "urd")
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError(
                f"OCR failed with configured languages '{language_config}' "
                f"and each configured language individually: {failures[-1]}"
            ) from configured_error
        raise

    focused_language = _dominant_ocr_language(extracted_text, languages)
    if focused_language:
        try:
            focused_text = extract(focused_language, preprocess=True)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Could not refine OCR with %s language data: %s", focused_language, exc)
        else:
            if focused_language != "urd" or _has_sufficient_arabic_text(focused_text):
                return focused_text
    return extracted_text

