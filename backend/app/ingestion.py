from dataclasses import dataclass
from io import BytesIO

import pymupdf
from pypdf import PdfReader

from .ocr import OCRError, OCRService

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_OCR_PAGES = 50
MIN_EMBEDDED_TEXT_CHARS = 24


class IngestionError(ValueError):
    pass


@dataclass(frozen=True)
class ExtractionResult:
    pages: list[tuple[int, str]]
    ocr_pages: int = 0
    total_pages: int = 1


def extract_pdf(data: bytes, ocr: OCRService | None = None) -> ExtractionResult:
    if not data:
        raise IngestionError("The uploaded PDF is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestionError("PDF exceeds the 20 MB local upload limit")
    if not data.startswith(b"%PDF-"):
        raise IngestionError("The uploaded file is not a valid PDF")

    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise IngestionError("Password-protected PDFs are not supported")
        total_pages = len(reader.pages)
        if total_pages > MAX_PDF_PAGES:
            raise IngestionError(f"PDF exceeds the {MAX_PDF_PAGES}-page local limit")
        extracted = {
            index: (page.extract_text() or "").strip()
            for index, page in enumerate(reader.pages, 1)
        }
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("The PDF could not be parsed") from exc

    pages_needing_ocr = [
        number for number, text in extracted.items() if len(text) < MIN_EMBEDDED_TEXT_CHARS
    ]
    ocr_succeeded: set[int] = set()
    if pages_needing_ocr and ocr is not None:
        if len(pages_needing_ocr) > MAX_OCR_PAGES:
            raise IngestionError(
                f"PDF requires OCR on more than {MAX_OCR_PAGES} pages; split it into smaller files"
            )
        try:
            with pymupdf.open(stream=data, filetype="pdf") as document:
                for page_number in pages_needing_ocr:
                    page = document[page_number - 1]
                    pixmap = page.get_pixmap(dpi=180, colorspace=pymupdf.csRGB, alpha=False)
                    try:
                        extracted[page_number] = ocr.extract(pixmap.tobytes("png"))
                        ocr_succeeded.add(page_number)
                    except OCRError:
                        pass
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError("Scanned PDF pages could not be rendered for OCR") from exc

    pages = [(number, text) for number, text in extracted.items() if text]
    if not pages:
        if ocr is None:
            raise IngestionError("No searchable text was found; enable OCR for scanned PDFs")
        raise IngestionError("No readable text was detected in the scanned PDF")
    return ExtractionResult(
        pages=pages,
        ocr_pages=len(ocr_succeeded),
        total_pages=total_pages,
    )


def extract_image(data: bytes, ocr: OCRService) -> ExtractionResult:
    if not data:
        raise IngestionError("The uploaded image is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestionError("Image exceeds the 20 MB local upload limit")
    try:
        text = ocr.extract(data)
    except OCRError as exc:
        raise IngestionError(str(exc)) from exc
    return ExtractionResult(pages=[(1, text)], ocr_pages=1, total_pages=1)
