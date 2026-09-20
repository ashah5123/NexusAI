from io import BytesIO

from pypdf import PdfReader

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 500


class IngestionError(ValueError):
    pass


def extract_pdf(data: bytes) -> list[tuple[int, str]]:
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
        if len(reader.pages) > MAX_PDF_PAGES:
            raise IngestionError(f"PDF exceeds the {MAX_PDF_PAGES}-page local limit")
        pages = [(index, (page.extract_text() or "").strip()) for index, page in enumerate(reader.pages, 1)]
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("The PDF could not be parsed") from exc

    pages = [(number, text) for number, text in pages if text]
    if not pages:
        raise IngestionError("No searchable text was found; scanned PDFs require OCR")
    return pages
