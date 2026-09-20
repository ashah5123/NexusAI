from io import BytesIO
from threading import Lock

from PIL import Image, ImageOps, UnidentifiedImageError
from rapidocr import RapidOCR

MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_EDGE = 3200


class OCRError(ValueError):
    pass


class OCRService:
    """Lazy in-process OCR using compact ONNX models bundled with RapidOCR."""

    def __init__(self) -> None:
        self._engine: RapidOCR | None = None
        self._lock = Lock()

    def _get_engine(self) -> RapidOCR:
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = RapidOCR()
        return self._engine

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def extract(self, data: bytes) -> str:
        try:
            with Image.open(BytesIO(data)) as opened:
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise OCRError("Image exceeds the 40-megapixel local OCR limit")
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
        except OCRError:
            raise
        except (UnidentifiedImageError, OSError) as exc:
            raise OCRError("The uploaded file is not a supported image") from exc

        result = self._get_engine()(image)
        lines = [text.strip() for text in (result.txts or ()) if text.strip()]
        if not lines:
            raise OCRError("No readable text was detected in the image")
        return "\n".join(lines)
