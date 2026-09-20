import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

MAX_MEDIA_BYTES = 100 * 1024 * 1024
MAX_MEDIA_DURATION_SECONDS = 2 * 60 * 60
TRANSCRIPT_CHUNK_WORDS = 120
WHISPER_MODEL = os.getenv("NEXUSAI_WHISPER_MODEL", "base")
WHISPER_CACHE = Path(os.getenv("NEXUSAI_WHISPER_CACHE", "./data/models/whisper"))


class TranscriptionError(ValueError):
    pass


@dataclass(frozen=True)
class TimedPassage:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    passages: list[TimedPassage]
    duration_seconds: float
    language: str


def group_segments(segments) -> list[TimedPassage]:
    passages: list[TimedPassage] = []
    texts: list[str] = []
    word_count = 0
    start = 0.0
    end = 0.0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if not texts:
            start = float(segment.start)
        texts.append(text)
        word_count += len(text.split())
        end = float(segment.end)
        if word_count >= TRANSCRIPT_CHUNK_WORDS:
            passages.append(TimedPassage(start=start, end=end, text=" ".join(texts)))
            texts = []
            word_count = 0
    if texts:
        passages.append(TimedPassage(start=start, end=end, text=" ".join(texts)))
    return passages


class TranscriptionService:
    """Runs Whisper out of process so native media libraries and model memory stay isolated."""

    def __init__(self) -> None:
        self._transcribe_lock = Lock()

    @property
    def loaded(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return WHISPER_MODEL

    def transcribe(self, data: bytes, suffix: str) -> TranscriptionResult:
        if not data:
            raise TranscriptionError("The uploaded media file is empty")
        if len(data) > MAX_MEDIA_BYTES:
            raise TranscriptionError("Media exceeds the 100 MB local transcription limit")

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                temporary.write(data)
                temporary_path = temporary.name
            with self._transcribe_lock:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "app.transcription_worker",
                        temporary_path,
                        WHISPER_MODEL,
                        str(WHISPER_CACHE),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=7200,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise TranscriptionError("Local transcription exceeded the two-hour processing limit") from exc
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)

        try:
            payload = json.loads(completed.stdout)
        except (json.JSONDecodeError, UnboundLocalError) as exc:
            raise TranscriptionError("The local transcription worker failed") from exc
        if completed.returncode != 0:
            raise TranscriptionError(payload.get("error", "The media file could not be transcribed"))
        passages = [TimedPassage(**item) for item in payload["passages"]]
        return TranscriptionResult(
            text="\n\n".join(passage.text for passage in passages),
            passages=passages,
            duration_seconds=payload["duration_seconds"],
            language=payload["language"],
        )
