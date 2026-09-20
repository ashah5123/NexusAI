import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import av
from faster_whisper import WhisperModel

from .transcription import MAX_MEDIA_DURATION_SECONDS, group_segments


def main() -> int:
    media_path, model_name, cache_path = sys.argv[1:4]
    try:
        try:
            with av.open(media_path) as container:
                duration = float(container.duration / av.time_base) if container.duration else None
        except Exception as exc:
            raise ValueError("The media file could not be decoded") from exc
        if duration and duration > MAX_MEDIA_DURATION_SECONDS:
            raise ValueError("Media exceeds the two-hour local transcription limit")

        Path(cache_path).mkdir(parents=True, exist_ok=True)
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=max(1, min(4, os.cpu_count() or 1)),
            num_workers=1,
            download_root=cache_path,
        )
        segments, info = model.transcribe(
            media_path,
            beam_size=3,
            vad_filter=not duration or duration >= 10,
            condition_on_previous_text=False,
        )
        passages = group_segments(list(segments))
        if not passages:
            raise ValueError("No speech was detected in the media file")
        actual_duration = float(info.duration)
        if actual_duration > MAX_MEDIA_DURATION_SECONDS:
            raise ValueError("Media exceeds the two-hour local transcription limit")
        print(
            json.dumps(
                {
                    "passages": [asdict(passage) for passage in passages],
                    "duration_seconds": actual_duration,
                    "language": info.language,
                }
            )
        )
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc) or "The media file could not be transcribed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
