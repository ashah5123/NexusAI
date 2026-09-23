from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from shutil import which


class SpeechError(Exception):
    """Raised when local speech synthesis cannot produce audio."""


class LocalSpeechService:
    def __init__(self) -> None:
        self.engine = "macos-say"
        self._binary = which("say")

    def available(self) -> bool:
        return self._binary is not None

    def voices(self) -> list[str]:
        if not self._binary:
            return []
        try:
            result = subprocess.run(
                [self._binary, "-v", "?"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        names: list[str] = []
        seen: set[str] = set()
        for line in result.stdout.splitlines():
            name = line.split(None, 1)[0].strip()
            if name and name not in seen:
                names.append(name)
                seen.add(name)
        return names

    def synthesize(self, text: str, voice: str | None, rate: int) -> Path:
        if not self._binary:
            raise SpeechError("Local neural speech is unavailable on this machine")
        output = tempfile.NamedTemporaryFile(prefix="nexusai-speech-", suffix=".aiff", delete=False)
        output_path = Path(output.name)
        output.close()
        command = [self._binary, "-r", str(rate), "-o", str(output_path)]
        if voice:
            command.extend(["-v", voice])
        command.append(text)
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            output_path.unlink(missing_ok=True)
            raise SpeechError("Local speech synthesis failed") from exc
        return output_path
