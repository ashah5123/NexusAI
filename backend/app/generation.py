from __future__ import annotations

import json
import os
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from .models import AnswerResponse, Citation


class OllamaAnswerService:
    def __init__(self) -> None:
        self.base_url = os.getenv("NEXUSAI_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model_name = os.getenv("NEXUSAI_ANSWER_MODEL", "qwen3:1.7b")

    def available(self) -> bool:
        try:
            with urlopen(f"{self.base_url}/api/tags", timeout=1.5) as response:
                data = json.load(response)
            return any(model.get("name", "").split(":")[0] == self.model_name.split(":")[0]
                       for model in data.get("models", []))
        except (OSError, URLError, ValueError):
            return False

    def answer(self, question: str, passages: list[dict], warning: str | None = None) -> AnswerResponse:
        started = time.perf_counter()
        citations = [
            Citation(
                number=index,
                document_id=row["id"],
                title=row["title"],
                source_type=row["source_type"],
                page_number=row.get("page_number"),
                start_seconds=row.get("start_seconds"),
                end_seconds=row.get("end_seconds"),
                passage=row["passage"],
            )
            for index, row in enumerate(passages, 1)
        ]
        if not passages:
            return AnswerResponse(
                question=question,
                answer="I could not find relevant evidence in your library.",
                citations=[],
                model=self.model_name,
                generated=False,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                warning=warning,
            )

        context = "\n\n".join(
            f"[{index}] {row['title']}\n{row['passage']}"
            for index, row in enumerate(passages, 1)
        )
        payload = {
            "model": self.model_name,
            "stream": False,
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied library evidence. Cite every factual claim "
                        "with its evidence number in square brackets, such as [1]. If the evidence "
                        "does not answer the question, say so plainly. Do not use outside knowledge."
                    ),
                },
                {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
            ],
            "options": {"temperature": 0.1, "num_ctx": 8192},
        }
        try:
            request = Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=120) as response:
                generated = json.load(response)["message"]["content"].strip()
            answer = generated or "I could not produce an answer from the available evidence."
            did_generate = bool(generated)
        except (OSError, URLError, ValueError, KeyError):
            answer = "The local answer model is unavailable. Review the relevant evidence below."
            did_generate = False
            warning = "Start Ollama to generate an answer; retrieval is still available."

        return AnswerResponse(
            question=question,
            answer=answer,
            citations=citations,
            model=self.model_name,
            generated=did_generate,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            warning=warning,
        )
