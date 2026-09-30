from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field


logger = logging.getLogger("nexusai.http")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


@dataclass
class RequestMetrics:
    started_at: float = field(default_factory=time.time)
    requests: int = 0
    duration_seconds: float = 0.0
    statuses: Counter = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, status_code: int, duration_seconds: float) -> None:
        with self._lock:
            self.requests += 1
            self.duration_seconds += duration_seconds
            self.statuses[str(status_code)] += 1

    def prometheus(self) -> str:
        with self._lock:
            lines = [
                "# HELP nexusai_http_requests_total Total HTTP requests.",
                "# TYPE nexusai_http_requests_total counter",
                f"nexusai_http_requests_total {self.requests}",
                "# HELP nexusai_http_request_duration_seconds_total Cumulative request duration.",
                "# TYPE nexusai_http_request_duration_seconds_total counter",
                f"nexusai_http_request_duration_seconds_total {self.duration_seconds:.6f}",
                "# HELP nexusai_process_uptime_seconds API process uptime.",
                "# TYPE nexusai_process_uptime_seconds gauge",
                f"nexusai_process_uptime_seconds {time.time() - self.started_at:.3f}",
            ]
            for status, count in sorted(self.statuses.items()):
                lines.append(f'nexusai_http_responses_total{{status="{status}"}} {count}')
        return "\n".join(lines) + "\n"


metrics = RequestMetrics()


def log_request(**fields) -> None:
    logger.info(json.dumps(fields, separators=(",", ":"), default=str))
