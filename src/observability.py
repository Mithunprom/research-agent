"""observability.py – In-process metrics and request logging for the API.

Provides two facilities used by ``agentic_rag_api.py``:

1. **Metrics** – A lightweight in-memory counter that tracks request count,
   cumulative latency, clarification rate, and average judge score.
   Exposed via ``GET /metrics``.

2. **Request logging** – Appends one JSON line per request to
   ``data/logs/requests.jsonl`` for offline analysis and debugging.

These are intentionally simple (no Prometheus, no external DB) so the
project stays self-contained. For production use, swap in a real
metrics backend.
"""

import json
import time
from pathlib import Path
from fastapi import Request

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "requests.jsonl"


class Metrics:
    """In-memory request metrics accumulator.

    Tracks total request count, cumulative latency, number of
    clarification responses, and sum of judge scores. All values
    reset when the server restarts.
    """

    def __init__(self) -> None:
        """Initialise all counters to zero."""
        self.n = 0
        self.total_latency = 0.0
        self.clarify_n = 0
        self.judge_score_sum = 0.0

    def record(self, latency_s: float, is_clarify: bool, judge_score: float | None) -> None:
        """Record metrics for a single completed request.

        Args:
            latency_s: Wall-clock seconds the request took.
            is_clarify: True if the agent returned a clarifying question.
            judge_score: Groundedness score (0-1) from the judge, or None
                         if the judge step was skipped (e.g. clarification).
        """
        self.n += 1
        self.total_latency += latency_s
        if is_clarify:
            self.clarify_n += 1
        if judge_score is not None:
            self.judge_score_sum += float(judge_score)

    def snapshot(self) -> dict:
        """Return a summary dict of all metrics since server start.

        Returns:
            Dict with keys: requests, avg_latency_s, clarify_rate,
            avg_judge_score.
        """
        avg_latency = self.total_latency / self.n if self.n else 0.0
        avg_judge = self.judge_score_sum / self.n if self.n else 0.0
        return {
            "requests": self.n,
            "avg_latency_s": round(avg_latency, 4),
            "clarify_rate": round((self.clarify_n / self.n) if self.n else 0.0, 4),
            "avg_judge_score": round(avg_judge, 4),
        }


metrics = Metrics()


async def log_request(
    req: Request,
    payload: dict,
    resp_payload: dict,
    latency_s: float,
) -> None:
    """Append a structured log line for a completed API request.

    Writes one JSON object per line to ``data/logs/requests.jsonl``
    containing the timestamp, request path, latency, input payload,
    and response metadata (whether an answer was returned and how
    many sources were cited).

    Args:
        req: The incoming FastAPI Request object.
        payload: The deserialized request body.
        resp_payload: The response dict returned to the client.
        latency_s: Wall-clock seconds the request took.
    """
    rec = {
        "ts": time.time(),
        "path": str(req.url.path),
        "latency_s": round(latency_s, 4),
        "payload": payload,
        "response_meta": {
            "has_answer": "answer" in resp_payload,
            "sources_n": len(resp_payload.get("sources", []) or resp_payload.get("citations", []) or []),
        },
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")