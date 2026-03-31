"""agentic_rag_api.py – FastAPI web server for the research agent.

Exposes the LangGraph research pipeline as a REST API so external
clients (frontends, Slack bots, notebooks) can submit questions and
receive structured answers without running the CLI.

Endpoints:
    GET  /health       – Liveness check (returns {"status": "ok"}).
    GET  /metrics      – Aggregated runtime metrics (request count,
                         avg latency, clarify rate, avg judge score).
    POST /agent_query  – Run the full research pipeline for a question.

Usage::

    uvicorn src.agentic_rag_api:app --host 0.0.0.0 --port 8005

Request body for /agent_query::

    {
        "question": "What is continual learning?",
        "session_id": "",          // optional, defaults to today's date
        "max_iters": 2             // optional, refinement loop cap
    }

Response::

    {
        "answer": "...",
        "judge": {"verdict": "PASS", "score": 0.85, ...},
        "sources": [...],
        "session_id": "2026-03-26",
        "run_id": "a1b2c3d4e5"
    }
"""

from fastapi import FastAPI, Request
from pydantic import BaseModel
import time
from dotenv import load_dotenv

load_dotenv()

from src.agent_graph import make_graph
from src.observability import metrics, log_request

app = FastAPI(
    title="Research Agent API",
    description="Adaptive research agent with RAG + web search",
)
agent = make_graph()


class AgentQueryRequest(BaseModel):
    """Request body for the /agent_query endpoint.

    Attributes:
        question: The research question to answer.
        session_id: Session identifier for conversation continuity.
                    Defaults to today's date if empty.
        max_iters: Maximum number of judge-refine iterations (default 2).
    """

    question: str
    session_id: str = ""
    max_iters: int = 2


@app.get("/health")
def health():
    """Liveness probe. Returns ``{"status": "ok"}`` when the server is up."""
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    """Return aggregated runtime metrics since server start.

    Returns a dict with: request count, average latency, clarification
    rate, and average judge score.
    """
    return metrics.snapshot()


@app.post("/agent_query")
async def agent_query(req: Request, body: AgentQueryRequest):
    """Run the full research agent pipeline and return the result.

    Invokes the LangGraph agent with the given question, records
    latency and judge metrics, logs the request to disk, and returns
    the answer with judge verdict, sources, session_id, and run_id.
    """
    t0 = time.time()
    out = agent.invoke({
        "question": body.question,
        "session_id": body.session_id,
        "max_iters": body.max_iters,
    })
    latency = time.time() - t0

    is_clarify = isinstance(out.get("answer", ""), str) and out["answer"].startswith("CLARIFY:")
    judge_score = None
    if isinstance(out.get("judge", {}), dict):
        judge_score = out["judge"].get("score", None)

    metrics.record(latency, is_clarify, judge_score)

    resp = {
        "answer": out.get("answer", ""),
        "judge": out.get("judge", {}),
        "sources": out.get("citations", []),
        "session_id": out.get("session_id", ""),
        "run_id": out.get("run_id", ""),
    }

    await log_request(req, payload=body.model_dump(), resp_payload=resp, latency_s=latency)
    return resp