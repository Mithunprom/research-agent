"""
tools_memory.py – Session-based memory persistence for the research agent.

Provides functions to create, load, and append to date-keyed research
sessions stored on disk. Each session lives in its own directory under
data/sessions/<session_id>/ and contains:
  - notes.md     : accumulated research notes (markdown, append-only)
  - sources.jsonl : one JSON object per line for each source reference

Flow:
    Agent starts a research session
        → default_session_id() returns today's date as the session key
        → session_dir() creates data/sessions/2026-03-17/ if needed
        → load_session() reads back any existing notes + sources
        → As the agent finds information, append_session() appends
          new notes and source references to the session files
        → Next run on the same day picks up where it left off
"""

from pathlib import Path
import json
from datetime import date
from typing import Any, Dict, List

SESS_DIR = Path("data/sessions")


def default_session_id() -> str:
    """Return today's date string as the session identifier.

    Flow: date.today() → "2026-03-17"

    Each day gets its own session directory so research is
    automatically grouped by day without manual session management.
    """
    return str(date.today())


def session_dir(session_id: str) -> Path:
    """Return the directory path for a given session, creating it if needed.

    Flow: session_id "2026-03-17"
        → data/sessions/2026-03-17/
        → mkdir -p (creates parent dirs if missing)
        → returns Path object
    """
    p = SESS_DIR / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_session(session_id: str):
    """Load previously saved notes and sources for a session.

    Flow:
        session_dir("2026-03-17")
            → read data/sessions/2026-03-17/notes.md → str (or "" if missing)
            → read data/sessions/2026-03-17/sources.jsonl
                → parse each line as JSON → list[dict] (or [] if missing)
            → return (notes, sources)

    Returns:
        tuple[str, list[dict]]: (notes_markdown, list_of_source_dicts)
    """
    p = session_dir(session_id)
    notes_path = p / "notes.md"
    sources_path = p / "sources.jsonl"

    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    sources = []
    if sources_path.exists():
        for line in sources_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sources.append(json.loads(line))
    return notes, sources


def append_session(session_id: str, notes_append: str, new_sources: list):
    """Append new research notes and source references to an existing session.

    Flow:
        notes_append (str) → strip → append to notes.md with blank line separator
        new_sources (list[dict]) → serialize each as JSON → append one per line
                                   to sources.jsonl

    Both operations are append-only so previous session content is never
    overwritten. Either argument can be empty/None and will be skipped.

    Args:
        session_id: date-keyed session identifier (e.g. "2026-03-17")
        notes_append: markdown text to add to notes.md
        new_sources: list of source dicts to append to sources.jsonl
    """
    p = session_dir(session_id)
    notes_path = p / "notes.md"
    sources_path = p / "sources.jsonl"
    if notes_append and notes_append.strip():
        with notes_path.open("a", encoding="utf-8") as f:
            f.write("\n\n" + notes_append.strip())

    if new_sources:
        with sources_path.open("a", encoding="utf-8") as f:
            for s in new_sources:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")


def append_turn(
    session_id: str,
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    judge: Dict[str, Any] | None = None,
) -> None:
    """Append a complete Q&A turn to the session's turns log.

    Each turn is stored as a single JSON line in turns.jsonl containing
    the question, answer, up to 10 source references, and an optional
    judge evaluation dict.

    Args:
        session_id: Date-keyed session identifier (e.g. "2026-03-17").
        question: The research question that was asked.
        answer: The generated answer text.
        sources: List of source dicts; truncated to the first 10.
        judge: Optional evaluation/scoring dict from the judge step.
    """
    p = session_dir(session_id)
    turns_path = p / "turns.jsonl"
    rec = {
        "question": question,
        "answer": answer,
        "sources": sources[:10],
        "judge": judge or {},
    }
    with turns_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_recent_turns(session_id: str, n: int = 6) -> List[Dict[str, Any]]:
    """Load the most recent *n* Q&A turns from a session.

    Reads turns.jsonl, keeps only the last *n* non-empty lines, and
    parses each as JSON. Malformed lines are silently skipped.

    Args:
        session_id: Date-keyed session identifier.
        n: Maximum number of recent turns to return (default 6).

    Returns:
        List of turn dicts, oldest first.
    """
    p = session_dir(session_id)
    turns_path = p / "turns.jsonl"
    if not turns_path.exists():
        return []
    lines = turns_path.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in lines if ln.strip()]
    last = lines[-n:]
    out = []
    for ln in last:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def format_turns_for_prompt(turns: List[Dict[str, Any]]) -> str:
    """Format prior Q&A turns into a compact text block for prompt injection.

    Each turn is rendered as a numbered block with the question and a
    truncated answer (max 800 chars) to provide conversational continuity
    without consuming excessive tokens."""
    if not turns:
        return ""
    blocks = []
    for i, t in enumerate(turns, start=1):
        q = (t.get("question") or "").strip()
        a = (t.get("answer") or "").strip()
        a = a[:800]  # cap
        blocks.append(f"TURN {i}\nQ: {q}\nA: {a}")
    return "\n\n".join(blocks)