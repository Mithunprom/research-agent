"""
run_cli.py – Command-line interface for the research agent.

Entry point for running a single research query from the terminal.
Loads environment variables, builds the LangGraph agent, runs the full
retrieve → answer → judge → (refine loop) → persist pipeline, and
prints the results to stdout.

Usage:
    python -m src.run_cli "What is quantitative easing?"

Flow:
    CLI args → join into question string
        → load_dotenv() reads .env (API keys for OpenAI, Tavily, etc.)
        → make_graph() builds the LangGraph research agent
        → app.invoke({"question": ...}) runs the full graph:
            init → retrieve_local → [maybe web] → compose → answer
            → judge → [refine loop if needed] → write_notes → persist
        → Print answer, judge verdict/score, source titles, and run/session IDs

Output is saved to:
    data/runs/<run_id>/answer.md, judge.json, citations.json
    data/sessions/<date>/notes.md, sources.jsonl
"""

import sys
from dotenv import load_dotenv
from src.agent_graph import make_graph


def main():
    """Parse the CLI question, run the research agent, and print results.

    Flow:
        sys.argv[1:] → join into a single question string
            → If empty, print usage and exit
            → load_dotenv() loads .env for API keys
            → make_graph() compiles the LangGraph agent
            → app.invoke() runs the full pipeline (retrieve, answer, judge, persist)
            → Print to stdout:
                - The generated answer (with [1], [2] citations)
                - Judge verdict and groundedness score
                - List of source titles (up to 6)
                - run_id and session_id for locating saved artifacts
    """
    load_dotenv()
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        print('Usage: python -m src.run_cli "your question"')
        return

    app = make_graph()
    out = app.invoke({"question": q, "session_id": ""})

    print("\n✅ ANSWER\n")
    print(out["answer"])
    print("\n✅ JUDGE\n", out["judge"])
    print("\n✅ SOURCES\n", [c.get("title") or c.get("source") for c in out["citations"][:6]])
    print("\nSaved run_id:", out["run_id"], " session_id:", out["session_id"])


if __name__ == "__main__":
    main()