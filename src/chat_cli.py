"""chat_cli.py – Interactive REPL for the research agent.

Launches a terminal-based read-eval-print loop that accepts natural-language
research questions, routes them through the LangGraph agent pipeline, and
prints the answer, judge evaluation, and session metadata.

Usage::

    python -m src.chat_cli
    python -m src.chat_cli --session feedback_loops --max-iters 3

The session persists across questions within a single run, so follow-up
queries can build on earlier context.  If no ``--session`` is provided,
the agent defaults to today's date as the session identifier.
"""

import argparse
from dotenv import load_dotenv
from src.agent_graph import make_graph


def main() -> None:
    """Entry point for the interactive research-agent CLI.

    Parses command-line arguments, builds the LangGraph agent, and enters
    an input loop that:
      1. Reads a question from stdin.
      2. Invokes the agent graph with the question, session id, and
         iteration cap.
      3. Captures the returned session id (auto-assigned on first run
         when none is provided).
      4. Prints the answer, judge verdict, session id, and run id.

    The loop exits when the user types ``exit`` or ``quit``.
    """
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="Interactive REPL for the research agent."
    )
    ap.add_argument("--session", type=str, default="", help="session id/name (e.g., feedback_loops). Default=today.")
    ap.add_argument("--max-iters", type=int, default=2)
    args = ap.parse_args()

    app = make_graph()

    print("Interactive Research Agent")
    print("Type 'exit' to quit.\n")

    session_id = args.session.strip()  # if empty, agent will default to today in init()

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        out = app.invoke({
            "question": q,
            "session_id": session_id,
            "max_iters": args.max_iters,
        })

        # capture session_id assigned on first run
        session_id = out.get("session_id") or session_id

        if out["answer"].startswith("CLARIFY:"):
            print("Agent:", out["answer"].replace("CLARIFY:", "").strip())
            extra = input("You (clarification): ").strip()
            if extra:
                q = q + "\nClarification: " + extra
                continue

        print("\nAgent:\n", out["answer"])
        print("\nJudge:", out["judge"])
        print("Session:", session_id, "Run:", out["run_id"])
        print("-" * 80)


if __name__ == "__main__":
    main()