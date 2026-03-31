"""eval_smoke.py – Lightweight evaluation gate for CI.

Runs a small sample of questions from an eval set through the full
research agent pipeline and checks that answer quality meets minimum
thresholds. Designed to run in CI (see ``.github/workflow/ci.yml``)
as a fast smoke test before merging.

Flow::

    Load eval_set.jsonl (up to N_EXAMPLES questions)
        -> For each question, invoke the agent graph
        -> Collect judge verdict and score
        -> Compute avg_judge_score and pass_rate
        -> Write report to artifacts/eval_smoke_report.json
        -> Exit non-zero if thresholds are not met

Environment variables (all optional):
    EVAL_N          – Number of examples to run (default 10)
    MIN_AVG_JUDGE   – Minimum average judge score to pass (default 0.75)
    MIN_PASS_RATE   – Minimum fraction of PASS verdicts (default 0.80)

Usage::

    python -m src.eval_smoke
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.agent_graph import make_graph

EVAL_PATH = Path("rag/eval/eval_set.jsonl")
OUT_PATH = Path("artifacts/eval_smoke_report.json")

MIN_AVG_JUDGE = float(os.getenv("MIN_AVG_JUDGE", "0.75"))
MIN_PASS_RATE = float(os.getenv("MIN_PASS_RATE", "0.80"))
N_EXAMPLES = int(os.getenv("EVAL_N", "10"))
WEB_SEARCH = os.getenv("WEB_SEARCH", "0") == "1"

def main() -> None:
    """Run the smoke evaluation and enforce quality thresholds.

    Loads up to ``N_EXAMPLES`` questions from ``eval_set.jsonl``,
    invokes the research agent for each, aggregates judge scores,
    writes a JSON report to ``artifacts/eval_smoke_report.json``,
    and exits with a non-zero code if the average score or pass
    rate falls below the configured minimums.

    Raises:
        FileNotFoundError: If the eval set file does not exist.
        SystemExit: If quality thresholds are not met (CI gate failure).
    """
    load_dotenv()

    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing {EVAL_PATH}. Create it (10-50 questions).")

    rows = [json.loads(l) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:N_EXAMPLES]

    app = make_graph()

    results = []
    pass_cnt = 0
    score_sum = 0.0

    # Use a fixed session in CI so memory doesn't explode
    session_id = "ci_smoke"

    for r in rows:
        q = r["question"]
        out = app.invoke({
            "question": q,
            "session_id": session_id,
            "max_iters": 5,
            "force_web": WEB_SEARCH,     # ✅ force Tavily/web path in CI
        })

        judge = out.get("judge", {}) or {}
        verdict = (judge.get("verdict") or "").upper()
        score = float(judge.get("score") or 0.0)

        is_pass = (verdict == "PASS") and (score >= 0.75)

        pass_cnt += 1 if is_pass else 0
        score_sum += score

        results.append({
            "question": q,
            "verdict": verdict,
            "score": score,
            "sources_n": len(out.get("citations", []) or []),
        })

    n = max(1, len(results))
    avg_score = score_sum / n
    pass_rate = pass_cnt / n

    report = {
        "n": n,
        "avg_judge_score": round(avg_score, 4),
        "pass_rate": round(pass_rate, 4),
        "thresholds": {
            "min_avg_judge": MIN_AVG_JUDGE,
            "min_pass_rate": MIN_PASS_RATE,
        },
        "results": results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Eval smoke summary:", report["n"], report["avg_judge_score"], report["pass_rate"])

    # Gate
    if avg_score < MIN_AVG_JUDGE or pass_rate < MIN_PASS_RATE:
        raise SystemExit(
            f"Eval gate failed: avg_judge_score={avg_score:.3f} pass_rate={pass_rate:.3f} "
            f"(min_avg={MIN_AVG_JUDGE}, min_pass={MIN_PASS_RATE})"
        )

    print("Eval gate passed.")


if __name__ == "__main__":
    main()