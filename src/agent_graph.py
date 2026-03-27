"""agent_graph.py – LangGraph-based research agent with planning, retrieval,
generation, and self-evaluation loop.

Builds a stateful graph that answers a research question by first planning
the retrieval strategy (local RAG, web, or hybrid), then retrieving and
reranking sources, generating a cited answer, and self-judging for
groundedness.  If the judge is unsatisfied the loop refines and re-plans.

Graph flow::

    init → plan →[needs clarification?]
               ├─ YES → clarify → persist → END
               └─ NO  → retrieve_local →[route_retrieval]
                            ├─ web_first     → web_search → compose
                            ├─ local_only    → compose
                            └─ hybrid (< 800 chars local) → web_search → compose
                        compose → answer → judge →[loop_or_finish]
                            ├─ PASS (score ≥ 0.75) or max_iters → write_notes → persist → END
                            └─ NEEDS_MORE_RESEARCH              → refine → plan (re-plan)

Each run persists:
    - data/runs/<run_id>/answer.md        final answer
    - data/runs/<run_id>/judge.json       groundedness verdict + score
    - data/runs/<run_id>/citations.json   numbered source references
    - data/sessions/<date>/notes.md       appended research notes
    - data/sessions/<date>/sources.jsonl  appended source records
    - data/sessions/<date>/turns.jsonl    per-turn Q/A log
"""

from __future__ import annotations
from typing import TypedDict, List, Dict, Any
from pathlib import Path
import json, re, uuid

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from src.tools_web import tavily_search, fetch_text
from src.tools_memory import (
    default_session_id,
    load_session,
    append_session,
    load_recent_turns,
    format_turns_for_prompt,
    append_turn,
)
from src.tools_rerank import rerank

INDEX_DIR = Path("data/faiss_index")
RUNS_DIR = Path("data/runs")

ANSWER_SYSTEM = """You are a research assistant.
Use ONLY the provided context from sources. If the answer is not supported, say so.
Write a concise structured answer with bullet points and citations like [1], [2].
At the end, include a short Bibliography mapping [i] -> source.
"""

JUDGE_SYSTEM = """You are a strict groundedness evaluator.
Given QUESTION, ANSWER, and CONTEXT sources, decide if the answer is supported.
Return ONLY JSON:
{{"verdict":"PASS|NEEDS_MORE_RESEARCH","score":0-1,"missing_points":[...],"suggested_queries":[...]}}
"""

NOTES_SYSTEM = """You maintain running research notes.
Summarize the key takeaways as bullets. Only include claims supported by the sources.
Keep it short (max 12 bullets)."""

PLANNER_SYSTEM = """You are a research planner.
Given the user's question and the conversation context, decide:
- needs_clarification: true/false
- clarifying_question: if needs_clarification, ask ONE question
- retrieval_mode: one of ["local_only","hybrid","web_first"]
- local_k: int (e.g., 20)
- web_k: int (e.g., 6)
- suggested_queries: 3-5 web queries (if web needed)
Return ONLY JSON with keys:
needs_clarification, clarifying_question, retrieval_mode, local_k, web_k, suggested_queries.
"""


def load_local_retriever(k: int = 4):
    """Load the FAISS index and return a LangChain retriever.

    Args:
        k: Number of top documents to retrieve per query.

    Returns:
        A LangChain VectorStoreRetriever backed by the on-disk FAISS index.

    Raises:
        RuntimeError: If the FAISS index directory does not exist.
    """
    if not INDEX_DIR.exists():
        raise RuntimeError(
            f"Missing FAISS index at {INDEX_DIR}. "
            "Run: python -m src.rag_ingest_local"
        )
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": k})


def extract_json_block(text: str) -> dict | None:
    """Extract the first JSON object from *text* that may contain surrounding prose.

    Uses a greedy regex to find the outermost ``{ ... }`` block, then
    attempts ``json.loads()``.  Returns ``None`` if no valid JSON is found.

    Useful for parsing LLM responses that wrap JSON in markdown fences or
    add commentary around the structured output.
    """
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class State(TypedDict):
    """Shared state dictionary passed through every node in the graph.

    Groups:
        Identity   – question, session_id, run_id
        Memory     – prior_notes, recent_turns
        Retrieval  – local_chunks (FAISS), web_sources (Tavily+fetch),
                     context (formatted text for LLM), citations (numbered refs)
        Outputs    – answer (LLM response), judge (groundedness verdict),
                     notes_append (summary bullets for session memory)
        Planning   – plan (raw planner JSON), needs_clarification,
                     clarifying_question, retrieval_mode
        Controls   – k_local (FAISS top-K), web_k (Tavily top-K),
                     max_iters (refinement cap), iters (current count)
    """
    question: str
    session_id: str
    run_id: str

    # memory
    prior_notes: str
    recent_turns: str

    # retrieval / sources
    local_chunks: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    context: str
    citations: List[Dict[str, Any]]

    # outputs
    answer: str
    judge: Dict[str, Any]
    notes_append: str

    # planning
    plan: dict
    needs_clarification: bool
    clarifying_question: str
    retrieval_mode: str

    # controls
    k_local: int
    web_k: int
    max_iters: int
    iters: int


def format_context(state: State) -> str:
    """Merge local chunks and web sources into a single numbered context string.

    Combines both source lists, deduplicates by id/url, caps at 10 sources,
    and formats each as ``[i] title location\\nexcerpt``.  The numbered list
    is also stored in ``state["citations"]`` for the bibliography.
    """
    citations = []

    for ch in state["local_chunks"]:
        citations.append(ch)
    for ws in state["web_sources"]:
        citations.append(ws)

    # Dedup by id/url
    seen = set()
    numbered = []
    for c in citations:
        key = c.get("id") or c.get("url") or (c.get("source"), c.get("page"))
        if key in seen:
            continue
        seen.add(key)
        numbered.append(c)

    state["citations"] = numbered[:10]
    lines = []
    for i, c in enumerate(state["citations"], start=1):
        title = c.get("title") or c.get("source") or "source"
        loc = ""
        if c.get("page"):
            loc = f" p{c['page']}"
        if c.get("url"):
            loc = f" {c['url']}"
        excerpt = c.get("text", "")[:900]
        lines.append(f"[{i}] {title}{loc}\n{excerpt}")

    return "\n\n".join(lines)


def make_graph():
    """Build and compile the LangGraph research agent.

    Constructs the full stateful graph with these nodes:

        init            – Set defaults, generate run_id, load prior session memory
        plan            – LLM decides retrieval strategy and whether to clarify
        clarify         – Return a clarifying question instead of answering
        retrieve_local  – Query FAISS index for top-K chunks, rerank to top 5
        route_retrieval – (conditional) Route based on planner's retrieval_mode
        web_search      – Tavily search + fetch_text, rerank to top 5
        compose         – Merge & deduplicate all sources into numbered context
        answer          – LLM generates a cited answer from context + prior notes
        judge           – LLM evaluates groundedness -> PASS or NEEDS_MORE_RESEARCH
        loop_or_finish  – (conditional) PASS & score >= 0.75 or max_iters -> finalize
        refine          – Replace question with judge's suggested_queries[0]
        write_notes     – LLM summarizes key takeaways as bullet points
        persist         – Save artifacts to data/runs/, append session memory

    Returns:
        Compiled LangGraph runnable. Call ``.invoke({"question": "..."})`` to run.
    """
    llm = ChatOpenAI(model="gpt-4.1-mini")

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYSTEM),
        ("human",
         "CONVERSATION SO FAR:\n{recent_turns}\n\n"
         "CURRENT QUESTION: {question}\n\n"
         "PRIOR NOTES:\n{prior_notes}\n\n"
         "CONTEXT SOURCES:\n{context}\n\n"
         "Write the answer continuing the same thread. Use citations like [1], [2].")
    ])

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM),
        ("human",
         "QUESTION:\n{question}\n\n"
         "ANSWER:\n{answer}\n\n"
         "CONTEXT SOURCES:\n{context}\n\n"
         "Return JSON only.")
    ])

    notes_prompt = ChatPromptTemplate.from_messages([
        ("system", NOTES_SYSTEM),
        ("human",
         "QUESTION: {question}\n\n"
         "ANSWER:\n{answer}\n\n"
         "CONTEXT:\n{context}\n\n"
         "Write notes bullets.")
    ])

    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM),
        ("human",
         "CONVERSATION SO FAR:\n{recent_turns}\n\n"
         "USER QUESTION:\n{question}\n\nReturn JSON only.")
    ])

    # ------------------------------------------------------------------
    # Node functions
    # ------------------------------------------------------------------

    def init(state: State) -> State:
        """Initialise state with defaults and load prior session memory.

        Sets session_id (today's date if empty), run_id (random hex),
        retrieval params, and loads prior notes and recent turns from the
        session directory.
        """
        state["session_id"] = state.get("session_id") or default_session_id()
        state["run_id"] = state.get("run_id") or uuid.uuid4().hex[:10]
        state["k_local"] = state.get("k_local") or 4
        state["web_k"] = state.get("web_k") or 4
        state["max_iters"] = state.get("max_iters") or 2
        state["iters"] = 0
        state["local_chunks"] = []
        state["web_sources"] = []
        state["citations"] = []
        state["context"] = ""
        state["answer"] = ""
        state["judge"] = {}
        state["notes_append"] = ""
        notes, _ = load_session(state["session_id"])
        state["prior_notes"] = notes[-4000:] if notes else ""
        turns = load_recent_turns(state["session_id"], n=6)
        state["recent_turns"] = format_turns_for_prompt(turns)
        return state

    def plan(state: State) -> State:
        """Ask the LLM planner to decide retrieval strategy for this question.

        Parses the planner's JSON response to set retrieval_mode
        (local_only / hybrid / web_first), k values, and whether the
        question needs clarification before proceeding.
        """
        msg = planner_prompt.format_messages(
            recent_turns=state.get("recent_turns", ""),
            question=state["question"],
        )
        resp = llm.invoke(msg)
        data = extract_json_block(resp.content) or {}

        state["plan"] = data
        state["needs_clarification"] = bool(data.get("needs_clarification", False))
        state["clarifying_question"] = (data.get("clarifying_question") or "").strip()
        state["retrieval_mode"] = (data.get("retrieval_mode") or "hybrid").strip()

        # allow planner to override k values
        state["k_local"] = int(data.get("local_k", state.get("k_local", 20)) or 20)
        state["web_k"] = int(data.get("web_k", state.get("web_k", 6)) or 6)

        return state

    def clarify(state: State) -> State:
        """Return a clarifying question instead of a research answer.

        Uses the planner's ``clarifying_question`` if available, otherwise
        falls back to a generic prompt.  Sets the judge verdict to
        NEEDS_CLARIFICATION so the persist node can log it.
        """
        q = (
            state["clarifying_question"]
            or "Can you clarify what exact outcome you want "
               "(summary, comparison, pros/cons, or step-by-step)?"
        )
        state["answer"] = "CLARIFY: " + q
        state["judge"] = {"verdict": "NEEDS_CLARIFICATION", "score": 0.0}
        return state

    def retrieve_local(state: State) -> State:
        """Query the FAISS index for top-K chunks, then rerank to the top 5.

        Uses the LangChain retriever (all-MiniLM-L6-v2 embeddings) to
        fetch ``k_local`` candidate documents, then applies cross-encoder
        reranking to select the best 5 for context composition.
        """
        retriever = load_local_retriever(k=state["k_local"])
        docs = retriever.invoke(state["question"])

        local = []
        for d in docs:
            local.append({
                "id": f"{Path(d.metadata.get('source', 'unknown')).name}:{d.metadata.get('page')}",
                "title": Path(d.metadata.get("source", "unknown")).name,
                "source": Path(d.metadata.get("source", "unknown")).name,
                "page": d.metadata.get("page"),
                "text": d.page_content,
            })

        local = rerank(state["question"], local, top_n=5)
        state["local_chunks"] = local
        return state

    def route_retrieval(state: State) -> str:
        """Conditional router: pick retrieval path based on the planner's mode.

        - ``web_first``  -> go straight to web_search
        - ``local_only`` -> skip web, go to compose
        - ``hybrid``     -> web_search if local text < 800 chars, else compose
        """
        mode = (state.get("retrieval_mode") or "hybrid").lower()

        if mode == "web_first":
            return "web_search"
        if mode == "local_only":
            return "compose"

        # hybrid: fall back to web if local evidence is thin
        total_chars = sum(len(x.get("text", "")) for x in state["local_chunks"])
        if total_chars < 800:
            return "web_search"
        return "compose"

    def web_search(state: State) -> State:
        """Search the web via Tavily, fetch full page text, and rerank results.

        For each Tavily result URL, ``fetch_text()`` downloads and extracts
        plain text.  Failed fetches are silently skipped.  The surviving
        results are reranked to the top 5.
        """
        results = tavily_search(state["question"], k=state["web_k"])
        web_sources = []
        for r in results:
            url = r.get("url")
            if not url:
                continue
            try:
                text = fetch_text(url)
            except Exception:
                continue
            web_sources.append({
                "url": url,
                "title": r.get("title", ""),
                "text": text,
            })
        web_sources = rerank(state["question"], web_sources, top_n=5)
        state["web_sources"] = web_sources
        return state

    def compose_context(state: State) -> State:
        """Build the numbered context string from all retrieved sources.

        Delegates to ``format_context()`` which merges, deduplicates,
        numbers, and formats local + web sources into a single string
        stored in ``state["context"]``.
        """
        state["context"] = format_context(state)
        return state

    def answer(state: State) -> State:
        """Generate a cited research answer using the LLM.

        Feeds question, prior notes, recent conversation turns, and the
        numbered context into the answer prompt.  The LLM produces a
        structured response with ``[1]``, ``[2]`` citation markers.
        """
        msg = answer_prompt.format_messages(
            question=state["question"],
            prior_notes=state["prior_notes"],
            context=state["context"],
            recent_turns=state.get("recent_turns", ""),
        )
        resp = llm.invoke(msg)
        state["answer"] = resp.content
        return state

    def judge(state: State) -> State:
        """Evaluate groundedness of the answer against the source context.

        The LLM returns a JSON verdict with score, missing points, and
        suggested follow-up queries.  The score is clamped to [0.0, 1.0].
        Falls back to NEEDS_MORE_RESEARCH if JSON parsing fails.
        """
        msg = judge_prompt.format_messages(
            question=state["question"],
            answer=state["answer"],
            context=state["context"],
        )
        resp = llm.invoke(msg)
        data = extract_json_block(resp.content) or {
            "verdict": "NEEDS_MORE_RESEARCH",
            "score": 0.0,
            "missing_points": ["Judge returned invalid JSON"],
            "suggested_queries": [],
        }
        try:
            data["score"] = float(data.get("score", 0.0))
        except Exception:
            data["score"] = 0.0
        data["score"] = max(0.0, min(1.0, data["score"]))
        state["judge"] = data
        return state

    def loop_or_finish(state: State) -> str:
        """Conditional router: decide whether to refine or finalise.

        Increments the iteration counter, then routes to ``"finalize"``
        (write_notes) if the judge passed with score >= 0.75 or
        max_iters is reached; otherwise routes to ``"refine"``.
        """
        state["iters"] += 1
        verdict = (state["judge"].get("verdict") or "").upper()
        score = float(state["judge"].get("score") or 0.0)
        if verdict == "PASS" and score >= 0.75:
            return "finalize"
        if state["iters"] >= state["max_iters"]:
            return "finalize"
        return "refine"

    def refine(state: State) -> State:
        """Replace the question with the judge's first suggested query.

        This feeds a more targeted query back into the plan node for the
        next retrieval iteration.
        """
        qs = state["judge"].get("suggested_queries") or []
        if qs:
            state["question"] = qs[0]
        return state

    def write_notes(state: State) -> State:
        """Summarise the research into bullet-point notes for session memory.

        The LLM produces up to 12 concise takeaway bullets backed by
        the source context.  Stored in ``state["notes_append"]`` for
        later persistence.
        """
        msg = notes_prompt.format_messages(
            question=state["question"],
            answer=state["answer"],
            context=state["context"],
        )
        resp = llm.invoke(msg)
        state["notes_append"] = resp.content
        return state

    def persist(state: State) -> State:
        """Save all run artifacts to disk and append to session memory.

        Writes answer.md, judge.json, and citations.json under
        ``data/runs/<run_id>/``, appends notes and sources to the
        session directory, and logs the turn to turns.jsonl.
        """
        outdir = RUNS_DIR / state["run_id"]
        outdir.mkdir(parents=True, exist_ok=True)

        (outdir / "answer.md").write_text(
            state["answer"], encoding="utf-8"
        )
        (outdir / "judge.json").write_text(
            json.dumps(state["judge"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (outdir / "citations.json").write_text(
            json.dumps(state["citations"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        append_session(state["session_id"], state["notes_append"], state["citations"])
        append_turn(
            state["session_id"],
            question=state["question"],
            answer=state["answer"],
            sources=state["citations"],
            judge=state["judge"],
        )
        return state

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    g = StateGraph(State)
    g.add_node("init", init)
    g.add_node("plan", plan)
    g.add_node("clarify", clarify)
    g.add_node("retrieve_local", retrieve_local)
    g.add_node("web_search", web_search)
    g.add_node("compose", compose_context)
    g.add_node("answer", answer)
    g.add_node("judge", judge)
    g.add_node("refine", refine)
    g.add_node("write_notes", write_notes)
    g.add_node("persist", persist)

    g.set_entry_point("init")
    g.add_edge("init", "plan")

    g.add_conditional_edges(
        "plan",
        lambda s: "clarify" if s.get("needs_clarification") else "retrieve_local",
        {"clarify": "clarify", "retrieve_local": "retrieve_local"},
    )

    g.add_edge("clarify", "persist")
    g.add_edge("persist", END)

    g.add_conditional_edges(
        "retrieve_local",
        route_retrieval,
        {"web_search": "web_search", "compose": "compose"},
    )

    g.add_edge("web_search", "compose")
    g.add_edge("compose", "answer")
    g.add_edge("answer", "judge")
    g.add_conditional_edges(
        "judge",
        loop_or_finish,
        {"refine": "refine", "finalize": "write_notes"},
    )
    g.add_edge("refine", "plan")
    g.add_edge("write_notes", "persist")
    g.add_edge("persist", END)

    return g.compile()