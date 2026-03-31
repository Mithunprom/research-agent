"""tools_rerank.py – Cross-encoder reranking for retrieved source candidates.

After the initial retrieval step (FAISS or Tavily) returns a broad set of
candidate chunks, this module rescores each candidate against the query
using a cross-encoder model and keeps only the top-N most relevant results.

Why rerank?
    Bi-encoder retrieval (FAISS) is fast but approximate — it encodes query
    and document independently, so subtle relevance signals can be missed.
    A cross-encoder sees (query, document) together and produces a more
    accurate relevance score, at the cost of being slower. Retrieving a
    broad set first (e.g. k=20) then reranking to top 5 gives the best
    of both worlds.

Model:
    ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — small, fast, and trained
    on the MS MARCO passage ranking dataset. Loaded lazily as a singleton
    to avoid reloading on every call.
"""

from sentence_transformers import CrossEncoder

_MODEL = None


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """Return the singleton cross-encoder model, loading it on first call.

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.

    Returns:
        A ``CrossEncoder`` instance ready to score (query, passage) pairs.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder(model_name)
    return _MODEL


def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
    text_key: str = "text",
) -> list[dict]:
    """Rerank candidate dicts by cross-encoder relevance to the query.

    Each candidate's text (truncated to 1200 chars) is paired with the
    query and scored by the cross-encoder. A ``rerank_score`` field is
    added to every candidate, and the list is sorted descending by score.

    Args:
        query: The search query to rank against.
        candidates: List of dicts, each containing a text field.
        top_n: Number of top-scoring candidates to return.
        text_key: Key in each candidate dict that holds the text content.

    Returns:
        The top *top_n* candidates sorted by ``rerank_score`` (highest first).
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, c.get(text_key, "")[:1200]) for c in candidates]
    scores = model.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)
    return candidates[:top_n]