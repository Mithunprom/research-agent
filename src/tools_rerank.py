from sentence_transformers import CrossEncoder

# Small + fast reranker (good default)
_MODEL = None

def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder(model_name)
    return _MODEL

def rerank(query: str, candidates: list[dict], top_n: int = 5, text_key: str = "text"):
    """
    candidates: list of dicts with a 'text' field
    returns: top_n candidates with added 'rerank_score'
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