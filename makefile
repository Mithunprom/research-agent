.PHONY: install ingest eval-smoke eval serve test fmt

install:
	pip install -r requirements.txt

# Build local FAISS index from docs/
ingest:
	python -m src.rag_ingest_local

# Fast eval gate for CI (small sample)
eval-smoke:
	python -m src.eval_smoke

# (Optional) run your full eval if you have it
eval:
	python -m src.rag_eval

serve:
	uvicorn src.rag_api:app --host 0.0.0.0 --port 8005

test:
	python -m pytest -q

fmt:
	python -m pip install -U ruff
	ruff check src --fix || true
	ruff format src || true