from pathlib import Path
import subprocess
import sys
def test_imports():
    import src.agent_graph
    import src.rag_ingest_local
    # builds FAISS index from docs/
    subprocess.check_call([sys.executable, "-m", "src.rag_ingest_local"])
    assert Path("data/faiss_index").exists()