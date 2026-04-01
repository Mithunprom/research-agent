"""Smoke test: verify all source modules can be imported without errors."""


def test_import_tools_memory():
    import src.tools_memory


def test_import_tools_web():
    import src.tools_web


def test_import_tools_rerank():
    import src.tools_rerank


def test_import_rag_ingest():
    import src.rag_ingest_local


def test_import_agent_graph():
    import src.agent_graph


def test_import_observability():
    import src.observability
