"""
rag_ingest_local.py – Local document ingestion for RAG.

Reads PDFs and text files from the docs/ directory, splits them into
overlapping chunks, embeds them with HuggingFace sentence-transformers,
and saves the resulting FAISS vector index to data/faiss_index/.
"""

from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_DIR = Path('docs')
INDEX_DIR = Path('data/faiss_index')


def read_pdf(path: Path) -> List[Document]:
    """Read a single PDF and return one Document per page.

    Extracts text from each page, normalises whitespace, and skips
    near-blank pages (< 30 chars). Each Document carries metadata:
    source (filename), page (1-indexed), and type ("pdf").
    """
    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if len(text) < 30:
            continue
        docs.append(Document(
            page_content=text,
            metadata={"source": path.name, "page": i + 1, "type": "pdf"}
        ))
    return docs

def read_text_file(path: Path) -> List[Document]:
    """Read a plain-text or markdown file and return it as a single Document.

    The entire file content is collapsed into one Document with normalised
    whitespace. Files with fewer than 30 characters are skipped (returns []).
    Metadata includes source (filename), page (None), and type (file extension).
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = " ".join(text.split())
    if len(text) < 30:
        return []
    return [Document(
        page_content=text,
        metadata={"source": path.name, "page": None, "type": path.suffix.lstrip(".")}
    )]

def load_all_docs() -> List[Document]:
    """Recursively scan DOCS_DIR and load all supported files.

    Walks docs/ for .pdf, .txt, and .md files, delegates to read_pdf()
    or read_text_file() accordingly, and returns a flat list of Documents.
    Raises FileNotFoundError if docs/ is missing, or RuntimeError if
    no documents are found.
    """
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Missing {DOCS_DIR}. Create it and add PDFs/notes.")
    docs: List[Document] = []

    for path in DOCS_DIR.rglob("*"):
        if path.is_dir():
            continue
        suf = path.suffix.lower()
        if suf == ".pdf":
            docs.extend(read_pdf(path))
        elif suf in [".txt", ".md"]:
            docs.extend(read_text_file(path))

    if not docs:
        raise RuntimeError("No documents found. Put PDFs/txt/md files in docs/")
    return docs

def main():
    """Entry point: load docs, chunk, embed, and save the FAISS index.

    Flow:
        Raw docs (PDF/txt/md)
            → Split into ~900-char text chunks (120-char overlap)
            → Each chunk is converted to a 384-dim embedding vector
              via sentence-transformers (all-MiniLM-L6-v2)
            → FAISS indexes these vectors for fast similarity search

    At query time the user question is embedded into the same 384-dim
    space and FAISS finds the nearest chunk vectors (semantic search).

    The default FAISS index here is flat (brute-force). For large-scale
    corpora, FAISS supports IVF (Inverted File Index) which partitions
    the vector space into k-means clusters so only nearby regions are
    searched, reducing query time from O(N) to O(N/k).
    """
    docs = load_all_docs()
    print(f"Loaded raw docs/pages: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    print(f"Chunks: {len(chunks)}")

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunks, emb)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))
    print(f"✅ Saved FAISS index -> {INDEX_DIR}")

if __name__ == "__main__":
    main()
