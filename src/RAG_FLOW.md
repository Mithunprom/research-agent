# How RAG Works: Chunking, Embedding, Indexing & Query Matching

## 1. Document Ingestion

Raw documents (PDF, txt, md) are loaded and split into pages or single-file blocks.

```
docs/
  paper.pdf        → [page1_text, page2_text, ...]
  notes.txt        → [full_text]
  summary.md       → [full_text]
```

## 2. Chunking

Each document is split into overlapping text chunks using `RecursiveCharacterTextSplitter`.

- **Chunk size**: ~900 characters
- **Overlap**: 120 characters (so context isn't lost at boundaries)

```
page1_text (3000 chars)
    → chunk_1 (chars 0–900)
    → chunk_2 (chars 780–1680)      ← 120-char overlap with chunk_1
    → chunk_3 (chars 1560–2460)
    → chunk_4 (chars 2440–3000)
```

Why overlap? Without it, a sentence split across two chunks would lose meaning in both.

## 3. Embedding

Each text chunk is converted into a **384-dimensional dense vector** using a sentence-transformer model (`all-MiniLM-L6-v2`).

```
"The Federal Reserve raised rates by 25bp"  →  [0.023, -0.117, 0.542, ..., 0.081]
                                                         (384 floats)
```

These vectors encode **semantic meaning** — chunks about similar topics produce vectors that are close together in 384-dim space, regardless of the exact words used.

## 4. FAISS Indexing

### Flat Index (default, used in this project)

All chunk vectors are stored as-is. At query time, the query vector is compared against **every** stored vector (brute-force).

- Fast enough for thousands of chunks
- Guarantees exact nearest neighbors
- O(N) per query

### IVF Index (for large-scale corpora)

For millions of chunks, brute-force is too slow. FAISS uses **Inverted File Index (IVF)** to partition the vector space:

```
Build time:
    All chunk vectors
        ↓
    K-means clustering (e.g. k=100)
        ↓
    100 clusters, each with a centroid (384-dim mean vector)
        ↓
    Each chunk vector is assigned to its nearest centroid's cluster

Result:
    Centroid_1  → [chunk_12, chunk_57, chunk_203, ...]
    Centroid_2  → [chunk_3, chunk_88, chunk_441, ...]
    ...
    Centroid_100 → [chunk_7, chunk_150, chunk_999, ...]
```

Centroids are **not** real chunks — they are the geometric center of each cluster. They serve as gatekeepers for fast lookup.

## 5. Query Time

### With Flat Index

```
User query: "What is monetary policy?"
        ↓
Embed query → [0.03, -0.11, 0.54, ...]  (384-dim)
        ↓
Compare against ALL stored chunk vectors (cosine / L2 distance)
        ↓
Return top-K nearest chunks
```

### With IVF Index

```
User query: "What is monetary policy?"
        ↓
Embed query → [0.03, -0.11, 0.54, ...]  (384-dim)
        ↓
Compare query against 100 centroids ONLY          ← cheap (100 comparisons)
        ↓
Select nearest `nprobe` centroids (e.g. top 10)
        ↓
Search ONLY vectors inside those 10 clusters      ← ~10K vectors, not 1M
        ↓
Return top-K nearest chunks
```

**nprobe** controls the accuracy/speed trade-off:
- Higher nprobe → searches more clusters → more accurate, slower
- Lower nprobe → searches fewer clusters → faster, might miss relevant chunks

## 6. LLM Generation

The top-K retrieved chunks are injected as context into the LLM prompt:

```
System: Answer the question using ONLY the context below.

Context:
  [chunk_57]: "The Federal Reserve's monetary policy committee..."
  [chunk_203]: "Interest rate decisions are made by..."
  [chunk_88]: "Quantitative easing involves purchasing..."

User: What is monetary policy?

LLM → generates answer grounded in the retrieved chunks
```

This is the "Retrieval-Augmented Generation" loop:
**Retrieve** relevant chunks via vector similarity → **Augment** the prompt with them → **Generate** an answer with the LLM.

## Summary Table

| Step | Input | Output | Tool |
|------|-------|--------|------|
| Ingest | PDF/txt/md files | Document objects with text + metadata | pypdf, pathlib |
| Chunk | Documents | ~900-char overlapping text chunks | RecursiveCharacterTextSplitter |
| Embed | Text chunks | 384-dim vectors | all-MiniLM-L6-v2 |
| Index | Vectors | FAISS index (flat or IVF) | FAISS |
| Query | User question | Top-K similar chunks | FAISS similarity search |
| Generate | Question + retrieved chunks | Grounded answer | LLM (GPT, etc.) |
