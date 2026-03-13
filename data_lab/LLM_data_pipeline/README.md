# Lab 1 — LLM Data Pipeline

---

## What's Different From the Original Lab

| Component | Original Lab | This Version |
|---|---|---|
| **Data Source** | Wikipedia / single API | **arXiv papers + HackerNews** (dual-source, no API key) |
| **Embeddings** | TF-IDF / basic | **Sentence Transformers** (`all-MiniLM-L6-v2`) |
| **Vector Store** | FAISS | **ChromaDB** (persistent, metadata-rich) |
| **Topic Analysis** | None | **BERTopic** (neural topic modeling) |
| **Dashboard** | None | **Plotly** interactive quality dashboard |
| **Semantic Search** | None | Live cosine-similarity search demo |

---

## Pipeline Architecture

```
[arXiv API] ──┐
              ├──► [Preprocess & Clean] ──► [Sentence Embeddings]
[HackerNews]──┘         │                         │
                   [Quality Score]           [ChromaDB Store]
                        │                         │
                  [BERTopic Model]         [Semantic Search]
                        │
               [Plotly Dashboard]
```

---

## Setup

> **Note:** To run this lab in Google Colab, upload the `Lab1_Enhanced.ipynb` notebook directly to [Google Colab](https://colab.research.google.com/).

```bash
pip install arxiv requests sentence-transformers chromadb \
    bertopic umap-learn hdbscan plotly pandas numpy \
    scikit-learn nltk tqdm numba
```

Then open and run `Lab1_Enhanced.ipynb` top to bottom. No API keys required.

---

## Notebook Structure

| Step | Cell | Description |
|------|------|-------------|
| 0 | Install & Import | Install all dependencies, configure logging |
| 1 | Schema & Config | `Article` dataclass, pipeline config dict |
| 2a | arXiv Collection | Fetch ML/AI papers via the free arXiv API |
| 2b | HackerNews Collection | Fetch tech stories via Algolia HN API |
| 3 | Preprocessing | Clean text, filter short docs, chunk with overlap, compute quality score |
| 4 | Sentence Embeddings | Batch-encode all chunks with `all-MiniLM-L6-v2` |
| 5 | BERTopic | Discover topics from article-level embeddings (no labels needed) |
| 6 | ChromaDB Store | Persist all chunk embeddings + metadata to disk |
| 7 | Semantic Search | Query the vector store with optional `source` filter |
| 8 | Dashboard | Six Plotly charts covering quality, topics, and volume |
| 9 | Pipeline Summary | Audit log saved to `pipeline_summary.json` |

---

## Key Design Decisions

**Dual-source collection** — arXiv gives clean, structured academic text with rich metadata (authors, categories, dates). HackerNews adds practitioner signals: community scores, discussions, and links to tools that academic papers may miss. Together they produce a more balanced corpus.

**Sentence Transformers over TF-IDF** — Dense 384-dim vectors capture semantic meaning rather than keyword frequency. Two documents can discuss the same concept using different words; cosine similarity on dense embeddings handles this where TF-IDF fails.

**ChromaDB over FAISS** — FAISS is a pure vector index; ChromaDB stores metadata alongside embeddings and supports filtered queries (e.g., return only arXiv results, or only docs with `quality_score > 0.6`) without maintaining a separate metadata store.

**BERTopic** — Uses the pre-computed sentence embeddings (no extra pass needed), dimensionality-reduces via UMAP, clusters with HDBSCAN, and labels topics with class-based TF-IDF. Requires no labelled data and handles outliers gracefully via topic `-1`.

**Quality scoring** — A computed `quality_score (0–1)` per article combines word count, source credibility, and HN community score. This becomes a first-class metadata field in ChromaDB, usable downstream for RAG retrieval filtering or fine-tuning data curation.

---
