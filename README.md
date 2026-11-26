# GENA1RAG — Production RAG Pipeline

A modular, offline-first Retrieval-Augmented Generation (RAG) ingestion and retrieval system in Python.

## Pipeline

```
Upload ➔ Load ➔ Split ➔ Clean ➔ Chunk ➔ Embedding ➔ Vector Store
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest a PDF
python main.py --file data/raw/your_document.pdf

# 3. Query
python main.py --query "What is retrieval-augmented generation?" --top-k 5
```

## Project Structure

```
GENA1RAG/
├── data/
│   ├── raw/              # Drop PDFs here
│   └── chromadb/          # Persistent vector database
├── src/
│   ├── config.py          # Centralised settings
│   ├── loader.py          # PDF → raw text (pymupdf)
│   ├── preprocessor.py    # clean() — de-hyphenation, unicode, whitespace
│   ├── chunker.py         # by_tokens() & by_headings() strategies
│   ├── vector_store.py    # ChromaDB read/write layer
│   └── pipeline.py        # Orchestrator: Load → Clean → Chunk → Store
├── tests/
│   ├── test_preprocessor.py
│   └── test_chunker.py
├── main.py                # CLI interface
├── requirements.txt
└── README.md
```

## CLI Reference

| Flag | Short | Description |
|------|-------|-------------|
| `--file` | `-f` | Path to a PDF to ingest |
| `--query` | `-q` | Natural-language query |
| `--top-k` | `-k` | Number of results (default: 5) |
| `--strategy` | `-s` | Chunking: `tokens` (default) or `headings` |
| `--verbose` | `-v` | Debug logging |

## Key Functions

### `clean(text)` — `src/preprocessor.py`
- **De-hyphenation**: recombines words broken across line-breaks (`multi-\n developer` → `multideveloper`)
- **Unicode cleanup**: strips bullets, ornaments, control characters
- **Whitespace normalisation**: collapses runs, trims edges

### `by_headings(text)` — `src/chunker.py`
- Hierarchical Markdown chunker (recognises `#` through `######`)
- Tracks heading breadcrumbs as metadata (`{"h1": "...", "h2": "..."}`)
- Prunes deeper keys when a higher heading appears

## Configuration

All tuneable parameters are in `src/config.py` and can be overridden via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `RAG_CHUNK_SIZE` | 512 | Tokens per chunk |
| `RAG_CHUNK_OVERLAP` | 64 | Overlap tokens |
| `RAG_COLLECTION` | `rag_docs` | ChromaDB collection name |
| `RAG_TOP_K` | 5 | Default retrieval count |

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- **[PyMuPDF](https://pymupdf.readthedocs.io/)** — fast PDF text extraction
- **[ChromaDB](https://docs.trychroma.com/)** — embedded vector database with built-in Sentence-Transformer embeddings (fully offline)
- **pytest** — testing framework
