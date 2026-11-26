"""
Flask web server — browser‑based RAG chat UI.

Endpoints
---------
  GET  /           → chat page (upload + query)
  POST /upload     → ingest a PDF
  POST /query      → semantic search, returns JSON
  GET  /status     → collection stats
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.config import CHROMADB_DIR, RAW_DATA_DIR
from src.pipeline import ingest, retrieve
from src.vector_store import collection_count

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    level=logging.INFO,
)


@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Accept a PDF, save it, and run the ingestion pipeline."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted."}), 400

    strategy = request.form.get("strategy", "tokens")

    # Save to data/raw/
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RAW_DATA_DIR / file.filename
    file.save(str(save_path))

    try:
        n_chunks = ingest(save_path, strategy=strategy)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "message": f"Ingested {n_chunks} chunks from '{file.filename}'.",
        "chunks": n_chunks,
        "total": collection_count(),
    })


@app.route("/query", methods=["POST"])
def query():
    """Run a semantic search and return results as JSON."""
    data = request.get_json(silent=True) or {}
    q = data.get("query", "").strip()
    top_k = data.get("top_k", 5)

    if not q:
        return jsonify({"error": "Empty query."}), 400

    if collection_count() == 0:
        return jsonify({"error": "No documents ingested yet. Upload a PDF first."}), 400

    hits = retrieve(q, top_k=int(top_k))
    return jsonify({"query": q, "results": hits})


@app.route("/status")
def status():
    """Return current collection stats."""
    return jsonify({"total_chunks": collection_count()})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
