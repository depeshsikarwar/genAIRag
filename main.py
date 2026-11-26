#!/usr/bin/env python3
"""
CLI entry‑point for the RAG pipeline.

Usage examples
--------------
  # Ingest a PDF
  python main.py --file data/raw/paper.pdf

  # Ingest with heading‑based chunking
  python main.py --file data/raw/paper.pdf --strategy headings

  # One‑shot query
  python main.py --query "What is retrieval‑augmented generation?" --top-k 3

  # Interactive chat mode
  python main.py --chat
  python main.py --file data/raw/paper.pdf --chat   # ingest then chat
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from src.config import DEFAULT_TOP_K
from src.pipeline import ingest, retrieve
from src.vector_store import collection_count

# ── ANSI colours (gracefully ignored on dumb terminals) ──────────────
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _print_hits(hits: list[dict], query: str, top_k: int) -> None:
    """Pretty‑print retrieval results."""
    print(f"\n{GREEN}🔍  Top {len(hits)} results for:{RESET} \"{query}\"\n")
    print(f"{DIM}{'─' * 72}{RESET}")
    for rank, hit in enumerate(hits, 1):
        meta_str = ", ".join(f"{k}={v}" for k, v in hit["metadata"].items())
        dist = hit["distance"]
        doc = hit["document"]
        preview = f"{doc[:200]}…" if len(doc) > 200 else doc
        print(f"  {BOLD}[{rank}]{RESET}  distance={dist:.4f}  |  {DIM}{meta_str}{RESET}")
        print(f"       {preview}\n")
    print(f"{DIM}{'─' * 72}{RESET}")


def _chat_loop(top_k: int) -> None:
    """Interactive chat‑style REPL for querying the vector store."""
    print(f"\n{CYAN}{'═' * 56}{RESET}")
    print(f"{CYAN}   📚  RAG Chat  —  ask anything about your documents{RESET}")
    print(f"{CYAN}{'═' * 56}{RESET}")
    print(f"{DIM}   Type your question and press Enter.")
    print(f"   Commands:  /top-k N  ·  /count  ·  /help  ·  /quit{RESET}\n")

    current_top_k = top_k

    while True:
        try:
            user_input = input(f"{BOLD}You ▸ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}👋  Goodbye!{RESET}\n")
            break

        if not user_input:
            continue

        # ── Slash commands ───────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            if cmd[0] in ("/quit", "/exit", "/q"):
                print(f"\n{DIM}👋  Goodbye!{RESET}\n")
                break
            elif cmd[0] == "/top-k" and len(cmd) == 2 and cmd[1].isdigit():
                current_top_k = int(cmd[1])
                print(f"{DIM}   top-k set to {current_top_k}{RESET}\n")
            elif cmd[0] == "/count":
                print(f"{DIM}   📦  {collection_count()} chunks in the vector store.{RESET}\n")
            elif cmd[0] == "/help":
                print(f"{DIM}   /top-k N   — change number of results (currently {current_top_k})")
                print(f"   /count     — show stored chunk count")
                print(f"   /quit      — exit chat{RESET}\n")
            else:
                print(f"{YELLOW}   Unknown command. Type /help{RESET}\n")
            continue

        # ── Query ────────────────────────────────────────────────
        if collection_count() == 0:
            print(f"\n{YELLOW}⚠️  Vector store is empty. Ingest a document first.{RESET}\n")
            continue

        hits = retrieve(user_input, top_k=current_top_k)
        _print_hits(hits, user_input, current_top_k)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG ingestion & retrieval CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python main.py --file data/raw/paper.pdf
              python main.py --query "Explain chunking strategies" --top-k 3
              python main.py --chat
              python main.py --file data/raw/paper.pdf --chat
        """),
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a PDF file to ingest.",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Natural‑language query (one‑shot mode).",
    )
    parser.add_argument(
        "--chat", "-c",
        action="store_true",
        help="Start interactive chat mode after optional ingestion.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=["tokens", "headings"],
        default="tokens",
        help="Chunking strategy (default: tokens).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        level=level,
    )

    if not args.file and not args.query and not args.chat:
        parser.print_help()
        return 1

    # ── Ingest ───────────────────────────────────────────────────────
    if args.file:
        try:
            n = ingest(args.file, strategy=args.strategy)
            print(f"\n{GREEN}✅  Ingested {n} chunks from '{args.file}'.{RESET}")
            print(f"   Collection now holds {collection_count()} documents.\n")
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"\n❌  Ingestion failed: {exc}\n", file=sys.stderr)
            return 1

    # ── One‑shot query ───────────────────────────────────────────────
    if args.query:
        if collection_count() == 0:
            print(f"\n{YELLOW}⚠️  Vector store is empty. Ingest a document first.{RESET}\n")
            return 1
        hits = retrieve(args.query, top_k=args.top_k)
        _print_hits(hits, args.query, args.top_k)

    # ── Interactive chat ─────────────────────────────────────────────
    if args.chat:
        _chat_loop(top_k=args.top_k)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
