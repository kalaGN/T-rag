from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from fin_rag.config import Settings, load_sources
from fin_rag.embedding_service import serve as serve_embedding_service
from fin_rag.ingestion.documents import collect_loaded_documents, to_llama_documents
from fin_rag.ingestion.llama_pipeline import run_llama_index
from fin_rag.ingestion.pipeline import build_index_plan
from fin_rag.retrieval.fusion import build_query_bundle, build_vector_retriever
from fin_rag.qa.workflow import ask_question


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(prog="fin-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Scan source documents")
    index_parser.add_argument("--settings", type=Path, help="Override settings yaml")
    index_parser.add_argument("--sources", type=Path, help="Override sources yaml")
    index_parser.add_argument("--dry-run", action="store_true", help="Only scan and summarize documents")
    embedding_parser = subparsers.add_parser("serve-embeddings", help="Run the local embedding service")
    embedding_parser.add_argument("--settings", type=Path, help="Override settings yaml")
    query_parser = subparsers.add_parser("query", help="Run a real query against the built index")
    query_parser.add_argument("question", help="User question")
    query_parser.add_argument("--settings", type=Path, help="Override settings yaml")
    query_parser.add_argument("--domain", action="append", help="Optional domain filter, repeatable")
    ask_parser = subparsers.add_parser("ask", help="Run citation-based QA against the built index")
    ask_parser.add_argument("question", help="User question")
    ask_parser.add_argument("--settings", type=Path, help="Override settings yaml")
    ask_parser.add_argument("--domain", action="append", help="Optional domain filter, repeatable")

    args = parser.parse_args()

    if args.command == "index":
        settings = Settings.from_yaml(args.settings)
        sources = load_sources(args.sources)
        plan = build_index_plan(settings=settings, sources=sources)
        summary = plan.scan()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.dry_run:
            return

        try:
            collected = collect_loaded_documents(settings=settings, sources=sources)
            documents = to_llama_documents(collected.documents)
            result = run_llama_index(settings=settings, documents=documents)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            json.dumps(
                {
                    "document_count": result.document_count,
                    "node_count": result.node_count,
                    "collection_name": result.collection_name,
                    "qdrant_url": result.qdrant_url,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "serve-embeddings":
        settings = Settings.from_yaml(args.settings)
        serve_embedding_service(settings)
        return

    if args.command == "query":
        settings = Settings.from_yaml(args.settings)
        try:
            retriever = build_vector_retriever(settings, domains=args.domain)
            nodes = retriever.retrieve(build_query_bundle(args.question))
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            json.dumps(
                [
                    {
                        "score": node.score,
                        "text": node.node.text[:400],
                        "metadata": node.node.metadata,
                    }
                    for node in nodes
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "ask":
        settings = Settings.from_yaml(args.settings)
        try:
            result = ask_question(settings, args.question, domains=args.domain)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            json.dumps(
                {
                    "answer": result.answer,
                    "sources": result.sources,
                    "refused": result.refused,
                    "validation": result.validation,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return


if __name__ == "__main__":
    main()
