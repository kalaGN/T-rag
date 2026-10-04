from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from t_rag.config import Settings
from t_rag.knowledge.service import KnowledgeService
from t_rag.knowledge.indexing import rebuild
from t_rag.retrieval.fusion import retrieve
from t_rag.qa.workflow import ask_question


def main() -> None:
    parser = argparse.ArgumentParser(prog="t-rag")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "query", "ask", "serve-web", "serve-embeddings", "kb", "source"):
        command = sub.add_parser(name)
        command.add_argument("--settings", type=Path)
        if name in {"index", "query", "ask", "source"}:
            command.add_argument("--kb", required=True)
        if name in {"query", "ask"}:
            command.add_argument("question")
            command.add_argument("--strategy", choices=["vector", "bm25", "fusion"], default="fusion")
        if name in {"serve-web"}:
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=7860)
        if name == "kb":
            command.add_argument("action", choices=["create", "list", "delete"])
            command.add_argument("value", nargs="?")
            command.add_argument("--confirm", action="store_true")
        if name == "source":
            command.add_argument("action", choices=["add", "upload", "remove", "list"])
            command.add_argument("--directory")
            command.add_argument("--include", action="append")
            command.add_argument("--exclude", action="append")
            command.add_argument("--file", action="append")
            command.add_argument("--id")
    args = parser.parse_args()
    try:
        settings = Settings.from_yaml(args.settings)
        if args.command == "serve-web":
            from t_rag.web.app import serve_web
            serve_web(args.settings, args.host, args.port)
            return
        if args.command == "serve-embeddings":
            from t_rag.models.embedding_service import serve
            serve(settings)
            return
        service = KnowledgeService(settings)
        from t_rag.storage.catalog import application_lock
        with application_lock(service.root):
            execute_command(args, settings, service)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


def execute_command(args, settings, service):
    service.recover()
    if args.command == "kb":
        if args.action == "list":
            result = service.list()
        elif args.action == "create":
            if not args.value:
                raise ValueError("请输入知识库名称。")
            result = service.create(args.value)
        else:
            service.delete(args.value, args.confirm)
            result = {"deleted": args.value}
    elif args.command == "source":
        if args.action == "add":
            if not args.directory:
                raise ValueError("请提供 --directory。")
            result = service.add_directory(args.kb, args.directory, args.include, args.exclude)
        elif args.action == "upload":
            result = service.upload(args.kb, args.file or [])
        elif args.action == "remove":
            service.remove_source(args.kb, args.id)
            result = {"removed": args.id}
        else:
            result = service.load(args.kb)["sources"]
    elif args.command == "index":
        result = rebuild(service, args.kb)
    elif args.command == "query":
        result = [hit.as_dict() for hit in retrieve(settings, args.kb, args.question, args.strategy)]
    else:
        result = asdict(ask_question(settings, args.question, args.kb, strategy=args.strategy))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
