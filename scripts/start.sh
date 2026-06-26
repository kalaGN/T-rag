#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SETTINGS_PATH="${SETTINGS_PATH:-$ROOT_DIR/config/settings.yaml}"
SOURCES_PATH="${SOURCES_PATH:-$ROOT_DIR/config/sources.yaml}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

MODE="${1:-web}"
shift || true

cd "$ROOT_DIR"

case "$MODE" in
  web)
    PYTHONPATH=src "$PYTHON_BIN" -m fin_rag.cli serve-web --settings "$SETTINGS_PATH" "$@"
    ;;
  api)
    PYTHONPATH=src "$PYTHON_BIN" -m fin_rag.cli serve-api --settings "$SETTINGS_PATH" "$@"
    ;;
  embeddings)
    PYTHONPATH=src "$PYTHON_BIN" -m fin_rag.cli serve-embeddings --settings "$SETTINGS_PATH" "$@"
    ;;
  index)
    PYTHONPATH=src "$PYTHON_BIN" -m fin_rag.cli index --settings "$SETTINGS_PATH" --sources "$SOURCES_PATH" "$@"
    ;;
  query)
    PYTHONPATH=src "$PYTHON_BIN" -m fin_rag.cli query "$@"
    ;;
  ask)
    PYTHONPATH=src "$PYTHON_BIN" -m fin_rag.cli ask "$@"
    ;;
  *)
    cat <<'EOF' >&2
Usage: scripts/start.sh [web|api|embeddings|index|query|ask] [args...]

Examples:
  scripts/start.sh
  scripts/start.sh api --host 0.0.0.0 --port 8010
  scripts/start.sh embeddings
  scripts/start.sh index
  scripts/start.sh ask "账期切换逻辑是什么"
EOF
    exit 1
    ;;
esac
