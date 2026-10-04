#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SETTINGS_PATH="${SETTINGS_PATH:-$ROOT_DIR/config/settings.yaml}"
if [[ ! -f "$SETTINGS_PATH" ]]; then
  SETTINGS_PATH="$ROOT_DIR/config/settings.example.yaml"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

MODE="${1:-web}"
shift || true

cd "$ROOT_DIR"

case "$MODE" in
  web)
    PYTHONPATH=src "$PYTHON_BIN" -m t_rag.cli serve-web --settings "$SETTINGS_PATH" "$@"
    ;;
  embeddings)
    PYTHONPATH=src "$PYTHON_BIN" -m t_rag.cli serve-embeddings --settings "$SETTINGS_PATH" "$@"
    ;;
  index)
    PYTHONPATH=src "$PYTHON_BIN" -m t_rag.cli index --settings "$SETTINGS_PATH" "$@"
    ;;
  query)
    PYTHONPATH=src "$PYTHON_BIN" -m t_rag.cli query --settings "$SETTINGS_PATH" "$@"
    ;;
  ask)
    PYTHONPATH=src "$PYTHON_BIN" -m t_rag.cli ask --settings "$SETTINGS_PATH" "$@"
    ;;
  *)
    cat <<'EOF' >&2
Usage: scripts/start.sh [web|embeddings|index|query|ask] [args...]

Examples:
  scripts/start.sh
  scripts/start.sh embeddings
  scripts/start.sh index --kb ID
  scripts/start.sh ask --kb ID "问题"
EOF
    exit 1
    ;;
esac
