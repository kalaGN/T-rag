#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SETTINGS_PATH="${SETTINGS_PATH:-$ROOT_DIR/config/settings.yaml}"
START_SCRIPT="$ROOT_DIR/scripts/start.sh"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
EMBEDDING_URL="${EMBEDDING_URL:-http://127.0.0.1:8001/v1/models}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-fin-rag-qdrant}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:latest}"
MODE="${1:-web}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$START_SCRIPT" ]]; then
  echo "Start script not found: $START_SCRIPT" >&2
  exit 1
fi

shift || true

EMBEDDING_PID=""
EMBEDDING_LOG=""

cleanup() {
  if [[ -n "$EMBEDDING_PID" ]] && kill -0 "$EMBEDDING_PID" >/dev/null 2>&1; then
    kill "$EMBEDDING_PID" >/dev/null 2>&1 || true
    wait "$EMBEDDING_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

wait_for_http() {
  local url="$1"
  local name="$2"
  local retries="${3:-30}"
  local delay="${4:-1}"
  local i

  for ((i = 1; i <= retries; i++)); do
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done

  echo "$name is not ready: $url" >&2
  return 1
}

ensure_qdrant() {
  if wait_for_http "$QDRANT_URL/healthz" "Qdrant" 1 1; then
    echo "Qdrant is already running at $QDRANT_URL"
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found, and Qdrant is not running." >&2
    exit 1
  fi

  if docker ps -a --format '{{.Names}}' | grep -Fx "$QDRANT_CONTAINER" >/dev/null 2>&1; then
    echo "Starting existing Qdrant container: $QDRANT_CONTAINER"
    docker start "$QDRANT_CONTAINER" >/dev/null
  else
    echo "Creating Qdrant container: $QDRANT_CONTAINER"
    docker run -d \
      --name "$QDRANT_CONTAINER" \
      -p 6333:6333 \
      -v "$QDRANT_CONTAINER-storage:/qdrant/storage" \
      "$QDRANT_IMAGE" >/dev/null
  fi

  wait_for_http "$QDRANT_URL/healthz" "Qdrant"
  echo "Qdrant is ready at $QDRANT_URL"
}

ensure_embeddings() {
  if wait_for_http "$EMBEDDING_URL" "Embedding service" 1 1; then
    echo "Embedding service is already running"
    return 0
  fi

  EMBEDDING_LOG="$(mktemp -t fin-rag-embedding.XXXXXX.log)"
  echo "Starting embedding service in background"
  SETTINGS_PATH="$SETTINGS_PATH" "$START_SCRIPT" embeddings >"$EMBEDDING_LOG" 2>&1 &
  EMBEDDING_PID="$!"

  wait_for_http "$EMBEDDING_URL" "Embedding service"
  echo "Embedding service is ready"
  echo "Embedding log: $EMBEDDING_LOG"
}

run_target() {
  case "$MODE" in
    web|api)
      SETTINGS_PATH="$SETTINGS_PATH" "$START_SCRIPT" "$MODE" "$@"
      ;;
    *)
      cat <<'EOF' >&2
Usage: scripts/dev.sh [web|api] [args...]

Examples:
  scripts/dev.sh
  scripts/dev.sh api --host 0.0.0.0 --port 8010
EOF
      exit 1
      ;;
  esac
}

cd "$ROOT_DIR"
ensure_qdrant
ensure_embeddings
run_target "$@"
