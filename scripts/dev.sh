#!/usr/bin/env bash
# Dev launcher: host llama-server x2 + full backend in podman compose.
#
#   scripts/dev.sh        # start llama models, `podman compose up -d`, migrate, wait for /health
#   scripts/dev.sh down   # stop llama models + `podman compose down`
#
# Llama stays on the host (GPU passthrough in compose is fragile on this box);
# db + azurite + api + worker run in compose with `network_mode: host` so the
# containers reach the models at localhost:8080/8081 from your .env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
LLM_PID_FILE="$LOG_DIR/llm.pid"
EMBED_PID_FILE="$LOG_DIR/embed.pid"

env_get() {
    local key="$1" default="$2" line=""
    if [[ -f .env ]]; then
        line="$(grep -m1 "^${key}=" .env || true)"
    fi
    if [[ -n "$line" ]]; then
        echo "${line#*=}"
    else
        echo "$default"
    fi
}

port_of() {
    # http://localhost:8080/v1 -> 8080
    echo "$1" | sed -e 's#.*://##' -e 's#/.*##' -e 's#.*:##'
}

wait_for_url() {
    local url="$1" label="$2" tries=0
    while ! curl -sf -o /dev/null "$url"; do
        tries=$((tries + 1))
        if [[ "$tries" -ge 60 ]]; then
            echo "error: timed out waiting for $label ($url)" >&2
            echo "tail:" >&2
            tail -n 20 "$LOG_DIR"/*.log >&2 || true
            exit 1
        fi
        sleep 2
    done
    echo "ok: $label ($url)"
}

stop_llama() {
    local pid_file="$1" label="$2"
    if [[ -f "$pid_file" ]]; then
        local pid
        pid="$(cat "$pid_file")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "stopped $label (pid $pid)"
        fi
        rm -f "$pid_file"
    fi
}

if [[ "${1:-up}" == "down" ]]; then
    stop_llama "$LLM_PID_FILE" "llm server"
    stop_llama "$EMBED_PID_FILE" "embedding server"
    podman compose down
    exit 0
fi

command -v llama >/dev/null || { echo "error: 'llama' not on PATH" >&2; exit 1; }
command -v podman >/dev/null || { echo "error: 'podman' not on PATH" >&2; exit 1; }
command -v uv >/dev/null || { echo "error: 'uv' not on PATH" >&2; exit 1; }
[[ -f .env ]] || { echo "error: missing .env (copy from .env.example first)" >&2; exit 1; }

mkdir -p "$LOG_DIR"

LLM_MODEL="$(env_get OPENAI_MODEL "unsloth/Qwen3.5-4B-GGUF:Q6_K")"
EMBED_MODEL="$(env_get EMBEDDING_MODEL "unsloth/embeddinggemma-2-GGUF:Q8_0")"
EMBED_DIM="$(env_get EMBEDDING_DIMENSIONS "768")"
LLM_PORT="$(port_of "$(env_get OPENAI_BASE_URL "http://localhost:8080/v1")")"
EMBED_PORT="$(port_of "$(env_get EMBEDDING_BASE_URL "http://localhost:8081/v1")")"

# Start (or reuse) the LLM server.
if curl -sf -o /dev/null "http://localhost:${LLM_PORT}/v1/models"; then
    echo "reuse: llm server already on :${LLM_PORT}"
else
    # shellcheck disable=SC2086
    nohup llama serve -hf "$LLM_MODEL" --port "$LLM_PORT" --ctx-size 8192 \
        >"$LOG_DIR/llm.log" 2>&1 &
    echo "$!" >"$LLM_PID_FILE"
    echo "started llm: $LLM_MODEL on :${LLM_PORT} (log: logs/llm.log)"
fi

# Start (or reuse) the embedding server. --embedding is required for /v1/embeddings.
if curl -sf -o /dev/null "http://localhost:${EMBED_PORT}/v1/models"; then
    echo "reuse: embedding server already on :${EMBED_PORT}"
else
    # shellcheck disable=SC2086
    nohup llama serve -hf "$EMBED_MODEL" --embedding --pooling last --port "$EMBED_PORT" \
        >"$LOG_DIR/embed.log" 2>&1 &
    echo "$!" >"$EMBED_PID_FILE"
    echo "started embeddings: $EMBED_MODEL on :${EMBED_PORT} (log: logs/embed.log)"
fi

wait_for_url "http://localhost:${LLM_PORT}/v1/models" "llm server"
wait_for_url "http://localhost:${EMBED_PORT}/v1/models" "embedding server"

# Confirm the embedding width matches Settings + vector(768) before migrating.
ACTUAL_DIM="$(python3 - "$EMBED_PORT" <<'EOF'
import json
import sys
import urllib.request

port = sys.argv[1]
body = json.dumps({"input": ["hello world"]}).encode()
req = urllib.request.Request(
    f"http://localhost:{port}/v1/embeddings",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=120) as resp:
    payload = json.load(resp)
print(len(payload["data"][0]["embedding"]))
EOF
)"
if [[ "$ACTUAL_DIM" != "$EMBED_DIM" ]]; then
    echo "error: embedding dim $ACTUAL_DIM != EMBEDDING_DIMENSIONS=$EMBED_DIM" >&2
    echo "Refusing to continue: changing dims needs a vector(n) migration + re-ingest." >&2
    exit 1
fi
echo "ok: embedding dim $ACTUAL_DIM"

podman compose up -d
uv run alembic upgrade head
wait_for_url "http://localhost:8000/health" "api"

echo ""
podman compose ps
echo ""
echo "ready: api http://localhost:8000/health  llm :${LLM_PORT}  embed :${EMBED_PORT}"
echo "stop:  scripts/dev.sh down"
