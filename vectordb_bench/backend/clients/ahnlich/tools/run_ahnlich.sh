#!/usr/bin/env bash
# Run the selected dataset/algorithm matrix against a managed or existing server.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-1369}"
THREADS="${THREADS:-4}"
ALLOCATOR_SIZE="${ALLOCATOR_SIZE:-8589934592}"
MESSAGE_SIZE="${MESSAGE_SIZE:-67108864}"
CONFIG_FILE="${CONFIG_FILE:-$ROOT/vectordb_bench/backend/clients/ahnlich/examples/ahnlich.yaml}"
SERVER_REF="${SERVER_REF:-unknown}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/.data/results/ahnlich-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd -- "$RESULTS_DIR" && pwd)"
if [ -e "$RESULTS_DIR/plan.json" ] || [ -e "$RESULTS_DIR/json" ]; then
    echo "Use a fresh RESULTS_DIR; this directory already contains results" >&2
    exit 1
fi
export RESULTS_DIR HOST PORT MESSAGE_SIZE SERVER_REF
export LOG_FILE="$RESULTS_DIR/vectordbbench.log"
export DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-$ROOT/.data/datasets}"
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$ROOT/.data/ir_datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.data/matplotlib}"
cp "$CONFIG_FILE" "$RESULTS_DIR/config.yaml"
"$PYTHON" -m vectordb_bench.backend.clients.ahnlich.tools.run_ahnlich_matrix \
    --plan-file "$RESULTS_DIR/plan.json" --config-file "$RESULTS_DIR/config.yaml" "$@"
if [ ! -f "$RESULTS_DIR/plan.json" ]; then
    exit 0 # --help does not create a plan.
fi
if "$PYTHON" - "$RESULTS_DIR/plan.json" <<'PY'
import json
import sys
sys.exit(0 if json.load(open(sys.argv[1]))["dry_run"] else 1)
PY
then
    exit 0
fi
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ -n "${AHNLICH_ROOT:-}${SERVER_BIN:-}" ]; then
    if [ "$HOST" != "127.0.0.1" ]; then
        echo "Managed servers require HOST=127.0.0.1; omit AHNLICH_ROOT and SERVER_BIN for remote runs" >&2
        exit 1
    fi
    "$PYTHON" - "$PORT" <<'PY'
import socket
import sys
with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0:
        raise SystemExit("Port is already in use; select another PORT for this isolated run")
PY
    if [ -z "${SERVER_BIN:-}" ]; then
        cargo build --release --manifest-path "$AHNLICH_ROOT/ahnlich/Cargo.toml" \
            --bin ahnlich-db --target-dir "$ROOT/.data/ahnlich-target" \
            2>&1 | tee "$RESULTS_DIR/build.txt"
        SERVER_BIN="$ROOT/.data/ahnlich-target/release/ahnlich-db"
        SERVER_REF="$(git -C "$AHNLICH_ROOT" rev-parse HEAD)"
        if [ -n "$(git -C "$AHNLICH_ROOT" diff HEAD -- ahnlich protos)" ]; then
            SERVER_REF="$SERVER_REF-working-tree"
        fi
    fi
    [ -x "$SERVER_BIN" ] || { echo "SERVER_BIN is not executable: $SERVER_BIN" >&2; exit 1; }
    "$SERVER_BIN" run --port "$PORT" --threadpool-size "$THREADS" \
        --allocator-size "$ALLOCATOR_SIZE" --message-size "$MESSAGE_SIZE" \
        --log-level info > "$RESULTS_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    export SERVER_LOG="$RESULTS_DIR/server.log"
fi
{
    printf 'Server revision: %s; endpoint: %s:%s\n' "$SERVER_REF" "$HOST" "$PORT"
    printf 'VectorDBBench revision: %s\n' "$(git rev-parse HEAD)"
    if [ -n "$(git status --porcelain)" ]; then
        printf 'VectorDBBench checkout contains uncommitted changes\n'
    fi
    if [ -n "$SERVER_PID" ]; then
        printf 'Managed server: threads=%s; allocator bytes=%s; message bytes=%s\n' \
            "$THREADS" "$ALLOCATOR_SIZE" "$MESSAGE_SIZE"
        "$SERVER_BIN" --version
    else
        printf 'Existing server; record its machine resources and DB settings separately\n'
    fi
    uname -a
    "$PYTHON" - <<'PY'
import importlib.metadata
import platform
print(f"Python: {platform.python_version()}")
for package in ("vectordb-bench", "grpcio", "protobuf", "ahnlich-client-py", "grpclib", "betterproto"):
    print(f"{package}: {importlib.metadata.version(package)}")
PY
} > "$RESULTS_DIR/environment.txt"
"$PYTHON" - "$HOST" "$PORT" <<'PY'
import grpc
import sys
with grpc.insecure_channel(f"{sys.argv[1]}:{sys.argv[2]}") as channel:
    grpc.channel_ready_future(channel).result(timeout=30)
PY
"$PYTHON" -m vectordb_bench.backend.clients.ahnlich.tools.run_ahnlich_matrix --run-plan "$RESULTS_DIR/plan.json"
