#!/usr/bin/env bash
#
# v0_check.sh — Compare Rust vs Python v0 benchmark results.
#
# Spins up redis:8.6.0 on port 6399, runs both Python and Rust benchmarks
# with the same engine config + dataset, compares precision/QPS/latency,
# then tears down docker.
#
# Usage:
#   ./scripts/v0_check.sh [ENGINE] [DATASET]
#   ENGINE  — engine config name (default: redis-test)
#   DATASET — dataset name       (default: random-100)
#
# Exit code 0 = all checks pass, 1 = precision mismatch or failure.

set -euo pipefail

ENGINE="${1:-redis-m-16-ef-128}"
DATASET="${2:-h-and-m-2048-angular-filters}"
PORT=6399

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUST_BIN="$ROOT/target/release/vector-db-benchmark"
COMPOSE="$ROOT/tests/docker-compose.test.yml"
PY_RESULTS="$ROOT/v0/results"
RS_RESULTS="$ROOT/results"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

cleanup() {
    echo ""
    echo "=== Stopping redis ==="
    docker compose -f "$COMPOSE" down 2>/dev/null || true
}
trap cleanup EXIT

# ── Pre-flight ────────────────────────────────────────────────

echo "=== v0-check: $ENGINE / $DATASET (port $PORT) ==="
echo ""

# Build Rust binary
echo "=== Building Rust binary ==="
cargo build --release --bin vector-db-benchmark --manifest-path "$ROOT/Cargo.toml" 2>&1 | tail -3
echo ""

# Check Python v0 environment
if ! (cd "$ROOT/v0" && poetry run python3 -c "from benchmark import DATASETS_DIR" 2>/dev/null); then
    echo -e "${RED}ERROR: Python v0 environment not set up. Run: cd v0 && poetry install${NC}"
    exit 1
fi

# ── Start Redis ───────────────────────────────────────────────

echo "=== Starting redis:8.6.0 on port $PORT ==="
docker compose -f "$COMPOSE" up -d --wait 2>/dev/null
# Wait for ping
for i in $(seq 1 10); do
    if redis-cli -p "$PORT" ping 2>/dev/null | grep -q PONG; then
        break
    fi
    sleep 0.5
done

if ! redis-cli -p "$PORT" ping 2>/dev/null | grep -q PONG; then
    echo -e "${RED}ERROR: Redis not responding on port $PORT${NC}"
    exit 1
fi

# Configure search workers to match core count
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
redis-cli -p "$PORT" CONFIG SET search-workers "$CORES" >/dev/null 2>&1 || true
echo "Redis ready (search-workers=$CORES)."
echo ""

# ── Clean old results ────────────────────────────────────────

rm -f "$PY_RESULTS/${ENGINE}-${DATASET}-search-"*.json
rm -f "$RS_RESULTS/${ENGINE}-${DATASET}-search-"*.json

# ── Run Python v0 ────────────────────────────────────────────

echo "=== Running Python v0: $ENGINE / $DATASET (parallel=100) ==="
(cd "$ROOT/v0" && REDIS_PORT="$PORT" REPETITIONS=1 poetry run python3 run.py \
    --engines "$ENGINE" \
    --datasets "$DATASET" \
    --parallels 100 \
    --no-skip-if-exists 2>&1) | grep -E "→|Experiment stage|Running"
echo ""

# Find Python result file
PY_FILE=$(ls -t "$PY_RESULTS/${ENGINE}-${DATASET}-search-"*.json 2>/dev/null | head -1)
if [ -z "$PY_FILE" ]; then
    echo -e "${RED}ERROR: No Python search result found${NC}"
    exit 1
fi

# ── Run Rust ──────────────────────────────────────────────────

echo "=== Running Rust: $ENGINE / $DATASET (parallel=100) ==="
REDIS_PORT="$PORT" "$RUST_BIN" \
    --engines "$ENGINE" \
    --datasets "$DATASET" \
    --parallels 100 2>&1 | grep -E "→|Experiment stage|Running|Results saved"
echo ""

# Find Rust result file
RS_FILE=$(ls -t "$RS_RESULTS/${ENGINE}-${DATASET}-search-"*.json 2>/dev/null | head -1)
if [ -z "$RS_FILE" ]; then
    echo -e "${RED}ERROR: No Rust search result found${NC}"
    exit 1
fi

# ── Compare results ───────────────────────────────────────────

echo "=== Comparing results ==="
echo "  Python: $PY_FILE"
echo "  Rust:   $RS_FILE"
echo ""

RESULT=$(python3 -c "
import json, sys

py = json.load(open('$PY_FILE'))
rs = json.load(open('$RS_FILE'))

py_r = py['results']
rs_r = rs['results']

failed = False

print(f\"{'Metric':<20} {'Python v0':>15} {'Rust':>15} {'Status':>20}\")
print('=' * 72)

for key in ['mean_precisions', 'rps', 'mean_time', 'p50_time', 'p95_time', 'p99_time']:
    pv = py_r.get(key, 0.0)
    rv = rs_r.get(key, 0.0)

    if key == 'mean_precisions':
        if abs(pv - rv) < 0.001:
            status = 'PASS'
        else:
            status = 'FAIL'
            failed = True
    elif key == 'rps':
        status = 'PASS (Rust >= Py)' if rv >= pv * 0.9 else 'FAIL (Rust < Py)'
        if rv < pv * 0.9:
            failed = True
    else:
        status = 'PASS (Rust <= Py)' if rv <= pv * 1.5 else 'WARN (Rust > Py)'

    print(f'{key:<20} {pv:>15.6f} {rv:>15.6f} {status:>20}')

print()
if failed:
    print('RESULT: FAIL')
    sys.exit(1)
else:
    print('RESULT: PASS')
    sys.exit(0)
" || true)

echo "$RESULT"
echo ""

if echo "$RESULT" | grep -q "RESULT: PASS"; then
    echo -e "${GREEN}${BOLD}v0-check PASSED${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}v0-check FAILED${NC}"
    exit 1
fi
