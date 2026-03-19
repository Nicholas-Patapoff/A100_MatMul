#!/usr/bin/env bash
# First-time JarvisLabs instance setup for A100_MatMul
# Usage: ./setup_instance.sh [--gpu GPU_TYPE] [--storage GB]
# Defaults: GPU=A100, storage=50GB

set -euo pipefail

GPU="${GPU:-A100}"
STORAGE="${STORAGE:-100}"
REPO="https://github.com/Nicholas-Patapoff/A100_MatMul.git"
REMOTE_DIR="/home/A100_MatMul"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu) GPU="$2"; shift 2 ;;
    --storage) STORAGE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "==> Creating instance (gpu=$GPU, storage=${STORAGE}GB)..."
CREATE_JSON=$(jl create --gpu "$GPU" --storage "$STORAGE" --yes --json)
echo "$CREATE_JSON"

MACHINE_ID=$(echo "$CREATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('machine_id') or d['id'])")
echo "==> Machine ID: $MACHINE_ID"

echo "==> Cloning repo..."
jl exec "$MACHINE_ID" -- git clone "$REPO" "$REMOTE_DIR"

echo "==> Running make generate..."
RUN_JSON=$(jl run --on "$MACHINE_ID" --json --yes -- sh -lc "cd $REMOTE_DIR && make generate")
RUN_ID=$(echo "$RUN_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
echo "==> Run ID: $RUN_ID"

echo "==> Waiting for make generate to complete..."
while true; do
  sleep 15
  STATUS=$(jl run status "$RUN_ID" --json | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "  status: $STATUS"
  if [[ "$STATUS" == "completed" || "$STATUS" == "failed" || "$STATUS" == "stopped" ]]; then
    break
  fi
done
jl run logs "$RUN_ID"

if [[ "$STATUS" != "completed" ]]; then
  echo "ERROR: make generate failed (status=$STATUS). Check logs above."
  exit 1
fi

echo "==> Running make..."
RUN_JSON=$(jl run --on "$MACHINE_ID" --json --yes -- sh -lc "cd $REMOTE_DIR && make")
RUN_ID=$(echo "$RUN_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
echo "==> Run ID: $RUN_ID"

echo "==> Waiting for make to complete..."
while true; do
  sleep 15
  STATUS=$(jl run status "$RUN_ID" --json | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "  status: $STATUS"
  if [[ "$STATUS" == "completed" || "$STATUS" == "failed" || "$STATUS" == "stopped" ]]; then
    break
  fi
done
jl run logs "$RUN_ID"

if [[ "$STATUS" != "completed" ]]; then
  echo "ERROR: make failed (status=$STATUS). Check logs above."
  exit 1
fi

echo "==> Pausing instance $MACHINE_ID..."
jl pause "$MACHINE_ID" --yes --json

echo ""
echo "==> Done! Instance $MACHINE_ID is paused."
echo "    To resume and run: jl resume $MACHINE_ID --yes --json"
