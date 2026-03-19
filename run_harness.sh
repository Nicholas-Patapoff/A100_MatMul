#!/usr/bin/env bash
# Resume the A100-MatMul instance, run harness, save results to git, then pause.
# Usage: ./run_harness.sh

set -euo pipefail

INSTANCE_NAME="A100-MatMul"
REMOTE_DIR="/home/A100_MatMul"
RESULTS_DIR="results"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
RESULT_FILE="$RESULTS_DIR/${TIMESTAMP}.txt"

echo "==> Looking up instance '$INSTANCE_NAME'..."
MACHINE_ID=$(jl list --json | python3 -c "
import sys, json
instances = json.load(sys.stdin)
match = [i for i in instances if i.get('name') == '$INSTANCE_NAME']
if not match:
    raise SystemExit('ERROR: No instance named $INSTANCE_NAME found. Run ./setup_instance.sh first.')
print(match[0]['machine_id'])
")
echo "==> Machine ID: $MACHINE_ID"

echo "==> Resuming instance..."
RESUME_JSON=$(jl resume "$MACHINE_ID" --yes --json)
echo "$RESUME_JSON"
MACHINE_ID=$(echo "$RESUME_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('machine_id') or d['id'])")
echo "==> Running machine ID: $MACHINE_ID"

echo "==> Running harness..."
RUN_JSON=$(jl run --on "$MACHINE_ID" --json --yes -- sh -lc "cd $REMOTE_DIR && ./harness")
RUN_ID=$(echo "$RUN_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
echo "==> Run ID: $RUN_ID"

echo "==> Waiting for harness to complete..."
while true; do
  sleep 10
  STATUS=$(jl run status "$RUN_ID" --json | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])")
  echo "  status: $STATUS"
  if [[ "$STATUS" == "succeeded" || "$STATUS" == "failed" || "$STATUS" == "stopped" ]]; then
    break
  fi
done

echo "==> Fetching logs..."
mkdir -p "$RESULTS_DIR"
jl run logs "$RUN_ID" | tee "$RESULT_FILE"

echo "==> Pausing instance $MACHINE_ID..."
jl pause "$MACHINE_ID" --yes --json

if [[ "$STATUS" != "succeeded" ]]; then
  echo "ERROR: harness failed (status=$STATUS). Check logs above."
  exit 1
fi

echo "==> Committing results to git..."
git add "$RESULT_FILE"
git commit -m "Add harness results $TIMESTAMP"
git push

echo ""
echo "==> Done! Results saved to $RESULT_FILE"
