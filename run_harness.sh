#!/usr/bin/env bash
# Resume the A100-MatMul instance, run harness, save results to git, then pause.
# Usage: ./run_harness.sh [--profile SIZE]
#   --profile SIZE  Run ncu on just the SIZExSIZExSIZE test (e.g. --profile 4096)

set -euo pipefail

INSTANCE_NAME="A100-MatMul"
REMOTE_DIR="/home/ubuntu/A100_MatMul"
RESULTS_DIR="results"
PROFILES_DIR="profiles"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
RESULT_FILE="$RESULTS_DIR/${TIMESTAMP}.txt"
PROFILE_SIZE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --profile) PROFILE_SIZE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

wait_for_run() {
  local run_id="$1"
  while true; do
    sleep 10
    STATUS=$(jl run status "$run_id" --json | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])")
    echo "  status: $STATUS"
    if [[ "$STATUS" == "succeeded" || "$STATUS" == "failed" || "$STATUS" == "stopped" ]]; then
      break
    fi
  done
}

start_run() {
  local label="$1"; shift
  echo "==> $label" >&2
  local out
  out=$(jl run --on "$MACHINE_ID" --json --yes -- "$@" 2>&1) || {
    echo "ERROR: jl run failed for '$label':" >&2
    echo "$out" >&2
    jl pause "$MACHINE_ID" --yes --json >&2
    exit 1
  }
  echo "$out" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'error' in d:
    import sys; print('ERROR: ' + d['error'], file=sys.stderr); exit(1)
print(d['run_id'])
"
}

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
PUBLIC_IP=$(echo "$RESUME_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['public_ip'])")
SSH_USER=$(echo "$RESUME_JSON" | python3 -c "import sys,json,re; cmd=json.load(sys.stdin).get('ssh_command','root@x'); m=re.search(r'(\w+)@', cmd); print(m.group(1) if m else 'root')")
echo "==> Running machine ID: $MACHINE_ID ($SSH_USER@$PUBLIC_IP)"

echo "==> Waiting for SSH to become available..."
while ! jl exec "$MACHINE_ID" -- true 2>/dev/null; do
  echo "  SSH not ready, retrying in 10s..."
  sleep 10
done

RUN_ID=$(start_run "Pulling latest changes and rebuilding..." sh -lc "cd $REMOTE_DIR && git pull && make")
echo "==> Run ID: $RUN_ID"
echo "==> Waiting for build to complete..."
wait_for_run "$RUN_ID"
jl run logs "$RUN_ID"

if [[ "$STATUS" != "succeeded" ]]; then
  echo "ERROR: build failed (status=$STATUS). Check logs above."
  jl pause "$MACHINE_ID" --yes --json
  exit 1
fi

RUN_ID=$(start_run "Running harness..." sh -lc "cd $REMOTE_DIR && ./harness")
echo "==> Run ID: $RUN_ID"
echo "==> Waiting for harness to complete..."
wait_for_run "$RUN_ID"

echo "==> Fetching logs..."
mkdir -p "$RESULTS_DIR"
jl run logs "$RUN_ID" | tee "$RESULT_FILE"

if [[ "$STATUS" != "succeeded" ]]; then
  echo "ERROR: harness failed (status=$STATUS)."
  jl pause "$MACHINE_ID" --yes --json
  exit 1
fi

if [[ -n "$PROFILE_SIZE" ]]; then
  PROFILE_NAME="${TIMESTAMP}_${PROFILE_SIZE}"
  REMOTE_PROFILE="/tmp/${PROFILE_NAME}"
  DATA_PATH="${REMOTE_DIR}/data/${PROFILE_SIZE}x${PROFILE_SIZE}x${PROFILE_SIZE}"

  RUN_ID=$(start_run "Profiling ${PROFILE_SIZE}x${PROFILE_SIZE}x${PROFILE_SIZE} with ncu..." sh -lc \
    "mkdir -p /tmp/profile_data && ln -sf ${DATA_PATH} /tmp/profile_data/ && ncu --set full -o ${REMOTE_PROFILE} ${REMOTE_DIR}/harness /tmp/profile_data")
  echo "==> Run ID: $RUN_ID"
  echo "==> Waiting for ncu to complete..."
  wait_for_run "$RUN_ID"
  jl run logs "$RUN_ID"

  if [[ "$STATUS" != "succeeded" ]]; then
    echo "ERROR: ncu failed (status=$STATUS)."
    jl pause "$MACHINE_ID" --yes --json
    exit 1
  fi

  echo "==> Downloading profile..."
  mkdir -p "$PROFILES_DIR"
  scp -i ~/.ssh/Jarvis -o StrictHostKeyChecking=no \
    ${SSH_USER}@${PUBLIC_IP}:${REMOTE_PROFILE}.ncu-rep \
    ${PROFILES_DIR}/${PROFILE_NAME}.ncu-rep
  echo "==> Profile saved to ${PROFILES_DIR}/${PROFILE_NAME}.ncu-rep"
fi

echo "==> Pausing instance $MACHINE_ID..."
jl pause "$MACHINE_ID" --yes --json

echo "==> Committing results to git..."
git add "$RESULT_FILE"
git commit -m "Add harness results $TIMESTAMP"
git push

echo ""
echo "==> Done! Results saved to $RESULT_FILE"
