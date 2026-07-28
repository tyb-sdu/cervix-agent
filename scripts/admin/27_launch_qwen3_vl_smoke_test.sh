#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data2/lxj/projects/CervixAgent
ENV_ROOT="$PROJECT_ROOT/.envs/vl_rag"
SCRIPT="$PROJECT_ROOT/scripts/admin/26_smoke_test_qwen3_vl_models.py"
RUN_ROOT="$PROJECT_ROOT/runs/vl_smoke_tests"

test -x "$ENV_ROOT/bin/python"
test -f "$SCRIPT"
mkdir -p "$RUN_ROOT"

if pgrep -af "26_smoke_test_qwen3_vl_models.py" >/dev/null; then
  echo "A Qwen3-VL smoke test is already running:"
  pgrep -af "26_smoke_test_qwen3_vl_models.py"
  exit 0
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
log_path="$RUN_ROOT/qwen3_vl_smoke_${timestamp}.log"
nohup "$ENV_ROOT/bin/python" "$SCRIPT" >"$log_path" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$RUN_ROOT/qwen3_vl_smoke.pid"
echo "PID=$pid"
echo "LOG=$log_path"
