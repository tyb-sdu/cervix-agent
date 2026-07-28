#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data2/lxj/projects/CervixAgent
ENV_ROOT="$PROJECT_ROOT/.envs/vl_rag"
SCRIPT="$PROJECT_ROOT/scripts/admin/29_index_pilot_text_with_qwen3_vl.py"
RUN_ROOT="$PROJECT_ROOT/runs/qdrant_indexing"

test -x "$ENV_ROOT/bin/python"
test -f "$SCRIPT"
mkdir -p "$RUN_ROOT"

if pgrep -af "29_index_pilot_text_with_qwen3_vl.py" >/dev/null; then
  echo "A Qdrant pilot indexing job is already running:"
  pgrep -af "29_index_pilot_text_with_qwen3_vl.py"
  exit 0
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
log_path="$RUN_ROOT/pilot_text_qwen3vl8b_${timestamp}.log"
nohup "$ENV_ROOT/bin/python" "$SCRIPT" --batch-size 4 >"$log_path" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$RUN_ROOT/pilot_text_qwen3vl8b.pid"
echo "PID=$pid"
echo "LOG=$log_path"
