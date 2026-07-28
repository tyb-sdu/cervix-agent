#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data2/lxj/projects/CervixAgent
SCRIPT="$PROJECT_ROOT/scripts/admin/24_create_vl_rag_environment.sh"
RUN_ROOT="$PROJECT_ROOT/runs/vl_rag_setup"

test -f "$SCRIPT"
mkdir -p "$RUN_ROOT"

if pgrep -af "24_create_vl_rag_environment.sh" >/dev/null; then
  echo "A VL-RAG environment setup job is already running:"
  pgrep -af "24_create_vl_rag_environment.sh"
  exit 0
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
log_path="$RUN_ROOT/vl_rag_setup_${timestamp}.log"
nohup bash "$SCRIPT" >"$log_path" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$RUN_ROOT/vl_rag_setup.pid"
echo "PID=$pid"
echo "LOG=$log_path"
