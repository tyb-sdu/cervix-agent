from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_workflow, workflow_checksum


PROJECT_DIRS = (
    "data/raw",
    "data/processed",
    "data/references",
    "structures/HPV_E6",
    "structures/IDO1",
    "runs",
    "artifacts/docking",
    "artifacts/md",
    "artifacts/experiments",
    "artifacts/cc_irs",
    "logs",
    "reports",
)


class ProjectError(RuntimeError):
    pass


def init_project(path: Path, name: str, force: bool = False) -> dict[str, Any]:
    root = path.expanduser().resolve()
    metadata_dir = root / ".cervixagent"
    state_path = metadata_dir / "project.json"

    if state_path.exists() and not force:
        raise ProjectError(f"项目已经初始化：{root}")

    root.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for relative in PROJECT_DIRS:
        (root / relative).mkdir(parents=True, exist_ok=True)

    workflow = load_workflow()
    checksum = workflow_checksum(workflow)
    lock_path = metadata_dir / "workflow.lock.json"
    lock_path.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    state = {
        "schema_version": 1,
        "project_name": name,
        "project_root": str(root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "initialized",
        "current_phase": "phase_1",
        "current_step": "P1-01",
        "workflow_locked": True,
        "workflow_sha256": checksum,
        "server_access": "unavailable",
        "commercial_schrodinger_license": False,
        "completed_steps": [],
    }
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return state


def load_project(path: Path) -> dict[str, Any]:
    root = path.expanduser().resolve()
    state_path = root / ".cervixagent" / "project.json"
    if not state_path.exists():
        raise ProjectError(f"目录尚未初始化为 CervixAgent 项目：{root}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def _write_project_state(root: Path, state: dict[str, Any]) -> None:
    state_path = root / ".cervixagent" / "project.json"
    temporary = state_path.with_name(state_path.name + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(state_path)


def complete_current_step(path: Path, step_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Advance exactly one locked workflow step and attach its audit evidence."""
    root = path.expanduser().resolve()
    state = load_project(root)
    workflow = load_workflow()
    locked_path = root / ".cervixagent" / "workflow.lock.json"
    if not locked_path.exists():
        raise ProjectError("缺少 workflow.lock.json，不能推进工作流")
    locked_workflow = json.loads(locked_path.read_text(encoding="utf-8"))
    expected_checksum = workflow_checksum(workflow)
    if workflow_checksum(locked_workflow) != expected_checksum:
        raise ProjectError("本地工作流锁与软件内置方案不一致，拒绝推进")
    if state.get("workflow_sha256") != expected_checksum:
        raise ProjectError("项目记录的工作流哈希不一致，拒绝推进")
    if state.get("current_step") != step_id:
        raise ProjectError(
            f"当前步骤是 {state.get('current_step')}，不能把 {step_id} 标记为完成"
        )

    ordered_steps: list[tuple[str, str, bool]] = []
    for phase in workflow["phases"]:
        for step in phase["steps"]:
            ordered_steps.append((phase["id"], step["id"], bool(step.get("human_gate"))))
    current_index = next(
        (index for index, (_, candidate, _) in enumerate(ordered_steps) if candidate == step_id),
        None,
    )
    if current_index is None:
        raise ProjectError(f"锁定工作流中不存在步骤 {step_id}")
    if ordered_steps[current_index][2]:
        raise ProjectError(f"{step_id} 是人工审批步骤，不能由智能体自动完成")

    completed = list(state.get("completed_steps", []))
    completed.append(
        {
            "step_id": step_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "evidence": evidence,
        }
    )
    state["completed_steps"] = completed
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    if current_index + 1 < len(ordered_steps):
        next_phase, next_step, _ = ordered_steps[current_index + 1]
        state["current_phase"] = next_phase
        state["current_step"] = next_step
        state["status"] = "active"
    else:
        state["status"] = "completed"
    _write_project_state(root, state)
    return state
