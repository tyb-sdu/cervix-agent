from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_public_sources, load_workflow, workflow_checksum
from .doctor import checks_as_dicts, run_checks
from .project import ProjectError, complete_current_step, load_project


class AuditError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    try:
        with path.open(mode, encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise AuditError(f"审计文件已存在，拒绝覆盖：{path}") from exc


def _command_output(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_snapshot() -> dict[str, Any]:
    return {
        "captured_at": _utc_now(),
        "operating_system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {
            "cervixagent": _package_version("cervixagent"),
            "rdkit": _package_version("rdkit"),
            "numpy": _package_version("numpy"),
        },
        "commands": {
            "git": _command_output(["git", "--version"]),
            "nvidia_smi": _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        },
    }


def seal_directory(directory: Path) -> dict[str, Any]:
    """Create a local tamper-evident seal; this is not immutable/WORM storage."""
    directory = directory.resolve()
    seal_path = directory / "seal.json"
    if seal_path.exists():
        raise AuditError(f"目录已经封存，拒绝重新封存：{directory}")
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative == "seal.json":
            continue
        files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    canonical = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    seal = {
        "schema_version": 1,
        "sealed_at": _utc_now(),
        "assurance": "local_tamper_evident_sha256_not_immutable_storage",
        "files": files,
        "aggregate_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    _write_json(seal_path, seal, exclusive=True)
    return seal


def verify_sealed_directory(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    seal_path = directory / "seal.json"
    if not seal_path.exists():
        return {
            "run_id": directory.name,
            "valid": False,
            "checked_at": _utc_now(),
            "errors": ["缺少 seal.json"],
            "files": {},
        }
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected = seal.get("files", {})
    actual_paths = {
        path.relative_to(directory).as_posix(): path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "seal.json"
    }
    errors: list[str] = []
    file_results: dict[str, dict[str, Any]] = {}
    for relative, expected_record in expected.items():
        path = actual_paths.get(relative)
        if path is None:
            errors.append(f"缺少文件：{relative}")
            file_results[relative] = {"valid": False, "reason": "missing"}
            continue
        actual_hash = _sha256_file(path)
        actual_bytes = path.stat().st_size
        valid = (
            actual_hash == expected_record.get("sha256")
            and actual_bytes == expected_record.get("bytes")
        )
        file_results[relative] = {
            "valid": valid,
            "bytes": actual_bytes,
            "sha256": actual_hash,
        }
        if not valid:
            errors.append(f"哈希或大小不匹配：{relative}")
    unexpected = sorted(set(actual_paths) - set(expected))
    for relative in unexpected:
        errors.append(f"封存后出现额外文件：{relative}")
    canonical = json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    aggregate = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if aggregate != seal.get("aggregate_sha256"):
        errors.append("seal.json 中的聚合哈希不匹配")
    return {
        "run_id": directory.name,
        "valid": not errors,
        "checked_at": _utc_now(),
        "errors": errors,
        "files": file_results,
        "aggregate_sha256": aggregate,
    }


def _source_status(root: Path) -> dict[str, Any]:
    registry = load_public_sources()["sources"]
    status: dict[str, Any] = {
        "ECNPDB": {
            "status": "unresolved",
            "reason": "尚未确认完整名称、官方地址、许可证或项目自建定义",
            "substitution_allowed": False,
        }
    }
    for source_name, key in (("COCONUT", "coconut_drug_discovery"), ("LOTUS", "lotus_smiles")):
        source = registry[key]
        path = root / source["relative_path"]
        status[source_name] = {
            "status": "engineering_snapshot_present" if path.exists() else "missing",
            "relative_path": source["relative_path"],
            "license": source["license"],
            "formal_p1_02_complete": False,
        }
    return status


def create_p1_01_baseline(project_path: Path, label: str = "p1-01-baseline") -> dict[str, Any]:
    root = project_path.expanduser().resolve()
    state = load_project(root)
    if state.get("current_step") != "P1-01":
        raise AuditError(f"当前步骤是 {state.get('current_step')}，不能重复完成 P1-01")

    checks = checks_as_dicts(run_checks(root))
    required = {"Python", "Disk", "Git", "RDKit"}
    failures = [
        item
        for item in checks
        if item["name"] in required and item["status"] != "available"
    ]
    ready = not failures
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip(".-")[:64]
    if not safe_label:
        raise AuditError("审计标签必须至少包含一个字母或数字")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}_P1-01_{safe_label}"
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    workflow = load_workflow()
    run_record = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": _utc_now(),
        "step_id": "P1-01",
        "purpose": "初始化环境与封存基线审计记录",
        "project_name": state["project_name"],
        "project_state_before": state,
        "workflow_sha256": workflow_checksum(workflow),
        "ready_for_p1_02": ready,
        "readiness_failures": failures,
    }
    policy = {
        "workflow_locked": True,
        "protocol_changes_by_agent": False,
        "server_access": state.get("server_access", "unavailable"),
        "commercial_schrodinger_license": state.get(
            "commercial_schrodinger_license", False
        ),
        "unlicensed_commercial_software_allowed": False,
        "p1_04_screening_started": False,
        "standardization_rules": {
            "salt_removal": "unconfirmed_not_applied",
            "tautomer_normalization": "unconfirmed_not_applied",
            "protonation_normalization": "unconfirmed_not_applied",
            "stereochemistry_changes": "not_allowed",
        },
        "audit_assurance": "本地 SHA-256 封存可检测篡改，但不等同于 WORM 或数字签名",
    }
    _write_json(run_dir / "run.json", run_record, exclusive=True)
    _write_json(run_dir / "environment.json", environment_snapshot(), exclusive=True)
    _write_json(run_dir / "checks.json", {"checks": checks}, exclusive=True)
    _write_json(run_dir / "policy.json", policy, exclusive=True)
    _write_json(run_dir / "data_sources.json", _source_status(root), exclusive=True)
    seal = seal_directory(run_dir)
    verification = verify_sealed_directory(run_dir)
    if not verification["valid"]:
        raise AuditError("P1-01 基线封存后验证失败：" + "; ".join(verification["errors"]))

    advanced = False
    state_after = state
    if ready:
        try:
            state_after = complete_current_step(
                root,
                "P1-01",
                {
                    "run_id": run_id,
                    "relative_path": str(run_dir.relative_to(root)),
                    "aggregate_sha256": seal["aggregate_sha256"],
                },
            )
        except ProjectError as exc:
            raise AuditError(str(exc)) from exc
        advanced = True
    return {
        "run_id": run_id,
        "relative_path": str(run_dir.relative_to(root)),
        "ready_for_p1_02": ready,
        "advanced_to_p1_02": advanced,
        "current_step": state_after.get("current_step"),
        "aggregate_sha256": seal["aggregate_sha256"],
        "readiness_failures": failures,
    }


def verify_audit_run(project_path: Path, run_id: str | None = None) -> dict[str, Any]:
    root = project_path.expanduser().resolve()
    load_project(root)
    runs_dir = root / "runs"
    if run_id is None:
        candidates = sorted(
            (path for path in runs_dir.iterdir() if path.is_dir() and (path / "seal.json").exists()),
            reverse=True,
        )
        if not candidates:
            raise AuditError("没有可验证的封存运行记录")
        run_dir = candidates[0]
    else:
        run_dir = runs_dir / run_id
        if not run_dir.is_dir():
            raise AuditError(f"运行记录不存在：{run_id}")
    return verify_sealed_directory(run_dir)
