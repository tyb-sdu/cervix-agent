from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
CORE_PYTHON = PROJECT_ROOT / ".envs" / "core" / "bin" / "python"
RAG_PYTHON = PROJECT_ROOT / ".envs" / "rag" / "bin" / "python"
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "baselines"


def run(command: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    paths = {
        "project_root": PROJECT_ROOT,
        "literature_root": PROJECT_ROOT / "data" / "literature",
        "models_root": PROJECT_ROOT / "models",
        "vectorstores_root": PROJECT_ROOT / "vectorstores",
        "datasets_root": PROJECT_ROOT / "data",
        "runs_root": PROJECT_ROOT / "runs",
        "temporary_root": PROJECT_ROOT / "tmp",
        "environments_root": PROJECT_ROOT / ".envs",
        "core_environment": PROJECT_ROOT / ".envs" / "core",
        "rag_environment": PROJECT_ROOT / ".envs" / "rag",
    }
    external_payloads = {
        "old_literature_root": Path("/data2/lxj/CervixAgent文献").exists(),
        "old_conda_environment": Path(
            "/data2/lxj/miniconda3/envs/cervixagent"
        ).exists(),
        "old_vectorstore_files": any(
            path.is_file()
            for path in Path("/data2/lxj/vectorstores/cervixagent").rglob("*")
        )
        if Path("/data2/lxj/vectorstores/cervixagent").exists()
        else False,
        "old_dataset_files": any(
            path.is_file()
            for path in Path("/data2/lxj/datasets/cervixagent").rglob("*")
        )
        if Path("/data2/lxj/datasets/cervixagent").exists()
        else False,
        "old_run_files": any(
            path.is_file()
            for path in Path("/data2/lxj/runs/cervixagent").rglob("*")
        )
        if Path("/data2/lxj/runs/cervixagent").exists()
        else False,
    }
    disk = shutil.disk_usage("/data2")
    manifest_summary_path = (
        PROJECT_ROOT
        / "manifests"
        / "literature_20260724_project_local"
        / "literature_acceptance_20260724_project_local.json"
    )
    pilot_summary_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "literature"
        / "pilot_100"
        / "pilot_100_summary.json"
    )
    latest_smoke_id = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "literature"
        / "pilot_100"
        / "latest_smoke_run.txt"
    ).read_text(encoding="ascii").strip()
    latest_full_id = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "literature"
        / "pilot_100"
        / "latest_full_run.txt"
    ).read_text(encoding="ascii").strip()
    smoke_summary_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "literature"
        / "pilot_100"
        / "runs"
        / latest_smoke_id
        / "summary.json"
    )
    full_summary_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "literature"
        / "pilot_100"
        / "runs"
        / latest_full_id
        / "summary.json"
    )

    baseline: dict[str, Any] = {
        "schema_version": 1,
        "baseline_id": "PROJECT-LOCAL-RAG-BASELINE-20260724-002",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER"),
        "project_isolation_policy": (
            "All CervixAgent data, environments, models, caches, vector stores, "
            "and runtime outputs must remain under the project root."
        ),
        "paths": {
            key: {
                "path": str(path),
                "exists": path.exists(),
                "under_project_root": (
                    path == PROJECT_ROOT or PROJECT_ROOT in path.parents
                ),
            }
            for key, path in paths.items()
        },
        "external_payloads": external_payloads,
        "disk_data2": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "git": {
            "head": run(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]
            ),
            "remote": run(
                ["git", "-C", str(PROJECT_ROOT), "remote", "-v"]
            ),
        },
        "gpu": {
            "inventory": run(
                [
                    "nvidia-smi",
                    (
                        "--query-gpu=index,name,driver_version,memory.total,"
                        "memory.free,compute_cap"
                    ),
                    "--format=csv,noheader,nounits",
                ]
            ),
        },
        "core_environment": {
            "python": run([str(CORE_PYTHON), "--version"]),
            "rdkit": run(
                [
                    str(CORE_PYTHON),
                    "-c",
                    "import rdkit; print(rdkit.__version__)",
                ]
            ),
            "unit_tests": run(
                [
                    str(CORE_PYTHON),
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(PROJECT_ROOT / "tests"),
                    "-v",
                ]
            ),
        },
        "rag_environment": {
            "python": run([str(RAG_PYTHON), "--version"]),
            "gpu": run(
                [
                    str(RAG_PYTHON),
                    "-c",
                    (
                        "import json, torch; print(json.dumps({"
                        "'torch': torch.__version__, "
                        "'cuda_runtime': torch.version.cuda, "
                        "'cuda_available': torch.cuda.is_available(), "
                        "'gpu_count': torch.cuda.device_count(), "
                        "'gpu_names': [torch.cuda.get_device_name(i) "
                        "for i in range(torch.cuda.device_count())]"
                        "}))"
                    ),
                ]
            ),
            "imports": run(
                [
                    str(RAG_PYTHON),
                    "-c",
                    (
                        "import docling, lxml, pandas, pyarrow, pymupdf, "
                        "pypdf, qdrant_client, sentence_transformers; "
                        "print('passed')"
                    ),
                ]
            ),
            "versions": run(
                [
                    str(RAG_PYTHON),
                    "-c",
                    (
                        "from importlib.metadata import version; "
                        "print('docling=' + version('docling')); "
                        "print('sentence-transformers=' + "
                        "version('sentence-transformers')); "
                        "print('qdrant-client=' + version('qdrant-client'))"
                    ),
                ]
            ),
        },
        "literature_manifest": load_json(manifest_summary_path),
        "pilot_selection": load_json(pilot_summary_path),
        "smoke_parse": load_json(smoke_summary_path),
        "full_parse": load_json(full_summary_path),
    }
    checks = {
        "all_required_paths_exist": all(
            entry["exists"] for entry in baseline["paths"].values()
        ),
        "all_configured_paths_under_project_root": all(
            entry["under_project_root"]
            for entry in baseline["paths"].values()
        ),
        "no_external_project_payloads": not any(external_payloads.values()),
        "git_head_matches_github": (
            baseline["git"]["head"]["stdout"]
            == "623c4b9846ddc5be25124590a238366dd24834ff"
        ),
        "two_rtx_4090_visible": (
            baseline["gpu"]["inventory"]["returncode"] == 0
            and baseline["gpu"]["inventory"]["stdout"].count(
                "NVIDIA GeForce RTX 4090"
            )
            == 2
        ),
        "rdkit_available": baseline["core_environment"]["rdkit"]["returncode"]
        == 0,
        "core_unit_tests_passed": (
            baseline["core_environment"]["unit_tests"]["returncode"] == 0
        ),
        "rag_gpu_validation_passed": (
            baseline["rag_environment"]["gpu"]["returncode"] == 0
            and '"cuda_available": true'
            in baseline["rag_environment"]["gpu"]["stdout"]
            and '"gpu_count": 2'
            in baseline["rag_environment"]["gpu"]["stdout"]
        ),
        "rag_imports_passed": (
            baseline["rag_environment"]["imports"]["returncode"] == 0
        ),
        "literature_manifest_accepted": (
            baseline["literature_manifest"]["status"] == "accepted"
            and baseline["literature_manifest"]["file_count"] == 2970
            and baseline["literature_manifest"]["total_bytes"]
            == 6_097_732_334
        ),
        "pilot_selection_ready": (
            baseline["pilot_selection"]["status"] == "ready_for_parsing"
            and baseline["pilot_selection"]["selection_count"] == 100
        ),
        "smoke_parse_completed": (
            baseline["smoke_parse"]["status"] == "completed"
            and baseline["smoke_parse"]["status_counts"].get("parsed") == 12
        ),
        "full_parse_completed": (
            baseline["full_parse"]["status"] == "completed"
            and baseline["full_parse"]["status_counts"].get("parsed") == 100
        ),
    }
    baseline["checks"] = checks
    baseline["status"] = "accepted" if all(checks.values()) else "review_required"

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / "project_local_rag_baseline_20260724_v2.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    checksum = sha256(output_path)
    (OUTPUT_ROOT / "project_local_rag_baseline_20260724_v2.sha256").write_text(
        f"{checksum}  {output_path.name}\n",
        encoding="ascii",
    )
    print(json.dumps({"status": baseline["status"], "checks": checks}, indent=2))
    print(f"sha256={checksum}")
    return 0 if baseline["status"] == "accepted" else 3


if __name__ == "__main__":
    sys.exit(main())
