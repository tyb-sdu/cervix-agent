from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "baselines"
PROJECT_PYTHON = Path("/data2/lxj/miniconda3/envs/cervixagent/bin/python")


def run(command: list[str], timeout: int = 30) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "available": False,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} not found",
        }
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "available": True,
        "executable": executable,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def python_package_status() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
    }
    try:
        import rdkit

        result["rdkit"] = {
            "available": True,
            "version": getattr(rdkit, "__version__", "unknown"),
        }
    except Exception as exc:
        result["rdkit"] = {"available": False, "error": repr(exc)}

    try:
        import torch

        result["torch"] = {
            "available": True,
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:
        result["torch"] = {"available": False, "error": repr(exc)}
    return result


def network_check(url: str) -> dict[str, Any]:
    request = Request(url, method="HEAD", headers={"User-Agent": "CervixAgent/0.2"})
    try:
        with urlopen(request, timeout=15) as response:
            return {
                "reachable": True,
                "status": getattr(response, "status", None),
                "final_url": response.geturl(),
            }
    except HTTPError as exc:
        return {
            "reachable": True,
            "status": exc.code,
            "final_url": exc.geturl(),
            "note": "The host responded with HTTP status.",
        }
    except (URLError, TimeoutError, OSError) as exc:
        return {"reachable": False, "error": repr(exc)}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "project_root": PROJECT_ROOT,
        "literature_root": Path("/data2/lxj/CervixAgent文献"),
        "models_root": Path("/data2/lxj/models"),
        "vectorstore_root": Path("/data2/lxj/vectorstores/cervixagent"),
        "datasets_root": Path("/data2/lxj/datasets/cervixagent"),
        "runs_root": Path("/data2/lxj/runs/cervixagent"),
        "temporary_root": Path("/data2/lxj/tmp/cervixagent"),
    }
    disk = shutil.disk_usage("/data2")
    baseline: dict[str, Any] = {
        "schema_version": 1,
        "baseline_id": "SERVER-BASELINE-20260724-002",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER"),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "paths": {
            key: {"path": str(path), "exists": path.exists()}
            for key, path in paths.items()
        },
        "disk_data2": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "system": {
            "os_release": run(["cat", "/etc/os-release"]),
            "lscpu": run(["lscpu"]),
            "memory": run(["free", "-b"]),
            "filesystem": run(["df", "-B1", "/data2"]),
        },
        "gpu": {
            "nvidia_smi_query": run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,compute_cap",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "nvidia_smi": run(["nvidia-smi"]),
            "nvcc": run(["nvcc", "--version"]),
        },
        "software": {
            "python3": run(["python3", "--version"]),
            "pip3": run(["pip3", "--version"]),
            "git": run(["git", "--version"]),
            "docker": run(["docker", "--version"]),
            "podman": run(["podman", "--version"]),
            "conda": run(["conda", "--version"]),
            "mamba": run(["mamba", "--version"]),
            "gcc": run(["gcc", "--version"]),
            "cmake": run(["cmake", "--version"]),
            "gromacs": run(["gmx", "--version"]),
            "gromacs_mpi": run(["gmx_mpi", "--version"]),
        },
        "python_packages": python_package_status(),
        "project_python_environment": {
            "executable": str(PROJECT_PYTHON),
            "exists": PROJECT_PYTHON.is_file(),
            "python_version": run([str(PROJECT_PYTHON), "--version"]),
            "rdkit_version": run(
                [
                    str(PROJECT_PYTHON),
                    "-c",
                    "import rdkit; print(rdkit.__version__)",
                ]
            ),
            "torch_version": run(
                [
                    str(PROJECT_PYTHON),
                    "-c",
                    (
                        "import torch; "
                        "print(torch.__version__); "
                        "print('cuda_available=' + str(torch.cuda.is_available()))"
                    ),
                ]
            ),
            "unit_tests": run(
                [
                    str(PROJECT_PYTHON),
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(PROJECT_ROOT / "tests"),
                    "-v",
                ],
                timeout=120,
            ),
        },
        "network": {
            "github": network_check("https://github.com/"),
            "huggingface": network_check("https://huggingface.co/"),
            "modelscope": network_check("https://modelscope.cn/"),
            "ncbi": network_check("https://www.ncbi.nlm.nih.gov/"),
            "rcsb_pdb": network_check("https://www.rcsb.org/"),
        },
        "project_state": {
            "workflow_lock_sha256": sha256(
                PROJECT_ROOT / ".cervixagent" / "workflow.lock.json"
            ),
            "preserved_workflow_sha256": sha256(
                PROJECT_ROOT / "protocol" / "original" / "workflow.lock.json"
            ),
            "git_status": run(["git", "-C", str(PROJECT_ROOT), "status", "--short", "--branch"]),
            "git_remote": run(["git", "-C", str(PROJECT_ROOT), "remote", "-v"]),
            "github_reference_commit": "d2eab1f18d71814856ddf8ff62247c5e334a30e2",
        },
    }

    required_paths_exist = all(
        entry["exists"] for entry in baseline["paths"].values()
    )
    workflow_preserved = (
        baseline["project_state"]["workflow_lock_sha256"]
        == baseline["project_state"]["preserved_workflow_sha256"]
        != None
    )
    baseline["checks"] = {
        "required_paths_exist": required_paths_exist,
        "workflow_preserved_exactly": workflow_preserved,
        "nvidia_driver_available": baseline["gpu"]["nvidia_smi_query"]["returncode"] == 0,
        "cuda_toolkit_nvcc_available": baseline["gpu"]["nvcc"]["returncode"] == 0,
        "python3_available": baseline["software"]["python3"]["returncode"] == 0,
        "rdkit_available_in_system_python": baseline["python_packages"]["rdkit"]["available"],
        "pytorch_available_in_system_python": baseline["python_packages"]["torch"]["available"],
        "project_python_environment_exists": baseline["project_python_environment"]["exists"],
        "rdkit_available_in_project_environment": (
            baseline["project_python_environment"]["rdkit_version"]["returncode"] == 0
        ),
        "pytorch_available_in_project_environment": (
            baseline["project_python_environment"]["torch_version"]["returncode"] == 0
        ),
        "project_unit_tests_passed": (
            baseline["project_python_environment"]["unit_tests"]["returncode"] == 0
        ),
        "gromacs_available": (
            baseline["software"]["gromacs"]["returncode"] == 0
            or baseline["software"]["gromacs_mpi"]["returncode"] == 0
        ),
        "all_network_hosts_reachable": all(
            item["reachable"] for item in baseline["network"].values()
        ),
    }
    baseline["status"] = (
        "baseline_recorded"
        if required_paths_exist and workflow_preserved
        else "baseline_failed"
    )

    output_path = OUTPUT_ROOT / "server_environment_baseline_20260724_v2.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    output_hash = sha256(output_path)
    checksum_path = OUTPUT_ROOT / "server_environment_baseline_20260724_v2.sha256"
    checksum_path.write_text(
        f"{output_hash}  {output_path.name}\n",
        encoding="ascii",
    )
    print(json.dumps(baseline["checks"], ensure_ascii=False, indent=2))
    print(f"status={baseline['status']}")
    print(f"output={output_path}")
    print(f"sha256={output_hash}")
    return 0 if baseline["status"] == "baseline_recorded" else 3


if __name__ == "__main__":
    sys.exit(main())
