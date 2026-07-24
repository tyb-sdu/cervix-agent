from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    required_for: str


def _command_check(command: str, label: str, required_for: str) -> Check:
    location = shutil.which(command)
    if location:
        return Check(label, "available", location, required_for)
    return Check(label, "not_available", f"未在 PATH 中找到 {command}", required_for)


def _module_check(module: str, label: str, required_for: str) -> Check:
    if importlib.util.find_spec(module) is not None:
        return Check(label, "available", f"Python 模块 {module} 可导入", required_for)
    return Check(label, "not_available", f"尚未安装 Python 模块 {module}", required_for)


def run_checks(base_path: Path | None = None) -> list[Check]:
    target = (base_path or Path.cwd()).resolve()
    usage = shutil.disk_usage(target)
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)

    checks: list[Check] = [
        Check(
            "Python",
            "available" if sys.version_info >= (3, 11) else "unsupported",
            platform.python_version(),
            "core",
        ),
        Check("Operating system", "info", platform.platform(), "core"),
        Check(
            "Disk",
            "available" if free_gb >= 5 else "warning",
            f"可用 {free_gb:.1f} GB / 总计 {total_gb:.1f} GB",
            "core",
        ),
        _command_check("git", "Git", "development"),
        _command_check("nvidia-smi", "NVIDIA driver", "gpu"),
        _module_check("rdkit", "RDKit", "phase_1_filtering"),
        _command_check("mk_prepare_ligand.py", "Meeko ligand preparation", "docking"),
        _command_check("mk_prepare_receptor.py", "Meeko receptor preparation", "docking"),
        _command_check("vina", "AutoDock Vina", "docking"),
        _command_check("autodock_gpu_64wi", "AutoDock-GPU", "docking"),
        _command_check("gmx", "GROMACS", "molecular_dynamics"),
    ]
    return checks


def checks_as_dicts(checks: Iterable[Check]) -> list[dict[str, str]]:
    return [asdict(item) for item in checks]

