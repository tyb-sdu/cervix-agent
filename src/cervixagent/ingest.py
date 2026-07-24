from __future__ import annotations

import csv
import importlib.metadata
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import seal_directory, verify_sealed_directory
from .data import sha256_file
from .project import load_project


class IngestError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rdkit():
    try:
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise IngestError(
            "RDKit 尚未安装；请在隔离环境中安装项目的 chem 可选依赖"
        ) from exc
    return Chem, rdBase


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def ingest_engineering_test(
    project_path: Path,
    input_path: Path | None = None,
    *,
    label: str = "engineering-test",
) -> dict[str, Any]:
    """Validate and canonically serialize a test set without scientific filtering."""
    root = project_path.expanduser().resolve()
    state = load_project(root)
    if state.get("current_step") != "P1-02":
        raise IngestError(
            f"当前步骤是 {state.get('current_step')}；工程入库试运行只允许在 P1-02 执行"
        )
    source_path = (
        input_path.expanduser().resolve()
        if input_path is not None
        else root / "data/processed/test_compounds_500.tsv"
    )
    if not source_path.is_file():
        raise IngestError(f"输入文件不存在：{source_path}")

    with source_path.open("r", encoding="utf-8-sig", newline="") as header_handle:
        header_reader = csv.DictReader(header_handle, delimiter="\t")
        required_fields = {"source", "source_id", "smiles"}
        missing_fields = required_fields - set(header_reader.fieldnames or [])
    if missing_fields:
        raise IngestError("输入 TSV 缺少字段：" + ", ".join(sorted(missing_fields)))

    Chem, rdBase = _load_rdkit()
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip(".-")[:64]
    if not safe_label:
        raise IngestError("入库标签必须至少包含一个字母或数字")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}_P1-02_{safe_label}"
    output_dir = root / "data" / "processed" / "ingestion_runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "records.tsv"
    rejects_path = output_dir / "rejects.tsv"

    output_fields = [
        "record_id",
        "source",
        "source_id",
        "original_smiles",
        "canonical_smiles",
        "fragment_count",
        "validation_status",
        "duplicate_of",
        "message",
    ]
    total = 0
    valid = 0
    invalid = 0
    duplicates = 0
    source_counts: Counter[str] = Counter()
    first_by_canonical: dict[str, str] = {}

    with (
        source_path.open("r", encoding="utf-8-sig", newline="") as input_handle,
        records_path.open("x", encoding="utf-8", newline="") as records_handle,
        rejects_path.open("x", encoding="utf-8", newline="") as rejects_handle,
    ):
        reader = csv.DictReader(input_handle, delimiter="\t")
        records_writer = csv.DictWriter(
            records_handle, fieldnames=output_fields, delimiter="\t", lineterminator="\n"
        )
        rejects_writer = csv.DictWriter(
            rejects_handle, fieldnames=output_fields, delimiter="\t", lineterminator="\n"
        )
        records_writer.writeheader()
        rejects_writer.writeheader()
        for row_number, row in enumerate(reader, start=1):
            total += 1
            record_id = (row.get("record_id") or f"ROW-{row_number:07d}").strip()
            source = (row.get("source") or "").strip()
            source_id = (row.get("source_id") or "").strip()
            smiles = (row.get("smiles") or "").strip()
            source_counts[source or "UNKNOWN"] += 1
            result = {
                "record_id": record_id,
                "source": source,
                "source_id": source_id,
                "original_smiles": smiles,
                "canonical_smiles": "",
                "fragment_count": "",
                "validation_status": "invalid",
                "duplicate_of": "",
                "message": "",
            }
            molecule = None
            if smiles:
                with rdBase.BlockLogs():
                    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
            if molecule is None:
                invalid += 1
                result["message"] = "RDKit 无法解析或清理该 SMILES；原始记录已保留"
                records_writer.writerow(result)
                rejects_writer.writerow(result)
                continue

            canonical = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
            )
            fragment_count = len(Chem.GetMolFrags(molecule))
            duplicate_of = first_by_canonical.get(canonical, "")
            if duplicate_of:
                duplicates += 1
            else:
                first_by_canonical[canonical] = record_id
            valid += 1
            result.update(
                {
                    "canonical_smiles": canonical,
                    "fragment_count": str(fragment_count),
                    "validation_status": "valid",
                    "duplicate_of": duplicate_of,
                    "message": (
                        "规范 SMILES 重复；记录未删除"
                        if duplicate_of
                        else "解析通过"
                    ),
                }
            )
            records_writer.writerow(result)

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": _utc_now(),
        "step_id": "P1-02",
        "scope": "engineering_test_only",
        "formal_p1_02_complete": False,
        "purpose": "验证 P1-02 入库代码，不代表三库正式入库完成",
        "software": {
            "rdkit": importlib.metadata.version("rdkit"),
            "license": "BSD-3-Clause",
        },
        "input": {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        "decisions": {
            "rdkit_parse_and_sanitize": True,
            "canonical_isomeric_smiles_serialization": True,
            "salt_or_fragment_removal": False,
            "tautomer_normalization": False,
            "protonation_or_charge_normalization": False,
            "stereochemistry_removal": False,
            "duplicate_record_removal": False,
            "michael_acceptor_filter": False,
            "lipinski_filter": False,
            "pains_filter": False,
        },
        "counts": {
            "input_records": total,
            "valid_records": valid,
            "invalid_records": invalid,
            "canonical_duplicate_records": duplicates,
            "unique_valid_structures": len(first_by_canonical),
            "by_source": dict(sorted(source_counts.items())),
        },
        "source_coverage": {
            "ECNPDB": "unresolved_not_substituted",
            "COCONUT": "engineering_sample_only",
            "LOTUS": "engineering_sample_only",
        },
        "outputs": {
            "records": "records.tsv",
            "rejects": "rejects.tsv",
        },
    }
    _write_manifest(output_dir / "manifest.json", manifest)
    seal = seal_directory(output_dir)
    return {
        **manifest,
        "relative_path": str(output_dir.relative_to(root)),
        "aggregate_sha256": seal["aggregate_sha256"],
    }


def verify_ingestion_run(
    project_path: Path, run_id: str | None = None
) -> dict[str, Any]:
    root = project_path.expanduser().resolve()
    load_project(root)
    runs_dir = root / "data" / "processed" / "ingestion_runs"
    if not runs_dir.is_dir():
        raise IngestError("尚无 P1-02 入库运行记录")
    if run_id is None:
        candidates = sorted(
            (
                path
                for path in runs_dir.iterdir()
                if path.is_dir() and (path / "seal.json").exists()
            ),
            reverse=True,
        )
        if not candidates:
            raise IngestError("尚无可验证的 P1-02 入库运行记录")
        run_dir = candidates[0]
    else:
        run_dir = runs_dir / run_id
        if not run_dir.is_dir():
            raise IngestError(f"入库运行记录不存在：{run_id}")
    verification = verify_sealed_directory(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    verification.update(
        {
            "step_id": manifest.get("step_id"),
            "scope": manifest.get("scope"),
            "formal_p1_02_complete": manifest.get("formal_p1_02_complete"),
            "counts": manifest.get("counts"),
        }
    )
    return verification
