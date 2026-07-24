from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .audit import seal_directory, verify_sealed_directory
from .config import load_compound_ingestion_schema, load_public_sources
from .data import iter_coconut_smiles, iter_lotus_smiles, sha256_file
from .project import load_project


class StagingError(RuntimeError):
    pass


ProgressCallback = Callable[[str, int], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rdkit():
    try:
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise StagingError("RDKit 尚未安装，不能执行 P1-02 两源暂存") from exc
    return Chem, rdBase


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


@contextmanager
def _exclusive_ingestion_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".cervixagent" / "p1_02_ingestion.lock"
    try:
        with lock_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump({"pid": os.getpid(), "created_at": _utc_now()}, handle)
            handle.write("\n")
    except FileExistsError as exc:
        raise StagingError(
            f"检测到另一个 P1-02 入库任务或遗留锁：{lock_path}；"
            "确认没有任务运行后再人工处理该锁"
        ) from exc
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _verify_inputs(root: Path) -> list[dict[str, Any]]:
    registry = load_public_sources()["sources"]
    manifest_path = root / "data" / "raw" / "download_manifest.json"
    if not manifest_path.exists():
        raise StagingError("缺少 data/raw/download_manifest.json，不能验证原始输入")
    download_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs: list[dict[str, Any]] = []
    for source_name, source_key in (
        ("COCONUT", "coconut_drug_discovery"),
        ("LOTUS", "lotus_smiles"),
    ):
        source = registry[source_key]
        path = root / source["relative_path"]
        if not path.is_file():
            raise StagingError(f"缺少 {source_name} 原始文件：{path}")
        recorded = download_manifest.get("files", {}).get(source_key)
        if not recorded:
            raise StagingError(f"下载清单中缺少 {source_key} 记录")
        actual_hash = sha256_file(path)
        if actual_hash != recorded.get("sha256"):
            raise StagingError(f"{source_name} 原始文件哈希与下载清单不一致，拒绝入库")
        inputs.append(
            {
                "source_name": source_name,
                "source_key": source_key,
                "path": path,
                "relative_path": source["relative_path"],
                "sha256": actual_hash,
                "bytes": path.stat().st_size,
                "url": source["url"],
                "landing_page": source["landing_page"],
                "version": source["version"],
                "license": source["license"],
            }
        )
    return inputs


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA user_version = 1;
        CREATE TABLE run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE source_file (
            source_name TEXT PRIMARY KEY,
            source_key TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            url TEXT NOT NULL,
            landing_page TEXT NOT NULL,
            version TEXT NOT NULL,
            license TEXT NOT NULL
        );
        CREATE TABLE canonical_structure (
            canonical_smiles TEXT PRIMARY KEY,
            first_record_key TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE compound_record (
            sequence_number INTEGER PRIMARY KEY,
            record_key TEXT NOT NULL UNIQUE,
            source_name TEXT NOT NULL,
            source_id TEXT NOT NULL,
            original_smiles TEXT NOT NULL,
            canonical_smiles TEXT,
            fragment_count INTEGER,
            validation_status TEXT NOT NULL CHECK(validation_status IN ('valid', 'invalid')),
            duplicate_of TEXT,
            message TEXT NOT NULL
        );
        CREATE INDEX idx_compound_source ON compound_record(source_name, source_id);
        CREATE INDEX idx_compound_status ON compound_record(validation_status);
        CREATE INDEX idx_compound_duplicate ON compound_record(duplicate_of);
        """
    )


def _insert_source_files(
    connection: sqlite3.Connection, inputs: list[dict[str, Any]]
) -> None:
    connection.executemany(
        """
        INSERT INTO source_file(
            source_name, source_key, relative_path, sha256, bytes,
            url, landing_page, version, license
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                item["source_name"],
                item["source_key"],
                item["relative_path"],
                item["sha256"],
                item["bytes"],
                item["url"],
                item["landing_page"],
                item["version"],
                item["license"],
            )
            for item in inputs
        ],
    )


def _source_records(source_name: str, path: Path) -> Iterator[tuple[str, str]]:
    if source_name == "COCONUT":
        yield from iter_coconut_smiles(path)
        return
    if source_name == "LOTUS":
        yield from iter_lotus_smiles(path)
        return
    raise StagingError(f"没有为 {source_name} 配置解析器")


def _process_records(
    connection: sqlite3.Connection,
    inputs: list[dict[str, Any]],
    rejects_path: Path,
    *,
    batch_size: int,
    limit_per_source: int | None,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    Chem, rdBase = _load_rdkit()
    total = valid = invalid = duplicates = 0
    by_source: dict[str, dict[str, int]] = {}
    cursor = connection.cursor()
    connection.execute("BEGIN")
    with rejects_path.open("x", encoding="utf-8", newline="") as reject_handle:
        reject_writer = csv.writer(reject_handle, delimiter="\t", lineterminator="\n")
        reject_writer.writerow(
            ["record_key", "source_name", "source_id", "original_smiles", "message"]
        )
        with rdBase.BlockLogs():
            for source in inputs:
                source_name = source["source_name"]
                prefix = "COC" if source_name == "COCONUT" else "LOT"
                stats = {"input": 0, "valid": 0, "invalid": 0, "duplicates": 0}
                by_source[source_name] = stats
                for source_index, (source_id, smiles) in enumerate(
                    _source_records(source_name, source["path"]), start=1
                ):
                    if limit_per_source is not None and source_index > limit_per_source:
                        break
                    stats["input"] += 1
                    total += 1
                    record_key = f"{prefix}-{source_index:09d}"
                    molecule = Chem.MolFromSmiles(smiles, sanitize=True) if smiles else None
                    if molecule is None:
                        invalid += 1
                        stats["invalid"] += 1
                        message = "RDKit 无法解析或清理该 SMILES；原始记录已保留"
                        cursor.execute(
                            """
                            INSERT INTO compound_record VALUES (?, ?, ?, ?, ?, NULL, NULL, 'invalid', NULL, ?)
                            """,
                            (total, record_key, source_name, source_id, smiles, message),
                        )
                        reject_writer.writerow(
                            [record_key, source_name, source_id, smiles, message]
                        )
                    else:
                        canonical = Chem.MolToSmiles(
                            molecule, canonical=True, isomericSmiles=True
                        )
                        fragment_count = len(Chem.GetMolFrags(molecule))
                        cursor.execute(
                            "INSERT OR IGNORE INTO canonical_structure VALUES (?, ?)",
                            (canonical, record_key),
                        )
                        duplicate_of = None
                        if cursor.rowcount == 0:
                            row = cursor.execute(
                                "SELECT first_record_key FROM canonical_structure WHERE canonical_smiles = ?",
                                (canonical,),
                            ).fetchone()
                            duplicate_of = row[0] if row else None
                            duplicates += 1
                            stats["duplicates"] += 1
                        valid += 1
                        stats["valid"] += 1
                        cursor.execute(
                            """
                            INSERT INTO compound_record VALUES (?, ?, ?, ?, ?, ?, ?, 'valid', ?, ?)
                            """,
                            (
                                total,
                                record_key,
                                source_name,
                                source_id,
                                smiles,
                                canonical,
                                fragment_count,
                                duplicate_of,
                                "规范结构重复；记录未删除" if duplicate_of else "解析通过",
                            ),
                        )
                    if total % batch_size == 0:
                        connection.commit()
                        connection.execute("BEGIN")
                    if progress is not None and stats["input"] % 50000 == 0:
                        progress(source_name, stats["input"])
                connection.commit()
                connection.execute("BEGIN")
    connection.commit()
    unique_valid = connection.execute(
        "SELECT COUNT(*) FROM canonical_structure"
    ).fetchone()[0]
    return {
        "input_records": total,
        "valid_records": valid,
        "invalid_records": invalid,
        "canonical_duplicate_records": duplicates,
        "unique_valid_structures": unique_valid,
        "by_source": by_source,
    }


def stage_public_snapshots(
    project_path: Path,
    *,
    label: str = "coconut-lotus-staging",
    batch_size: int = 10000,
    limit_per_source: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    root = project_path.expanduser().resolve()
    state = load_project(root)
    if state.get("current_step") != "P1-02":
        raise StagingError(
            f"当前步骤是 {state.get('current_step')}；两源暂存只允许在 P1-02 执行"
        )
    if batch_size < 100:
        raise StagingError("batch_size 不能小于 100")
    if limit_per_source is not None and limit_per_source < 1:
        raise StagingError("limit_per_source 必须是正整数")
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip(".-")[:64]
    if not safe_label:
        raise StagingError("暂存标签必须至少包含一个字母或数字")

    inputs = _verify_inputs(root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}_P1-02_{safe_label}"
    staging_root = root / "data" / "processed" / "staging_runs"
    staging_root.mkdir(parents=True, exist_ok=True)
    building_dir = staging_root / f".{run_id}.building"
    final_dir = staging_root / run_id

    with _exclusive_ingestion_lock(root):
        building_dir.mkdir(parents=False, exist_ok=False)
        database_path = building_dir / "compounds.sqlite"
        rejects_path = building_dir / "rejects.tsv"
        connection: sqlite3.Connection | None = None
        try:
            _write_json(building_dir / "contract.json", load_compound_ingestion_schema())
            connection = sqlite3.connect(database_path)
            connection.isolation_level = None
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute("PRAGMA temp_store = MEMORY")
            connection.execute("PRAGMA cache_size = -200000")
            _create_schema(connection)
            connection.execute("BEGIN")
            _insert_source_files(connection, inputs)
            metadata = {
                "run_id": run_id,
                "created_at": _utc_now(),
                "step_id": "P1-02",
                "scope": "two_source_public_snapshot_staging",
                "formal_p1_02_complete": "false",
                "rdkit_version": importlib.metadata.version("rdkit"),
            }
            connection.executemany(
                "INSERT INTO run_metadata(key, value) VALUES (?, ?)", metadata.items()
            )
            connection.commit()
            counts = _process_records(
                connection,
                inputs,
                rejects_path,
                batch_size=batch_size,
                limit_per_source=limit_per_source,
                progress=progress,
            )
            integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
            if integrity != "ok":
                raise StagingError(f"SQLite quick_check 失败：{integrity}")
            connection.close()
            connection = None

            summary = {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": metadata["created_at"],
                "step_id": "P1-02",
                "scope": "two_source_public_snapshot_staging",
                "formal_p1_02_complete": False,
                "completion_blockers": [
                    "ECNPDB identity, source, license and data definition are unresolved",
                    "Computational-chemistry standardization rules are not approved",
                    "This run contains only COCONUT and LOTUS snapshots",
                ],
                "software": {
                    "rdkit": metadata["rdkit_version"],
                    "license": "BSD-3-Clause",
                    "sqlite": sqlite3.sqlite_version,
                },
                "limited_run": limit_per_source is not None,
                "limit_per_source": limit_per_source,
                "input_files": [
                    {key: value for key, value in item.items() if key != "path"}
                    for item in inputs
                ],
                "counts": counts,
                "decisions": {
                    "salt_or_fragment_removal": False,
                    "tautomer_normalization": False,
                    "protonation_or_charge_normalization": False,
                    "stereochemistry_removal": False,
                    "duplicate_record_removal": False,
                    "michael_acceptor_filter": False,
                    "lipinski_filter": False,
                    "pains_filter": False,
                },
                "sqlite_quick_check": integrity,
                "outputs": {"database": "compounds.sqlite", "rejects": "rejects.tsv"},
            }
            _write_json(building_dir / "summary.json", summary)
            seal = seal_directory(building_dir)
            verification = verify_sealed_directory(building_dir)
            if not verification["valid"]:
                raise StagingError(
                    "暂存结果封存验证失败：" + "; ".join(verification["errors"])
                )
            building_dir.replace(final_dir)
            return {
                **summary,
                "relative_path": str(final_dir.relative_to(root)),
                "aggregate_sha256": seal["aggregate_sha256"],
            }
        except Exception as exc:
            if connection is not None:
                connection.close()
            if building_dir.exists():
                failure_path = building_dir / "failure.json"
                if not failure_path.exists():
                    _write_json(
                        failure_path,
                        {
                            "run_id": run_id,
                            "failed_at": _utc_now(),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                if not (building_dir / "seal.json").exists():
                    seal_directory(building_dir)
                failed_dir = staging_root / f"{run_id}.failed"
                building_dir.replace(failed_dir)
            if isinstance(exc, StagingError):
                raise
            raise StagingError(f"两源暂存失败：{exc}") from exc


def verify_staging_run(
    project_path: Path, run_id: str | None = None
) -> dict[str, Any]:
    root = project_path.expanduser().resolve()
    load_project(root)
    staging_root = root / "data" / "processed" / "staging_runs"
    if not staging_root.is_dir():
        raise StagingError("尚无两源暂存运行记录")
    if run_id is None:
        candidates = sorted(
            (
                path
                for path in staging_root.iterdir()
                if path.is_dir()
                and not path.name.endswith(".failed")
                and (path / "seal.json").exists()
            ),
            reverse=True,
        )
        if not candidates:
            raise StagingError("尚无可验证的成功暂存记录")
        run_dir = candidates[0]
    else:
        run_dir = staging_root / run_id
        if not run_dir.is_dir():
            raise StagingError(f"两源暂存记录不存在：{run_id}")
    verification = verify_sealed_directory(run_dir)
    summary_path = run_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    database_checks: dict[str, Any] = {}
    database_path = run_dir / "compounds.sqlite"
    if not database_path.exists():
        verification["errors"].append("缺少 compounds.sqlite")
    else:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
            quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
            actual_counts = {
                "input_records": connection.execute(
                    "SELECT COUNT(*) FROM compound_record"
                ).fetchone()[0],
                "valid_records": connection.execute(
                    "SELECT COUNT(*) FROM compound_record WHERE validation_status='valid'"
                ).fetchone()[0],
                "invalid_records": connection.execute(
                    "SELECT COUNT(*) FROM compound_record WHERE validation_status='invalid'"
                ).fetchone()[0],
                "canonical_duplicate_records": connection.execute(
                    "SELECT COUNT(*) FROM compound_record WHERE duplicate_of IS NOT NULL"
                ).fetchone()[0],
                "unique_valid_structures": connection.execute(
                    "SELECT COUNT(*) FROM canonical_structure"
                ).fetchone()[0],
            }
            orphan_duplicates = connection.execute(
                """
                SELECT COUNT(*)
                FROM compound_record AS record
                LEFT JOIN compound_record AS first_record
                  ON record.duplicate_of = first_record.record_key
                WHERE record.duplicate_of IS NOT NULL
                  AND first_record.record_key IS NULL
                """
            ).fetchone()[0]
            database_checks = {
                "quick_check": quick_check,
                "actual_counts": actual_counts,
                "orphan_duplicate_references": orphan_duplicates,
            }
            if quick_check != "ok":
                verification["errors"].append(f"SQLite quick_check 失败：{quick_check}")
            if orphan_duplicates != 0:
                verification["errors"].append(
                    f"存在 {orphan_duplicates} 条找不到首条记录的重复引用"
                )
            summary_counts = summary.get("counts", {})
            for key, value in actual_counts.items():
                if summary_counts.get(key) != value:
                    verification["errors"].append(
                        f"summary.json 与 SQLite 计数不一致：{key}"
                    )
            if actual_counts["valid_records"] + actual_counts["invalid_records"] != actual_counts["input_records"]:
                verification["errors"].append("有效数与无效数之和不等于输入记录数")
            if actual_counts["valid_records"] - actual_counts["canonical_duplicate_records"] != actual_counts["unique_valid_structures"]:
                verification["errors"].append("有效、重复与唯一结构计数关系不成立")
        except sqlite3.Error as exc:
            verification["errors"].append(f"无法只读核验 SQLite：{exc}")
        finally:
            if connection is not None:
                connection.close()
    if summary.get("formal_p1_02_complete") is not False:
        verification["errors"].append("两源暂存不得标记 P1-02 正式完成")
    for filter_name in ("michael_acceptor_filter", "lipinski_filter", "pains_filter"):
        if summary.get("decisions", {}).get(filter_name) is not False:
            verification["errors"].append(f"两源暂存中 {filter_name} 必须保持关闭")
    verification["valid"] = not verification["errors"]
    verification.update(
        {
            "scope": summary.get("scope"),
            "formal_p1_02_complete": summary.get("formal_p1_02_complete"),
            "counts": summary.get("counts"),
            "sqlite_quick_check": database_checks.get("quick_check"),
            "database_checks": database_checks,
        }
    )
    return verification
