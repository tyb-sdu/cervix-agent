from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


MW_MAX = 500.0
LOGP_MAX = 5.0
HBD_MAX = 5
HBA_MAX = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rdkit():
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    return Chem, Crippen, Descriptors, Lipinski


def run_prefilter(input_db: Path, output_root: Path, run_id: str) -> dict[str, object]:
    Chem, Crippen, Descriptors, Lipinski = _load_rdkit()
    input_db = input_db.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / run_id
    if final_dir.exists():
        raise RuntimeError(f"output already exists: {final_dir}")
    building_dir = output_root / f".{run_id}.building"
    building_dir.mkdir(parents=False, exist_ok=False)

    queue_path = building_dir / "unique_structure_queue.tsv"
    summary_path = building_dir / "summary.json"
    manifest_path = building_dir / "manifest.json"

    counters: Counter[str] = Counter()
    fieldnames = [
        "representative_record_key",
        "source_name",
        "source_id",
        "original_smiles",
        "canonical_smiles",
        "fragment_count",
        "calculation_status",
        "mw",
        "logp",
        "hbd",
        "hba",
        "lipinski_fail_fields",
        "michael_status",
        "pains_status",
        "note",
    ]

    connection = sqlite3.connect(f"file:{input_db.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT
                record_key AS representative_record_key,
                source_name,
                source_id,
                original_smiles,
                canonical_smiles,
                fragment_count,
                validation_status,
                duplicate_of,
                message
            FROM compound_record
            WHERE validation_status='valid' AND duplicate_of IS NULL
            ORDER BY sequence_number
            """
        )
        with queue_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in rows:
                counters["unique_valid_input"] += 1
                fragment_count = int(row["fragment_count"] or 0)
                base = {
                    "representative_record_key": row["representative_record_key"],
                    "source_name": row["source_name"],
                    "source_id": row["source_id"],
                    "original_smiles": row["original_smiles"],
                    "canonical_smiles": row["canonical_smiles"] or "",
                    "fragment_count": fragment_count,
                    "calculation_status": "",
                    "mw": "",
                    "logp": "",
                    "hbd": "",
                    "hba": "",
                    "lipinski_fail_fields": "",
                    "michael_status": "not_run",
                    "pains_status": "not_run",
                    "note": "",
                }
                if fragment_count != 1:
                    base["calculation_status"] = "needs_review"
                    base["note"] = (
                        "multi-fragment structure retained; salt/fragment policy not approved"
                    )
                    counters["needs_review_multi_fragment"] += 1
                    writer.writerow(base)
                    continue

                molecule = Chem.MolFromSmiles(row["canonical_smiles"], sanitize=True)
                if molecule is None:
                    base["calculation_status"] = "needs_review"
                    base["note"] = "canonical SMILES could not be reparsed"
                    counters["needs_review_reparse"] += 1
                    writer.writerow(base)
                    continue

                mw = float(Descriptors.MolWt(molecule))
                logp = float(Crippen.MolLogP(molecule))
                hbd = int(Lipinski.NumHDonors(molecule))
                hba = int(Lipinski.NumHAcceptors(molecule))
                failed = []
                if mw > MW_MAX:
                    failed.append("mw")
                if logp > LOGP_MAX:
                    failed.append("logp")
                if hbd > HBD_MAX:
                    failed.append("hbd")
                if hba > HBA_MAX:
                    failed.append("hba")
                base.update(
                    {
                        "calculation_status": (
                            "provisional_lipinski_fail" if failed else "ready_for_next_review"
                        ),
                        "mw": f"{mw:.6f}",
                        "logp": f"{logp:.6f}",
                        "hbd": hbd,
                        "hba": hba,
                        "lipinski_fail_fields": ",".join(failed),
                        "note": (
                            "shadow-mode fixed-threshold tag; not formal P1-04"
                            if failed
                            else "descriptors computed; Michael/PAINS not run"
                        ),
                    }
                )
                counters["single_fragment_descriptor_ready"] += 1
                if failed:
                    counters["provisional_lipinski_fail"] += 1
                else:
                    counters["provisional_lipinski_pass"] += 1
                writer.writerow(base)
    finally:
        connection.close()

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "lossless_prefilter_shadow_mode",
        "formal_p1_04_complete": False,
        "input": {
            "path": str(input_db),
            "sha256": sha256_file(input_db),
        },
        "rules": {
            "invalid_records": "retained in original SQLite; not in unique queue",
            "exact_canonical_duplicates": (
                "one representative is queued; all source mappings remain in original SQLite"
            ),
            "multi_fragment": "needs_review; no salt or fragment removal",
            "mw_max": MW_MAX,
            "logp_max": LOGP_MAX,
            "hbd_max": HBD_MAX,
            "hba_max": HBA_MAX,
            "michael_acceptor": "not_run",
            "pains": "not_run",
        },
        "software": {
            "rdkit": Chem.rdBase.rdkitVersion,
            "mode": "shadow tagging only",
        },
        "counts": dict(counters),
        "outputs": {
            "unique_structure_queue": str(queue_path.relative_to(building_dir)),
            "summary": str(summary_path.relative_to(building_dir)),
            "manifest": str(manifest_path.relative_to(building_dir)),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": utc_now(),
        "source_summary": summary,
        "queue_sha256": sha256_file(queue_path),
        "reversible": True,
        "original_data_modified": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    building_dir.replace(final_dir)
    return {**summary, "relative_path": str(final_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reversible shadow-mode prefilter.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-db", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    root = args.project_root.resolve()
    input_db = args.input_db or (
        root
        / "data/processed/staging_runs/20260715T083632101367Z_P1-02_coconut-lotus-full-staging/compounds.sqlite"
    )
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_lossless_prefilter_shadow"
    )
    result = run_prefilter(input_db, root / "data/processed/prefilter_runs", run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
