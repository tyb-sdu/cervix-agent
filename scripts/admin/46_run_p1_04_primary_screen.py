from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PREFILTER_RUN = "20260729T041834Z_lossless_prefilter_shadow"
DEFAULT_RULE_CONFIG = "20260729_p1_04_conservative_michael_pains.json"
FIXED_LIPINSKI = {"mw_max": 500.0, "logp_max": 5.0, "hbd_max": 5, "hba_max": 10}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rdkit() -> tuple[Any, Any, Any, Any, Any]:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski

    return Chem, Crippen, Descriptors, FilterCatalog, Lipinski


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"rule config field {key!r} must be an object")
    return value


def load_rule_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise RuntimeError(f"unsupported rule config schema: {path}")

    lipinski = _require_mapping(payload, "lipinski")
    for key, expected in FIXED_LIPINSKI.items():
        if key not in lipinski or float(lipinski[key]) != float(expected):
            raise RuntimeError(
                f"rule config {key} must preserve fixed protocol value {expected!r}"
            )
    integrity = _require_mapping(payload, "integrity_controls")
    required_integrity = {
        "input_snapshot": str,
        "require_single_fragment": bool,
        "recompute_lipinski_descriptors": bool,
        "descriptor_comparison_tolerance": (float, int),
        "on_input_inconsistency": str,
    }
    for key, expected_type in required_integrity.items():
        if not isinstance(integrity.get(key), expected_type):
            raise RuntimeError(f"invalid integrity control: {key}")
    if not integrity["require_single_fragment"] or not integrity["recompute_lipinski_descriptors"]:
        raise RuntimeError("P1-04 requires single-fragment and independent descriptor checks")

    michael = _require_mapping(payload, "michael_acceptor")
    rules = michael.get("rules")
    if not isinstance(rules, list) or len(rules) != 4:
        raise RuntimeError("rule config must contain exactly four Michael rule records")
    expected_ids = {
        "alpha_beta_unsaturated_carbonyl",
        "conjugated_lactone",
        "nitroalkene",
        "quinone",
    }
    actual_ids = {rule.get("id") for rule in rules if isinstance(rule, dict)}
    if actual_ids != expected_ids:
        raise RuntimeError(f"unexpected Michael rule IDs: {sorted(actual_ids)}")
    if any(not isinstance(rule.get("smarts"), str) or not rule["smarts"] for rule in rules):
        raise RuntimeError("every Michael rule must contain a nonempty SMARTS string")

    pains = _require_mapping(payload, "pains")
    if pains.get("implementation") != "RDKit FilterCatalog":
        raise RuntimeError("P1-04 PAINS implementation must be RDKit FilterCatalog")
    if pains.get("catalog") != "PAINS (the RDKit union of PAINS_A, PAINS_B and PAINS_C)":
        raise RuntimeError("P1-04 PAINS catalog must be the recorded PAINS A+B+C union")
    if pains.get("expected_entry_count") != 480:
        raise RuntimeError("P1-04 PAINS catalog must contain the recorded 480 entries")
    return payload


def build_queries(Chem: Any, config: dict[str, Any]) -> dict[str, Any]:
    queries: dict[str, Any] = {}
    for rule in config["michael_acceptor"]["rules"]:
        query = Chem.MolFromSmarts(rule["smarts"])
        if query is None:
            raise RuntimeError(f"invalid SMARTS for {rule['id']}: {rule['smarts']}")
        queries[rule["id"]] = query
    return queries


def validation_cases() -> list[tuple[str, str, set[str]]]:
    return [
        ("methyl_vinyl_ketone", "CC(=O)C=C", {"alpha_beta_unsaturated_carbonyl"}),
        ("methyl_acrylate", "COC(=O)C=C", {"alpha_beta_unsaturated_carbonyl"}),
        ("saturated_ketone", "CCC(=O)C", set()),
        (
            "alpha_methylene_gamma_butyrolactone",
            "C=C1CCOC1=O",
            {"alpha_beta_unsaturated_carbonyl", "conjugated_lactone"},
        ),
        ("gamma_butyrolactone", "O=C1OCCC1", set()),
        ("nitroethylene", "C=C[N+](=O)[O-]", {"nitroalkene"}),
        ("nitromethane", "C[N+](=O)[O-]", set()),
        ("nitrobenzene", "O=[N+]([O-])c1ccccc1", set()),
        (
            "p_benzoquinone",
            "O=C1C=CC(=O)C=C1",
            {"alpha_beta_unsaturated_carbonyl", "quinone"},
        ),
        (
            "o_benzoquinone",
            "O=C1C(=O)C=CC=C1",
            {"alpha_beta_unsaturated_carbonyl", "quinone"},
        ),
        ("catechol", "Oc1ccccc1O", set()),
    ]


def validate_rules(Chem: Any, queries: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for label, smiles, expected in validation_cases():
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
        if molecule is None:
            raise RuntimeError(f"invalid built-in validation SMILES: {label}")
        observed = {name for name, query in queries.items() if molecule.HasSubstructMatch(query)}
        passed = observed == expected
        results.append(
            {
                "label": label,
                "smiles": smiles,
                "expected_rule_types": sorted(expected),
                "observed_rule_types": sorted(observed),
                "passed": passed,
            }
        )
        if not passed:
            failures.append(label)
    if failures:
        raise RuntimeError(f"SMARTS validation failed for: {', '.join(failures)}")
    return results


def catalog_fingerprint(catalog: Any) -> str:
    digest = hashlib.sha256()
    for index in range(catalog.GetNumEntries()):
        entry = catalog.GetEntryWithIdx(index)
        digest.update(f"{index}\t{entry.GetDescription()}\n".encode("utf-8"))
    return digest.hexdigest()


def build_pains_catalog(FilterCatalog: Any, expected_entry_count: int) -> tuple[Any, str]:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    observed_entry_count = catalog.GetNumEntries()
    if observed_entry_count != expected_entry_count:
        raise RuntimeError(
            "RDKit PAINS catalog size differs from recorded config: "
            f"expected {expected_entry_count}, observed {observed_entry_count}"
        )
    return catalog, catalog_fingerprint(catalog)


def output_fieldnames() -> list[str]:
    return [
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
        "recomputed_mw",
        "recomputed_logp",
        "recomputed_hbd",
        "recomputed_hba",
        "recomputed_lipinski_fail_fields",
        "input_integrity_status",
        "input_integrity_notes",
        "michael_status",
        "michael_match_types",
        "michael_rule_type_count",
        "pains_status",
        "pains_description_count",
        "pains_descriptions",
        "primary_filter_status",
        "primary_filter_reasons",
        "screening_note",
    ]


def default_input_queue(root: Path) -> Path:
    return (
        root
        / "data/processed/prefilter_runs"
        / DEFAULT_PREFILTER_RUN
        / "unique_structure_queue.tsv"
    )


def number_matches(value: str, observed: float, tolerance: float) -> bool:
    if value == "":
        return False
    try:
        return abs(float(value) - observed) <= tolerance
    except ValueError:
        return False


def integer_matches(value: str, observed: int) -> bool:
    try:
        return int(value) == observed
    except ValueError:
        return False


def lipinski_fail_fields(mw: float, logp: float, hbd: int, hba: int) -> list[str]:
    failed: list[str] = []
    if mw > FIXED_LIPINSKI["mw_max"]:
        failed.append("mw")
    if logp > FIXED_LIPINSKI["logp_max"]:
        failed.append("logp")
    if hbd > FIXED_LIPINSKI["hbd_max"]:
        failed.append("hbd")
    if hba > FIXED_LIPINSKI["hba_max"]:
        failed.append("hba")
    return failed


def stage_input_snapshot(source: Path, destination: Path, label: str) -> str:
    source_before = sha256_file(source)
    shutil.copyfile(source, destination)
    snapshot_hash = sha256_file(destination)
    source_after = sha256_file(source)
    if source_before != source_after or source_before != snapshot_hash:
        raise RuntimeError(f"{label} changed while being snapshotted; run was not started")
    return snapshot_hash


def run_primary_screen(
    *,
    root: Path,
    input_queue: Path,
    rule_config_path: Path,
    output_root: Path,
    run_id: str,
    limit: int | None,
) -> dict[str, Any]:
    Chem, Crippen, Descriptors, FilterCatalog, Lipinski = load_rdkit()
    root = root.resolve()
    input_queue = input_queue.resolve()
    rule_config_path = rule_config_path.resolve()
    output_root = output_root.resolve()
    if not input_queue.is_file():
        raise FileNotFoundError(f"input queue not found: {input_queue}")
    if not rule_config_path.is_file():
        raise FileNotFoundError(f"rule config not found: {rule_config_path}")
    if not run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be a single safe path name")

    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / run_id
    if final_dir.exists():
        raise RuntimeError(f"output already exists: {final_dir}")
    building_dir = output_root / f".{run_id}.building"
    if building_dir.exists():
        raise RuntimeError(f"unfinished build directory already exists: {building_dir}")
    building_dir.mkdir(parents=False, exist_ok=False)

    input_snapshot_path = building_dir / "input_queue_snapshot.tsv"
    config_snapshot_path = building_dir / "rule_config_snapshot.json"
    input_snapshot_hash = stage_input_snapshot(input_queue, input_snapshot_path, "input queue")
    config_snapshot_hash = stage_input_snapshot(rule_config_path, config_snapshot_path, "rule config")
    config = load_rule_config(config_snapshot_path)
    queries = build_queries(Chem, config)
    validation = validate_rules(Chem, queries)
    pains_catalog, pains_catalog_hash = build_pains_catalog(
        FilterCatalog, config["pains"]["expected_entry_count"]
    )
    tolerance = float(config["integrity_controls"]["descriptor_comparison_tolerance"])

    audit_path = building_dir / "p1_04_audit_all.tsv"
    candidate_path = building_dir / "strict_primary_candidates.tsv"
    summary_path = building_dir / "summary.json"
    manifest_path = building_dir / "manifest.json"
    validation_path = building_dir / "rule_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "created_at": utc_now(),
                "rdkit_version": Chem.rdBase.rdkitVersion,
                "all_cases_passed": True,
                "cases": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    counters: Counter[str] = Counter()
    michael_counts: Counter[str] = Counter()
    pains_descriptions: Counter[str] = Counter()
    fields = output_fieldnames()
    with (
        input_snapshot_path.open("r", encoding="utf-8", newline="") as input_handle,
        audit_path.open("w", encoding="utf-8", newline="") as audit_handle,
        candidate_path.open("w", encoding="utf-8", newline="") as candidate_handle,
    ):
        reader = csv.DictReader(input_handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError("input queue has no header")
        required_columns = {
            "representative_record_key",
            "canonical_smiles",
            "fragment_count",
            "calculation_status",
            "mw",
            "logp",
            "hbd",
            "hba",
            "lipinski_fail_fields",
        }
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise RuntimeError(f"input queue lacks required fields: {sorted(missing)}")
        audit_writer = csv.DictWriter(audit_handle, fieldnames=fields, delimiter="\t")
        candidate_writer = csv.DictWriter(candidate_handle, fieldnames=fields, delimiter="\t")
        audit_writer.writeheader()
        candidate_writer.writeheader()

        for row in reader:
            if limit is not None and counters["input_rows"] >= limit:
                break
            counters["input_rows"] += 1
            record = {name: row.get(name, "") for name in fields}
            record.update(
                {
                    "recomputed_mw": "",
                    "recomputed_logp": "",
                    "recomputed_hbd": "",
                    "recomputed_hba": "",
                    "recomputed_lipinski_fail_fields": "",
                    "input_integrity_status": "",
                    "input_integrity_notes": "",
                    "michael_status": "",
                    "michael_match_types": "",
                    "michael_rule_type_count": 0,
                    "pains_status": "",
                    "pains_description_count": 0,
                    "pains_descriptions": "",
                    "primary_filter_status": "",
                    "primary_filter_reasons": "",
                    "screening_note": "",
                }
            )
            try:
                fragment_count = int(row["fragment_count"])
            except ValueError:
                record.update(
                    {
                        "input_integrity_status": "inconsistent",
                        "input_integrity_notes": "invalid_fragment_count",
                        "michael_status": "not_evaluated_needs_review",
                        "pains_status": "not_evaluated_needs_review",
                        "primary_filter_status": "needs_review",
                        "primary_filter_reasons": "invalid_fragment_count",
                        "screening_note": "Input integrity check failed; never eligible for strict output.",
                    }
                )
                counters["needs_review_input_inconsistent"] += 1
                audit_writer.writerow(record)
                continue
            if fragment_count != 1:
                record.update(
                    {
                        "input_integrity_status": "verified_non_single_fragment",
                        "input_integrity_notes": "fragment_count_not_one",
                        "michael_status": "not_evaluated_needs_review",
                        "pains_status": "not_evaluated_needs_review",
                        "primary_filter_status": "needs_review",
                        "primary_filter_reasons": "multi_fragment",
                        "screening_note": "No salt or parent-fragment choice was made.",
                    }
                )
                counters["needs_review_multi_fragment"] += 1
                audit_writer.writerow(record)
                continue

            molecule = Chem.MolFromSmiles(row["canonical_smiles"], sanitize=True)
            if molecule is None:
                record.update(
                    {
                        "input_integrity_status": "inconsistent",
                        "input_integrity_notes": "reparse_failed",
                        "michael_status": "not_evaluated_needs_review",
                        "pains_status": "not_evaluated_needs_review",
                        "primary_filter_status": "needs_review",
                        "primary_filter_reasons": "reparse_failed",
                        "screening_note": "Canonical SMILES could not be reparsed during independent P1-04 check.",
                    }
                )
                counters["needs_review_reparse_failed"] += 1
                audit_writer.writerow(record)
                continue

            mw = float(Descriptors.MolWt(molecule))
            logp = float(Crippen.MolLogP(molecule))
            hbd = int(Lipinski.NumHDonors(molecule))
            hba = int(Lipinski.NumHAcceptors(molecule))
            actual_failed = lipinski_fail_fields(mw, logp, hbd, hba)
            expected_source_status = (
                "provisional_lipinski_fail" if actual_failed else "ready_for_next_review"
            )
            source_failed = [value for value in row["lipinski_fail_fields"].split(",") if value]
            integrity_notes: list[str] = []
            if row["calculation_status"] != expected_source_status:
                integrity_notes.append("source_status_mismatch")
            if sorted(source_failed) != sorted(actual_failed):
                integrity_notes.append("source_lipinski_fields_mismatch")
            if not number_matches(row["mw"], mw, tolerance):
                integrity_notes.append("source_mw_mismatch")
            if not number_matches(row["logp"], logp, tolerance):
                integrity_notes.append("source_logp_mismatch")
            if not integer_matches(row["hbd"], hbd):
                integrity_notes.append("source_hbd_mismatch")
            if not integer_matches(row["hba"], hba):
                integrity_notes.append("source_hba_mismatch")
            record.update(
                {
                    "recomputed_mw": f"{mw:.6f}",
                    "recomputed_logp": f"{logp:.6f}",
                    "recomputed_hbd": hbd,
                    "recomputed_hba": hba,
                    "recomputed_lipinski_fail_fields": ",".join(actual_failed),
                }
            )
            if integrity_notes:
                record.update(
                    {
                        "input_integrity_status": "inconsistent",
                        "input_integrity_notes": "|".join(integrity_notes),
                        "michael_status": "not_evaluated_needs_review",
                        "pains_status": "not_evaluated_needs_review",
                        "primary_filter_status": "needs_review",
                        "primary_filter_reasons": "input_integrity_mismatch",
                        "screening_note": "Input mismatch retained for review; never eligible for strict output.",
                    }
                )
                counters["needs_review_input_inconsistent"] += 1
                audit_writer.writerow(record)
                continue

            record["input_integrity_status"] = "verified"
            if actual_failed:
                record.update(
                    {
                        "michael_status": "not_evaluated_lipinski_fail",
                        "pains_status": "not_evaluated_lipinski_fail",
                        "primary_filter_status": "fail_lipinski",
                        "primary_filter_reasons": ",".join(actual_failed),
                        "screening_note": "Independent fixed-threshold Lipinski failure retained in audit.",
                    }
                )
                counters["fail_lipinski"] += 1
                audit_writer.writerow(record)
                continue

            counters["lipinski_pass_evaluated"] += 1
            matched_rules = [
                name for name, query in queries.items() if molecule.HasSubstructMatch(query)
            ]
            record["michael_match_types"] = "|".join(matched_rules)
            record["michael_rule_type_count"] = len(matched_rules)
            if not matched_rules:
                record.update(
                    {
                        "michael_status": "no_match",
                        "pains_status": "not_evaluated_not_michael",
                        "primary_filter_status": "fail_michael",
                        "primary_filter_reasons": "no_strict_michael_rule_match",
                        "screening_note": "No strict Michael SMARTS matched.",
                    }
                )
                counters["fail_michael"] += 1
                audit_writer.writerow(record)
                continue

            counters["michael_positive"] += 1
            for matched_rule in matched_rules:
                michael_counts[matched_rule] += 1
            pains_matches = pains_catalog.GetMatches(molecule)
            if pains_matches:
                descriptions = sorted({entry.GetDescription() for entry in pains_matches})
                record.update(
                    {
                        "michael_status": "matched",
                        "pains_status": "matched",
                        "pains_description_count": len(descriptions),
                        "pains_descriptions": "|".join(descriptions),
                        "primary_filter_status": "fail_pains",
                        "primary_filter_reasons": "PAINS",
                        "screening_note": "PAINS hit retained in audit but excluded from strict output.",
                    }
                )
                counters["fail_pains"] += 1
                for description in descriptions:
                    pains_descriptions[description] += 1
                audit_writer.writerow(record)
                continue

            record.update(
                {
                    "michael_status": "matched",
                    "pains_status": "no_match",
                    "primary_filter_status": "pass_primary_filter",
                    "primary_filter_reasons": "",
                    "screening_note": "Passes strict recorded P1-04 structural criteria.",
                }
            )
            counters["pass_primary_filter"] += 1
            audit_writer.writerow(record)
            candidate_writer.writerow(record)

    parent_summary_path = input_queue.parent / "summary.json"
    parent_summary_hash = sha256_file(parent_summary_path) if parent_summary_path.is_file() else None
    formal_complete = limit is None and counters["needs_review_input_inconsistent"] == 0
    summary = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "P1-04 strict Michael + independently recomputed fixed Lipinski + RDKit PAINS screen",
        "formal_p1_04_complete_for_recorded_scope": formal_complete,
        "trial_limit": limit,
        "reversible": True,
        "original_data_modified": False,
        "input": {
            "source_queue_path": str(input_queue),
            "source_queue_sha256": input_snapshot_hash,
            "snapshot": input_snapshot_path.name,
            "snapshot_sha256": input_snapshot_hash,
            "parent_summary_path": str(parent_summary_path) if parent_summary_hash else None,
            "parent_summary_sha256": parent_summary_hash,
        },
        "rule_config": {
            "source_path": str(rule_config_path),
            "source_and_snapshot_sha256": config_snapshot_hash,
            "snapshot": config_snapshot_path.name,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "rdkit": Chem.rdBase.rdkitVersion,
            "pains_catalog_entry_count": pains_catalog.GetNumEntries(),
            "pains_catalog_description_sha256": pains_catalog_hash,
        },
        "counts": dict(counters),
        "michael_rule_counts": dict(michael_counts),
        "top_pains_descriptions": pains_descriptions.most_common(100),
        "outputs": {
            "input_queue_snapshot": input_snapshot_path.name,
            "audit_all": audit_path.name,
            "strict_primary_candidates": candidate_path.name,
            "rule_validation": validation_path.name,
            "rule_config_snapshot": config_snapshot_path.name,
            "summary": summary_path.name,
            "manifest": manifest_path.name,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in (
            input_snapshot_path,
            config_snapshot_path,
            audit_path,
            candidate_path,
            validation_path,
            summary_path,
        )
    }
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": utc_now(),
        "artifact_sha256": artifact_hashes,
        "reversible": True,
        "original_data_modified": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    building_dir.replace(final_dir)
    return {**summary, "output_directory": str(final_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run recorded P1-04 primary screen without modifying the source compound data."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-queue", type=Path)
    parser.add_argument("--rule-config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--limit",
        type=int,
        help="For a nonformal trial only: process at most this many input queue rows.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    input_queue = args.input_queue or default_input_queue(root)
    rule_config = args.rule_config or (
        root / "protocol/execution_parameters" / DEFAULT_RULE_CONFIG
    )
    output_root = args.output_root or (root / "data/processed/p1_04_runs")
    suffix = "_trial" if args.limit is not None else "_formal"
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + suffix
    result = run_primary_screen(
        root=root,
        input_queue=input_queue,
        rule_config_path=rule_config,
        output_root=output_root,
        run_id=run_id,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
