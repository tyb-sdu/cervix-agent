from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PREFILTER_RUN = "20260729T041834Z_lossless_prefilter_shadow"
DEFAULT_RULE_CONFIG = "20260729_p1_04_conservative_michael_pains.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rdkit() -> tuple[Any, Any]:
    from rdkit import Chem
    from rdkit.Chem import FilterCatalog

    return Chem, FilterCatalog


def load_rule_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported rule config schema: {path}")
    rules = payload.get("michael_acceptor", {}).get("rules", [])
    if len(rules) != 4:
        raise RuntimeError("rule config must contain exactly four Michael rule records")
    expected_ids = {
        "alpha_beta_unsaturated_carbonyl",
        "conjugated_lactone",
        "nitroalkene",
        "quinone",
    }
    actual_ids = {rule.get("id") for rule in rules}
    if actual_ids != expected_ids:
        raise RuntimeError(f"unexpected Michael rule IDs: {sorted(actual_ids)}")
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
    """Known chemical sanity cases for a conservative structural screen."""
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
                "expected": sorted(expected),
                "observed": sorted(observed),
                "passed": passed,
            }
        )
        if not passed:
            failures.append(label)
    if failures:
        raise RuntimeError(f"SMARTS validation failed for: {', '.join(failures)}")
    return results


def build_pains_catalog(FilterCatalog: Any, expected_entry_count: int) -> Any:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    observed_entry_count = catalog.GetNumEntries()
    if observed_entry_count != expected_entry_count:
        raise RuntimeError(
            "RDKit PAINS catalog size differs from recorded config: "
            f"expected {expected_entry_count}, observed {observed_entry_count}"
        )
    return catalog


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
        "michael_status",
        "michael_match_types",
        "michael_match_count",
        "pains_status",
        "pains_match_count",
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


def run_primary_screen(
    *,
    root: Path,
    input_queue: Path,
    rule_config_path: Path,
    output_root: Path,
    run_id: str,
    limit: int | None,
) -> dict[str, Any]:
    Chem, FilterCatalog = load_rdkit()
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

    config = load_rule_config(rule_config_path)
    queries = build_queries(Chem, config)
    validation = validate_rules(Chem, queries)
    pains_catalog = build_pains_catalog(FilterCatalog, config["pains"]["expected_entry_count"])

    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / run_id
    if final_dir.exists():
        raise RuntimeError(f"output already exists: {final_dir}")
    building_dir = output_root / f".{run_id}.building"
    if building_dir.exists():
        raise RuntimeError(f"unfinished build directory already exists: {building_dir}")
    building_dir.mkdir(parents=False, exist_ok=False)

    audit_path = building_dir / "p1_04_audit_all.tsv"
    candidate_path = building_dir / "strict_primary_candidates.tsv"
    summary_path = building_dir / "summary.json"
    manifest_path = building_dir / "manifest.json"
    validation_path = building_dir / "rule_validation.json"
    config_snapshot_path = building_dir / "rule_config_snapshot.json"
    shutil.copyfile(rule_config_path, config_snapshot_path)
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
        input_queue.open("r", encoding="utf-8", newline="") as input_handle,
        audit_path.open("w", encoding="utf-8", newline="") as audit_handle,
        candidate_path.open("w", encoding="utf-8", newline="") as candidate_handle,
    ):
        reader = csv.DictReader(input_handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError("input queue has no header")
        missing = {"representative_record_key", "canonical_smiles", "calculation_status"} - set(
            reader.fieldnames
        )
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
            source_status = row.get("calculation_status", "")
            record["michael_match_types"] = ""
            record["michael_match_count"] = 0
            record["pains_match_count"] = 0
            record["pains_descriptions"] = ""

            if source_status == "provisional_lipinski_fail":
                record.update(
                    {
                        "michael_status": "not_evaluated_lipinski_fail",
                        "pains_status": "not_evaluated_lipinski_fail",
                        "primary_filter_status": "fail_lipinski",
                        "primary_filter_reasons": row.get("lipinski_fail_fields", "") or "lipinski",
                        "screening_note": "Fixed-threshold Lipinski failure retained in audit.",
                    }
                )
                counters["fail_lipinski"] += 1
                audit_writer.writerow(record)
                continue

            if source_status != "ready_for_next_review":
                record.update(
                    {
                        "michael_status": "not_evaluated_needs_review",
                        "pains_status": "not_evaluated_needs_review",
                        "primary_filter_status": "needs_review",
                        "primary_filter_reasons": source_status or "nonstandard_input_status",
                        "screening_note": "No salt/fragment or reparsing policy was applied.",
                    }
                )
                counters["needs_review"] += 1
                audit_writer.writerow(record)
                continue

            molecule = Chem.MolFromSmiles(row.get("canonical_smiles", ""), sanitize=True)
            if molecule is None:
                record.update(
                    {
                        "michael_status": "not_evaluated_reparse_failed",
                        "pains_status": "not_evaluated_reparse_failed",
                        "primary_filter_status": "needs_review",
                        "primary_filter_reasons": "reparse_failed",
                        "screening_note": "Canonical SMILES could not be reparsed during P1-04.",
                    }
                )
                counters["needs_review_reparse_failed"] += 1
                audit_writer.writerow(record)
                continue

            counters["lipinski_pass_evaluated"] += 1
            matched_rules = [
                name for name, query in queries.items() if molecule.HasSubstructMatch(query)
            ]
            record["michael_match_types"] = "|".join(matched_rules)
            record["michael_match_count"] = len(matched_rules)
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
                        "pains_match_count": len(descriptions),
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

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": utc_now(),
        "scope": "P1-04 strict Michael + fixed Lipinski + RDKit PAINS screen",
        "formal_p1_04_complete_for_recorded_scope": limit is None,
        "trial_limit": limit,
        "reversible": True,
        "original_data_modified": False,
        "input": {
            "queue_path": str(input_queue),
            "queue_sha256": sha256_file(input_queue),
        },
        "rule_config": {
            "path": str(rule_config_path),
            "sha256": sha256_file(rule_config_path),
        },
        "software": {
            "rdkit": Chem.rdBase.rdkitVersion,
            "pains_catalog_entry_count": pains_catalog.GetNumEntries(),
        },
        "counts": dict(counters),
        "michael_rule_counts": dict(michael_counts),
        "top_pains_descriptions": pains_descriptions.most_common(100),
        "outputs": {
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
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": utc_now(),
        "input_queue_sha256": sha256_file(input_queue),
        "rule_config_sha256": sha256_file(rule_config_path),
        "audit_all_sha256": sha256_file(audit_path),
        "strict_primary_candidates_sha256": sha256_file(candidate_path),
        "rule_validation_sha256": sha256_file(validation_path),
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
        description="Run recorded P1-04 Michael/Lipinski/PAINS primary screen without modifying input data."
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
