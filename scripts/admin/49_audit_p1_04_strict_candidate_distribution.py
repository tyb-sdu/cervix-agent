#!/usr/bin/env python3
"""Profile P1-04 strict candidates without changing or ranking them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_COLUMNS = {
    "representative_record_key",
    "source_name",
    "source_id",
    "canonical_smiles",
    "mw",
    "logp",
    "hbd",
    "hba",
    "michael_match_types",
    "michael_rule_type_count",
    "pains_status",
    "primary_filter_status",
}
DESCRIPTORS = ("mw", "logp", "hbd", "hba")
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_value(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def nearest_rank(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a quantile from an empty list")
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    rank = math.ceil(q * len(values))
    index = max(0, min(len(values) - 1, rank - 1))
    return values[index]


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_and_verify_formal_run(
    formal_run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    summary_path = formal_run_dir / "summary.json"
    manifest_path = formal_run_dir / "manifest.json"
    candidate_path = formal_run_dir / "strict_primary_candidates.tsv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if summary.get("formal_p1_04_complete_for_recorded_scope") is not True:
        raise ValueError("Source P1-04 run is not marked formally complete")
    if summary.get("original_data_modified") is not False:
        raise ValueError("Source P1-04 summary does not preserve original data")
    expected_hash = manifest.get("artifact_sha256", {}).get(
        candidate_path.name
    )
    if not expected_hash:
        raise ValueError("Candidate file is absent from the source manifest")
    if sha256(candidate_path) != expected_hash:
        raise ValueError("Candidate file hash does not match the source manifest")
    return summary, manifest, candidate_path


def profile_candidates(
    candidate_path: Path, expected_count: int
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    source_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    combination_counts: Counter[str] = Counter()
    rule_type_count_counts: Counter[int] = Counter()
    source_rule_counts: Counter[tuple[str, str]] = Counter()
    descriptor_values: dict[str, list[float]] = {
        name: [] for name in DESCRIPTORS
    }
    duplicate_record_keys = 0
    duplicate_canonical_smiles = 0
    seen_record_keys: set[str] = set()
    seen_canonical_smiles: set[str] = set()
    row_count = 0

    with candidate_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        missing = EXPECTED_COLUMNS - fields
        if missing:
            raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            if row["primary_filter_status"] != "pass_primary_filter":
                raise ValueError(
                    f"Unexpected primary status at line {row_number}: "
                    f"{row['primary_filter_status']}"
                )
            if row["pains_status"] != "no_match":
                raise ValueError(
                    f"Unexpected PAINS status at line {row_number}: "
                    f"{row['pains_status']}"
                )
            record_key = row["representative_record_key"]
            canonical_smiles = row["canonical_smiles"]
            if record_key in seen_record_keys:
                duplicate_record_keys += 1
            seen_record_keys.add(record_key)
            if canonical_smiles in seen_canonical_smiles:
                duplicate_canonical_smiles += 1
            seen_canonical_smiles.add(canonical_smiles)

            source = row["source_name"] or "(missing)"
            source_counts[source] += 1
            rules = tuple(
                sorted(
                    value
                    for value in row["michael_match_types"].split("|")
                    if value
                )
            )
            if not rules:
                raise ValueError(f"No Michael rule at line {row_number}")
            combination = "|".join(rules)
            combination_counts[combination] += 1
            for rule in rules:
                rule_counts[rule] += 1
                source_rule_counts[(source, rule)] += 1
            declared_rule_count = int(row["michael_rule_type_count"])
            if declared_rule_count != len(rules):
                raise ValueError(
                    f"Michael rule count mismatch at line {row_number}"
                )
            rule_type_count_counts[declared_rule_count] += 1
            for descriptor in DESCRIPTORS:
                descriptor_values[descriptor].append(float(row[descriptor]))

    if row_count != expected_count:
        raise ValueError(
            f"Candidate row count {row_count} != summary count {expected_count}"
        )
    if duplicate_record_keys or duplicate_canonical_smiles:
        raise ValueError(
            "Strict candidate table is not unique: "
            f"record_keys={duplicate_record_keys}, "
            f"canonical_smiles={duplicate_canonical_smiles}"
        )

    for values in descriptor_values.values():
        values.sort()
    profile = {
        "row_count": row_count,
        "source_counts": dict(sorted(source_counts.items())),
        "michael_rule_counts": dict(sorted(rule_counts.items())),
        "michael_rule_combination_counts": dict(
            sorted(combination_counts.items())
        ),
        "michael_rule_type_count_counts": {
            str(key): value
            for key, value in sorted(rule_type_count_counts.items())
        },
        "source_rule_counts": [
            {"source_name": source, "michael_rule": rule, "count": count}
            for (source, rule), count in sorted(source_rule_counts.items())
        ],
        "duplicate_representative_record_key_count": duplicate_record_keys,
        "duplicate_canonical_smiles_count": duplicate_canonical_smiles,
    }
    return profile, descriptor_values


def descriptor_rows(
    descriptor_values: dict[str, list[float]]
) -> list[dict[str, Any]]:
    rows = []
    for descriptor in DESCRIPTORS:
        values = descriptor_values[descriptor]
        for q in QUANTILES:
            rows.append(
                {
                    "descriptor": descriptor,
                    "quantile": f"{q:.2f}",
                    "value": f"{nearest_rank(values, q):.6f}",
                    "method": "nearest_rank",
                    "count": len(values),
                }
            )
    return rows


def build_summary(
    project_root: Path,
    formal_run_dir: Path,
    formal_summary: dict[str, Any],
    formal_manifest: dict[str, Any],
    candidate_path: Path,
    profile: dict[str, Any],
    descriptor_values: dict[str, list[float]],
) -> dict[str, Any]:
    expected_text = "5000-10000"
    actual = profile["row_count"]
    within_expected = 5000 <= actual <= 10000
    return {
        "schema_version": 1,
        "audit_type": "P1-04_strict_candidate_distribution_only",
        "created_at_utc": utc_now(),
        "scope": (
            "Descriptive audit only. No additional filter, score, rank, "
            "threshold, docking decision or protocol change is applied."
        ),
        "source_formal_run": {
            "run_id": formal_summary["run_id"],
            "path": str(formal_run_dir.relative_to(project_root)).replace(
                "\\", "/"
            ),
            "summary_sha256": sha256(formal_run_dir / "summary.json"),
            "manifest_sha256": sha256(formal_run_dir / "manifest.json"),
            "candidate_path": str(candidate_path.relative_to(project_root)).replace(
                "\\", "/"
            ),
            "candidate_sha256": sha256(candidate_path),
            "candidate_manifest_sha256": formal_manifest[
                "artifact_sha256"
            ][candidate_path.name],
            "formal_complete_for_recorded_scope": formal_summary[
                "formal_p1_04_complete_for_recorded_scope"
            ],
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "script": (
                "scripts/admin/"
                "49_audit_p1_04_strict_candidate_distribution.py"
            ),
            "git_commit": git_value(project_root, "rev-parse", "HEAD"),
        },
        "integrity": {
            "source_modified": False,
            "candidate_rows_expected": formal_summary["counts"][
                "pass_primary_filter"
            ],
            "candidate_rows_observed": actual,
            "unique_representative_record_keys": (
                profile["duplicate_representative_record_key_count"] == 0
            ),
            "unique_canonical_smiles": (
                profile["duplicate_canonical_smiles_count"] == 0
            ),
        },
        "profile": {
            key: value
            for key, value in profile.items()
            if key != "source_rule_counts"
        },
        "descriptor_quantiles": {
            descriptor: {
                f"{q:.2f}": nearest_rank(values, q)
                for q in QUANTILES
            }
            for descriptor, values in descriptor_values.items()
        },
        "locked_expected_primary_screen_count": expected_text,
        "observed_candidate_count": actual,
        "within_locked_expected_count_range": within_expected,
        "planning_gap": (
            None
            if within_expected
            else (
                "The recorded strict P1-04 rules yield more candidates than "
                "the proposal's expected range. This audit does not tighten "
                "rules or discard candidates; a documented reviewer-approved "
                "secondary prioritization step is required."
            )
        ),
        "safety_invariants": {
            "protocol_changed": False,
            "candidate_rows_removed": False,
            "candidate_scores_created": False,
            "candidate_ranking_created": False,
            "docking_started": False,
        },
    }


def write_output(
    output_dir: Path,
    summary: dict[str, Any],
    profile: dict[str, Any],
    descriptor_values: dict[str, list[float]],
    candidate_path: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    staging = output_dir.with_name(output_dir.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Refusing to reuse {staging}")
    staging.mkdir(parents=True)

    (staging / "candidate_distribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_tsv(
        staging / "representative_source_counts.tsv",
        (
            {
                "source_name": source,
                "candidate_count": count,
                "fraction": f"{count / profile['row_count']:.8f}",
            }
            for source, count in profile["source_counts"].items()
        ),
        ["source_name", "candidate_count", "fraction"],
    )
    write_tsv(
        staging / "michael_rule_counts.tsv",
        (
            {
                "michael_rule": rule,
                "candidate_count": count,
                "fraction": f"{count / profile['row_count']:.8f}",
                "note": "Counts overlap when a candidate matches multiple rules.",
            }
            for rule, count in profile["michael_rule_counts"].items()
        ),
        ["michael_rule", "candidate_count", "fraction", "note"],
    )
    write_tsv(
        staging / "michael_rule_combination_counts.tsv",
        (
            {
                "michael_rule_combination": combination,
                "candidate_count": count,
                "fraction": f"{count / profile['row_count']:.8f}",
            }
            for combination, count in profile[
                "michael_rule_combination_counts"
            ].items()
        ),
        ["michael_rule_combination", "candidate_count", "fraction"],
    )
    write_tsv(
        staging / "representative_source_by_michael_rule.tsv",
        profile["source_rule_counts"],
        ["source_name", "michael_rule", "count"],
    )
    write_tsv(
        staging / "descriptor_quantiles.tsv",
        descriptor_rows(descriptor_values),
        ["descriptor", "quantile", "value", "method", "count"],
    )
    (staging / "README.md").write_text(
        "\n".join(
            [
                "# P1-04 strict-candidate distribution audit",
                "",
                "This package profiles the already selected strict candidates.",
                "It does not add a filter, score, rank, or docking result.",
                "",
                "The `source_name` field is the representative source retained "
                "after upstream structure deduplication; it is not a complete "
                "cross-database provenance count.",
                "",
                "The original candidate table is referenced by SHA-256 and is "
                "not copied or modified.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    artifacts = sorted(path for path in staging.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "source_candidate_sha256_before": sha256(candidate_path),
        "artifacts": [
            {
                "path": str(path.relative_to(staging)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if sha256(candidate_path) != manifest["source_candidate_sha256_before"]:
        raise RuntimeError("Source candidate file changed during profiling")
    staging.replace(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--formal-run-id", required=True)
    parser.add_argument("--audit-run-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    for label, value in (
        ("formal-run-id", args.formal_run_id),
        ("audit-run-id", args.audit_run_id),
    ):
        if not RUN_ID_RE.fullmatch(value):
            raise ValueError(f"Invalid {label}: {value!r}")
    formal_run_dir = (
        project_root / "data/processed/p1_04_runs" / args.formal_run_id
    )
    formal_summary, formal_manifest, candidate_path = (
        load_and_verify_formal_run(formal_run_dir)
    )
    source_hash_before = sha256(candidate_path)
    profile, descriptor_values = profile_candidates(
        candidate_path,
        int(formal_summary["counts"]["pass_primary_filter"]),
    )
    if sha256(candidate_path) != source_hash_before:
        raise RuntimeError("Source candidate file changed during parsing")
    summary = build_summary(
        project_root,
        formal_run_dir,
        formal_summary,
        formal_manifest,
        candidate_path,
        profile,
        descriptor_values,
    )

    print(
        json.dumps(
            {
                "candidate_count": profile["row_count"],
                "within_locked_expected_count_range": summary[
                    "within_locked_expected_count_range"
                ],
                "source_counts": profile["source_counts"],
                "michael_rule_combination_counts": profile[
                    "michael_rule_combination_counts"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.validate_only:
        return 0

    tracked_diff = git_value(
        project_root, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_diff:
        raise RuntimeError(
            "Refusing formal audit while tracked files are dirty: "
            + tracked_diff
        )
    output_dir = (
        project_root
        / "data/processed/p1_04_candidate_audits"
        / args.audit_run_id
    )
    write_output(
        output_dir,
        summary,
        profile,
        descriptor_values,
        candidate_path,
    )
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
