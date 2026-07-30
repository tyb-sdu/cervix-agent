#!/usr/bin/env python3
"""Verify the integrity and safety invariants of a P1-05 audit package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    audit = load_json(run_dir / "target_structure_audit.json")
    manifest = load_json(run_dir / "manifest.json")

    expected_artifacts = {
        item["path"] for item in manifest.get("artifacts", [])
    }
    actual_artifacts = {
        str(path.relative_to(run_dir)).replace("\\", "/")
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if expected_artifacts != actual_artifacts:
        raise AssertionError(
            f"manifest artifact set mismatch: expected={sorted(expected_artifacts)}, "
            f"actual={sorted(actual_artifacts)}"
        )
    for item in manifest["artifacts"]:
        path = run_dir / item["path"]
        if path.stat().st_size != item["size_bytes"]:
            raise AssertionError(f"size mismatch: {path}")
        if sha256(path) != item["sha256"]:
            raise AssertionError(f"hash mismatch: {path}")

    lock_rel = audit["workflow_lock"]["path"]
    lock_path = project_root / lock_rel
    if sha256(lock_path) != audit["workflow_lock"]["sha256"]:
        raise AssertionError("workflow.lock hash changed")
    amendment = audit["amendment"]
    amendment_path = project_root / amendment["path"]
    if amendment["exists"] and sha256(amendment_path) != amendment["sha256"]:
        raise AssertionError("amendment hash changed")

    lock = load_json(lock_path)
    locked_targets = lock["fixed_parameters"]["targets"]
    observation_count = 0
    scheme_observation_nonmatch_count = 0
    missing_count = 0
    ambiguous_count = 0
    for target in audit["targets"]:
        key = target["target_key"]
        locked = locked_targets[key]
        if target["pdb_id"] != str(locked["pdb"]).upper():
            raise AssertionError(f"PDB mismatch for {key}")
        if target["locked_reactive_residues"] != [
            str(value).upper() for value in locked["reactive_residues"]
        ]:
            raise AssertionError(f"residue-label mismatch for {key}")
        source = project_root / target["source_file"]
        snapshot = run_dir / "input_snapshots" / source.name
        if sha256(source) != target["source_sha256_before"]:
            raise AssertionError(f"source changed: {source}")
        if sha256(source) != target["source_sha256_after"]:
            raise AssertionError(f"source before/after mismatch: {source}")
        if sha256(snapshot) != target["source_sha256_before"]:
            raise AssertionError(f"snapshot mismatch: {snapshot}")
        for check in target["residue_checks"]:
            for chain_check in check["chain_checks"]:
                for scheme in (
                    "pdb_author_interpretation",
                    "reference_database_interpretation",
                ):
                    observation_count += 1
                    observation = chain_check[scheme]
                    if not observation["scheme_row_present"] or not observation[
                        "coordinate_present"
                    ]:
                        missing_count += 1
                    elif not observation["residue_name_matches"]:
                        scheme_observation_nonmatch_count += 1
                    if observation.get("physical_site_key") is None:
                        ambiguous_count += 1

    safety = audit["safety_invariants"]
    for key in (
        "source_structures_modified",
        "hydrogens_added",
        "residues_changed",
        "atoms_or_ligands_deleted",
        "receptor_preparation_performed",
        "docking_started",
        "docking_authorized",
        "formal_p1_05_complete",
    ):
        if safety.get(key) is not False:
            raise AssertionError(f"safety invariant not false: {key}")
    if audit["overall_status"] != "blocked_pending_target_residue_resolution":
        raise AssertionError(
            "Unexpected overall status; this verifier is for the unsigned "
            "preflight package."
        )

    result = {
        "verification_status": "pass",
        "meaning": (
            "The audit package is internally consistent; this does not approve "
            "P1-05 or authorize docking."
        ),
        "run_dir": str(run_dir),
        "manifest_artifact_count": len(expected_artifacts),
        "observation_count": observation_count,
        "scheme_observation_nonmatch_count": (
            scheme_observation_nonmatch_count
        ),
        "scheme_observation_nonmatch_note": (
            "A nonmatch in one numbering interpretation is not automatically "
            "a residue-identity conflict; consult blocking_issue_counts."
        ),
        "missing_count": missing_count,
        "ambiguous_count": ambiguous_count,
        "blocking_issue_counts": audit["blocking_issue_counts"],
        "human_gate_required": True,
        "docking_authorized": False,
        "formal_p1_05_complete": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
