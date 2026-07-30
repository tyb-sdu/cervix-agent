#!/usr/bin/env python3
"""Create a read-only P1-05 target-structure preflight audit.

This script deliberately does not prepare receptors, add hydrogens, change
residues, delete ligands/cofactors, or start docking.  It checks the target
residue labels locked in the original workflow against both PDB author
numbering and the mmCIF reference-sequence mapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import gemmi


TARGET_FILES = {
    "hpv16_e6": Path("structures/HPV_E6/4XR8.cif"),
    "ido1": Path("structures/IDO1/2D0T.cif"),
}

TARGET_REFERENCE_ACCESSIONS = {
    "hpv16_e6": "P03126",
    "ido1": "P14902",
}

AMENDMENT_PATH = Path(
    "protocol/amendments/20260724_coconut_lotus_only.json"
)

RESIDUE_LABEL_RE = re.compile(r"^([A-Z]{3})(-?\d+)([A-Z]?)$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
WATER_NAMES = {"HOH", "DOD", "WAT"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_cif_value(value: Any) -> Any:
    if value in (None, False, ".", "?"):
        return None
    return value


def normalize_sequence(value: Any) -> str | None:
    value = clean_cif_value(value)
    if value is None:
        return None
    return "".join(str(value).split()).upper()


def category_rows(block: gemmi.cif.Block, prefix: str) -> list[dict[str, Any]]:
    category = block.get_mmcif_category(prefix)
    if not category:
        return []
    length = len(next(iter(category.values())))
    return [
        {key: clean_cif_value(values[index]) for key, values in category.items()}
        for index in range(length)
    ]


def integer(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def floating(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def find_chain(model: gemmi.Model, chain_name: str) -> gemmi.Chain:
    for chain in model:
        if chain.name == chain_name:
            return chain
    raise ValueError(f"Chain {chain_name!r} is absent from the first model")


def find_polymer_residue(
    chain: gemmi.Chain, author_number: int, insertion_code: str | None = None
) -> gemmi.Residue | None:
    insertion_code = insertion_code or ""
    for residue in chain:
        if residue.het_flag != "A" or residue.seqid.num != author_number:
            continue
        found_code = str(residue.seqid.icode).strip()
        if found_code == insertion_code:
            return residue
    return None


def find_atom(residue: gemmi.Residue | None, atom_name: str) -> gemmi.Atom | None:
    if residue is None:
        return None
    for atom in residue:
        if atom.name.strip() == atom_name:
            return atom
    return None


def nearest_atom(
    atom: gemmi.Atom | None, labelled_atoms: Iterable[tuple[str, gemmi.Atom]]
) -> dict[str, Any] | None:
    if atom is None:
        return None
    distances = [
        (atom.pos.dist(other.pos), label) for label, other in labelled_atoms
    ]
    if not distances:
        return None
    distance, label = min(distances)
    return {"atom": label, "distance_angstrom": round(distance, 4)}


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


@dataclass(frozen=True)
class TargetInput:
    key: str
    pdb_id: str
    reactive_residues: tuple[str, ...]
    reference_accession: str
    path: Path


def load_locked_targets(project_root: Path) -> tuple[dict[str, Any], list[TargetInput]]:
    lock_path = project_root / "protocol/original/workflow.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("locked") is not True:
        raise ValueError("workflow.lock.json is not marked locked=true")

    phase_step = None
    for phase in lock.get("phases", []):
        for step in phase.get("steps", []):
            if step.get("id") == "P1-05":
                phase_step = step
                break
    if not phase_step or phase_step.get("human_gate") is not True:
        raise ValueError("P1-05 is missing or is not marked human_gate=true")

    locked = lock["fixed_parameters"]["targets"]
    if set(locked) != set(TARGET_FILES):
        raise ValueError(
            "Target keys differ from the implemented read-only audit mapping: "
            f"locked={sorted(locked)}, implemented={sorted(TARGET_FILES)}"
        )

    targets = []
    for key in sorted(locked):
        item = locked[key]
        pdb_id = str(item["pdb"]).upper()
        path = project_root / TARGET_FILES[key]
        if path.stem.upper() != pdb_id:
            raise ValueError(f"{key}: structure path does not match locked PDB {pdb_id}")
        if not path.is_file():
            raise FileNotFoundError(path)
        residues = tuple(str(value).upper() for value in item["reactive_residues"])
        for label in residues:
            if not RESIDUE_LABEL_RE.fullmatch(label):
                raise ValueError(f"{key}: unsupported residue label {label!r}")
        targets.append(
            TargetInput(
                key=key,
                pdb_id=pdb_id,
                reactive_residues=residues,
                reference_accession=TARGET_REFERENCE_ACCESSIONS[key],
                path=path,
            )
        )
    return lock, targets


def target_reference_context(
    block: gemmi.cif.Block, accession: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    refs = category_rows(block, "_struct_ref.")
    matches = [row for row in refs if row.get("pdbx_db_accession") == accession]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one _struct_ref row for {accession}; found {len(matches)}"
        )
    ref = matches[0]
    alignments = [
        row
        for row in category_rows(block, "_struct_ref_seq.")
        if row.get("ref_id") == ref.get("id")
        and row.get("pdbx_db_accession") == accession
    ]
    chains = sorted(
        {
            str(row["pdbx_strand_id"])
            for row in alignments
            if row.get("pdbx_strand_id")
        }
    )
    if not alignments or not chains:
        raise ValueError(f"No chain alignment rows found for {accession}")
    return ref, alignments, chains


def scheme_rows_for_chain(
    scheme_rows: list[dict[str, Any]], chain_name: str
) -> list[dict[str, Any]]:
    return [row for row in scheme_rows if row.get("pdb_strand_id") == chain_name]


def author_scheme_row(
    chain_rows: list[dict[str, Any]], position: int, insertion_code: str
) -> dict[str, Any] | None:
    matches = []
    for row in chain_rows:
        if integer(row.get("auth_seq_num")) != position:
            continue
        row_code = str(row.get("pdb_ins_code") or "").strip()
        if row_code == insertion_code:
            matches.append(row)
    if len(matches) > 1:
        raise ValueError(
            f"Multiple numbering rows for author residue {position}{insertion_code}"
        )
    return matches[0] if matches else None


def reference_scheme_row(
    chain_rows: list[dict[str, Any]],
    alignments: list[dict[str, Any]],
    chain_name: str,
    reference_position: int,
) -> dict[str, Any] | None:
    matching_alignments = []
    for row in alignments:
        if row.get("pdbx_strand_id") != chain_name:
            continue
        db_begin = integer(row.get("db_align_beg"))
        db_end = integer(row.get("db_align_end"))
        if db_begin is None or db_end is None:
            continue
        if db_begin <= reference_position <= db_end:
            matching_alignments.append(row)
    if len(matching_alignments) != 1:
        return None
    alignment = matching_alignments[0]
    db_begin = integer(alignment["db_align_beg"])
    seq_begin = integer(alignment["seq_align_beg"])
    if db_begin is None or seq_begin is None:
        return None
    entity_seq_id = seq_begin + (reference_position - db_begin)
    matches = [
        row for row in chain_rows if integer(row.get("seq_id")) == entity_seq_id
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple numbering rows for reference residue {reference_position}"
        )
    return matches[0] if matches else None


def residue_observation(
    chain: gemmi.Chain,
    scheme_row: dict[str, Any] | None,
    expected_name: str,
    numbering_scheme: str,
    requested_position: int,
) -> dict[str, Any]:
    if scheme_row is None:
        return {
            "numbering_scheme": numbering_scheme,
            "requested_position": requested_position,
            "scheme_row_present": False,
            "coordinate_present": False,
            "expected_residue_name": expected_name,
            "observed_residue_name": None,
            "residue_name_matches": False,
            "author_position": None,
            "entity_sequence_position": None,
            "insertion_code": None,
            "sg_atom_present": False,
        }

    author_position = integer(scheme_row.get("auth_seq_num"))
    insertion_code = str(scheme_row.get("pdb_ins_code") or "").strip()
    coordinate = (
        find_polymer_residue(chain, author_position, insertion_code)
        if author_position is not None
        else None
    )
    scheme_name = str(scheme_row.get("mon_id") or "").upper() or None
    coordinate_name = coordinate.name.upper() if coordinate is not None else None
    observed_name = coordinate_name or scheme_name
    sg = find_atom(coordinate, "SG")
    sg_details = (
        {
            "name": sg.name.strip(),
            "element": sg.element.name,
            "occupancy": round(float(sg.occ), 4),
            "b_factor": round(float(sg.b_iso), 4),
            "altloc": str(sg.altloc).strip() or None,
        }
        if sg is not None
        else None
    )
    return {
        "numbering_scheme": numbering_scheme,
        "requested_position": requested_position,
        "scheme_row_present": True,
        "coordinate_present": coordinate is not None,
        "expected_residue_name": expected_name,
        "observed_residue_name": observed_name,
        "residue_name_matches": observed_name == expected_name,
        "author_position": author_position,
        "entity_sequence_position": integer(scheme_row.get("seq_id")),
        "insertion_code": insertion_code or None,
        "sg_atom_present": sg is not None,
        "sg_atom": sg_details,
    }


def reference_position_for_scheme_row(
    scheme_row: dict[str, Any],
    alignments: list[dict[str, Any]],
    chain_name: str,
) -> int | None:
    seq_id = integer(scheme_row.get("seq_id"))
    if seq_id is None:
        return None
    for alignment in alignments:
        if alignment.get("pdbx_strand_id") != chain_name:
            continue
        seq_begin = integer(alignment.get("seq_align_beg"))
        seq_end = integer(alignment.get("seq_align_end"))
        db_begin = integer(alignment.get("db_align_beg"))
        if seq_begin is None or seq_end is None or db_begin is None:
            continue
        if seq_begin <= seq_id <= seq_end:
            return db_begin + (seq_id - seq_begin)
    return None


def relevant_atoms_by_chain(model: gemmi.Model) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for chain in model:
        metal_atoms: list[tuple[str, gemmi.Atom]] = []
        hetero_heavy_atoms: list[tuple[str, gemmi.Atom]] = []
        hetero_residues: set[str] = set()
        for residue in chain:
            if residue.het_flag != "H" or residue.name in WATER_NAMES:
                continue
            hetero_residues.add(residue.name)
            for atom in residue:
                label = (
                    f"{chain.name}:{residue.name}{residue.seqid.num}:"
                    f"{atom.name.strip()}"
                )
                if atom.element.name in {"Zn", "Fe"}:
                    metal_atoms.append((label, atom))
                if atom.element.name != "H":
                    hetero_heavy_atoms.append((label, atom))
        result[chain.name] = {
            "metal_atoms": metal_atoms,
            "hetero_heavy_atoms": hetero_heavy_atoms,
            "hetero_residue_names": sorted(hetero_residues),
        }
    return result


def audit_target(target: TargetInput, project_root: Path) -> dict[str, Any]:
    relative_path = target.path.relative_to(project_root)
    block = gemmi.cif.read_file(str(target.path)).sole_block()
    structure = gemmi.read_structure(str(target.path))
    if not structure:
        raise ValueError(f"{relative_path}: no models")
    model = structure[0]

    entity_rows = category_rows(block, "_entity.")
    entity_poly_rows = category_rows(block, "_entity_poly.")
    ref, alignments, target_chains = target_reference_context(
        block, target.reference_accession
    )
    scheme_rows = category_rows(block, "_pdbx_poly_seq_scheme.")
    sequence_differences = [
        row
        for row in category_rows(block, "_struct_ref_seq_dif.")
        if row.get("pdbx_seq_db_accession_code") == target.reference_accession
    ]
    chain_atoms = relevant_atoms_by_chain(model)
    entity_by_id = {row.get("id"): row for row in entity_rows}
    entity_poly_by_id = {
        row.get("entity_id"): row for row in entity_poly_rows
    }
    chain_inventory = []
    for chain_name in target_chains:
        chain = find_chain(model, chain_name)
        chain_scheme = scheme_rows_for_chain(scheme_rows, chain_name)
        entity_ids = sorted(
            {row.get("entity_id") for row in chain_scheme if row.get("entity_id")}
        )
        polymer_residues = [
            residue for residue in chain if residue.het_flag == "A"
        ]
        auth_positions = [residue.seqid.num for residue in polymer_residues]
        label_positions = [
            integer(row.get("seq_id"))
            for row in chain_scheme
            if integer(row.get("seq_id")) is not None
        ]
        chain_inventory.append(
            {
                "auth_chain_id": chain_name,
                "label_asym_ids": sorted(
                    {str(row.get("asym_id")) for row in chain_scheme if row.get("asym_id")}
                ),
                "entity_ids": entity_ids,
                "entity_descriptions": [
                    entity_by_id.get(entity_id, {}).get("pdbx_description")
                    for entity_id in entity_ids
                ],
                "polymer_type": [
                    entity_poly_by_id.get(entity_id, {}).get("type")
                    for entity_id in entity_ids
                ],
                "polymer_residue_count": len(polymer_residues),
                "author_position_range": (
                    [min(auth_positions), max(auth_positions)]
                    if auth_positions
                    else None
                ),
                "label_sequence_position_range": (
                    [min(label_positions), max(label_positions)]
                    if label_positions
                    else None
                ),
                "nonwater_hetero_residues": chain_atoms[chain_name][
                    "hetero_residue_names"
                ],
            }
        )

    residue_checks = []
    for locked_label in target.reactive_residues:
        match = RESIDUE_LABEL_RE.fullmatch(locked_label)
        assert match is not None
        expected_name, position_text, insertion_code = match.groups()
        position = int(position_text)
        chain_checks = []
        for chain_name in target_chains:
            chain = find_chain(model, chain_name)
            chain_scheme = scheme_rows_for_chain(scheme_rows, chain_name)
            author_row = author_scheme_row(
                chain_scheme, position, insertion_code
            )
            reference_row = reference_scheme_row(
                chain_scheme, alignments, chain_name, position
            )
            author_observation = residue_observation(
                chain, author_row, expected_name, "pdb_author", position
            )
            reference_observation = residue_observation(
                chain, reference_row, expected_name, "reference_database", position
            )
            for observation in (author_observation, reference_observation):
                author_position = observation["author_position"]
                coordinate = (
                    find_polymer_residue(
                        chain,
                        author_position,
                        observation.get("insertion_code"),
                    )
                    if author_position is not None
                    else None
                )
                sg = find_atom(coordinate, "SG")
                observation["nearest_same_chain_metal"] = nearest_atom(
                    sg, chain_atoms[chain_name]["metal_atoms"]
                )
                observation["nearest_same_chain_nonwater_hetero_heavy_atom"] = (
                    nearest_atom(sg, chain_atoms[chain_name]["hetero_heavy_atoms"])
                )
                observation["physical_site_key"] = (
                    f"{chain_name}:{author_position}"
                    f"{observation.get('insertion_code') or ''}"
                    if author_position is not None
                    else None
                )
            chain_checks.append(
                {
                    "chain": chain_name,
                    "pdb_author_interpretation": author_observation,
                    "reference_database_interpretation": reference_observation,
                }
            )
        residue_checks.append(
            {
                "locked_label": locked_label,
                "expected_residue_name": expected_name,
                "requested_position": position,
                "chain_checks": chain_checks,
                "all_chains_match_under_pdb_author_numbering": all(
                    row["pdb_author_interpretation"]["residue_name_matches"]
                    for row in chain_checks
                ),
                "all_chains_match_under_reference_database_numbering": all(
                    row["reference_database_interpretation"]["residue_name_matches"]
                    for row in chain_checks
                ),
            }
        )

    cross_numbering_aliases = []
    for left in residue_checks:
        for right in residue_checks:
            if left["locked_label"] >= right["locked_label"]:
                continue
            for chain_name in target_chains:
                left_chain = next(
                    row for row in left["chain_checks"] if row["chain"] == chain_name
                )
                right_chain = next(
                    row for row in right["chain_checks"] if row["chain"] == chain_name
                )
                combinations = (
                    (
                        "pdb_author",
                        left_chain["pdb_author_interpretation"],
                        "reference_database",
                        right_chain["reference_database_interpretation"],
                    ),
                    (
                        "reference_database",
                        left_chain["reference_database_interpretation"],
                        "pdb_author",
                        right_chain["pdb_author_interpretation"],
                    ),
                )
                for left_scheme, left_observation, right_scheme, right_observation in combinations:
                    same_site = (
                        left_observation["physical_site_key"] is not None
                        and left_observation["physical_site_key"]
                        == right_observation["physical_site_key"]
                    )
                    if (
                        same_site
                        and left_observation["residue_name_matches"]
                        and right_observation["residue_name_matches"]
                    ):
                        cross_numbering_aliases.append(
                            {
                                "chain": chain_name,
                                "left_locked_label": left["locked_label"],
                                "left_numbering_scheme": left_scheme,
                                "right_locked_label": right["locked_label"],
                                "right_numbering_scheme": right_scheme,
                                "physical_site_key": left_observation[
                                    "physical_site_key"
                                ],
                                "warning": (
                                    "Two locked labels can resolve to the same physical "
                                    "coordinate when different numbering schemes are used."
                                ),
                            }
                        )

    cysteines = []
    for chain_name in target_chains:
        chain = find_chain(model, chain_name)
        chain_scheme = scheme_rows_for_chain(scheme_rows, chain_name)
        scheme_by_author = {
            (
                integer(row.get("auth_seq_num")),
                str(row.get("pdb_ins_code") or "").strip(),
            ): row
            for row in chain_scheme
        }
        for residue in chain:
            if residue.het_flag != "A" or residue.name != "CYS":
                continue
            insertion_code = str(residue.seqid.icode).strip()
            scheme_row = scheme_by_author.get((residue.seqid.num, insertion_code))
            sg = find_atom(residue, "SG")
            cysteines.append(
                {
                    "chain": chain_name,
                    "author_position": residue.seqid.num,
                    "insertion_code": insertion_code or None,
                    "reference_accession": target.reference_accession,
                    "reference_position": (
                        reference_position_for_scheme_row(
                            scheme_row, alignments, chain_name
                        )
                        if scheme_row
                        else None
                    ),
                    "sg_atom_present": sg is not None,
                    "nearest_same_chain_metal": nearest_atom(
                        sg, chain_atoms[chain_name]["metal_atoms"]
                    ),
                    "nearest_same_chain_nonwater_hetero_heavy_atom": nearest_atom(
                        sg, chain_atoms[chain_name]["hetero_heavy_atoms"]
                    ),
                }
            )

    label_has_unambiguous_match = all(
        (
            check["all_chains_match_under_pdb_author_numbering"]
            ^ check["all_chains_match_under_reference_database_numbering"]
        )
        for check in residue_checks
    )
    no_cross_alias = not cross_numbering_aliases
    target_residue_gate_passes = label_has_unambiguous_match and no_cross_alias

    target_entity = entity_by_id.get(ref.get("entity_id"))
    target_polymer = entity_poly_by_id.get(ref.get("entity_id"))
    raw_sequence = (
        normalize_sequence(target_polymer.get("pdbx_seq_one_letter_code"))
        if target_polymer
        else None
    )
    canonical_sequence = (
        normalize_sequence(
            target_polymer.get("pdbx_seq_one_letter_code_can")
        )
        if target_polymer
        else None
    )
    database_status_rows = category_rows(block, "_pdbx_database_status.")
    revision_rows = category_rows(block, "_pdbx_audit_revision_history.")
    entry_rows = category_rows(block, "_entry.")
    blocking_issues: list[dict[str, Any]] = []
    if len(target_chains) > 1:
        blocking_issues.append(
            {
                "id": "CHAIN_SELECTION_PENDING",
                "severity": "human_gate",
                "detail": (
                    "The PDB ID contains multiple copies of the target entity; "
                    "the protocol does not lock a single chain."
                ),
            }
        )
    if target_entity and target_entity.get("pdbx_mutation"):
        blocking_issues.append(
            {
                "id": "ENGINEERED_CONSTRUCT_MUTATION_CONTEXT",
                "severity": "human_gate",
                "detail": (
                    f"Target entity annotation: {target_entity['pdbx_mutation']}"
                ),
            }
        )
    for check in residue_checks:
        for chain_check in check["chain_checks"]:
            for scheme_key in (
                "pdb_author_interpretation",
                "reference_database_interpretation",
            ):
                observation = chain_check[scheme_key]
                if not observation["scheme_row_present"]:
                    blocking_issues.append(
                        {
                            "id": "TARGET_RESIDUE_MAPPING_MISSING",
                            "severity": "hard",
                            "detail": (
                                f"{check['locked_label']} has no {scheme_key} "
                                f"mapping on chain {chain_check['chain']}."
                            ),
                        }
                    )
                elif not observation["coordinate_present"]:
                    blocking_issues.append(
                        {
                            "id": "TARGET_RESIDUE_COORDINATE_MISSING",
                            "severity": "hard",
                            "detail": (
                                f"{check['locked_label']} maps to an absent "
                                f"coordinate on chain {chain_check['chain']} "
                                f"under {scheme_key}."
                            ),
                        }
                    )
                elif not observation["residue_name_matches"]:
                    blocking_issues.append(
                        {
                            "id": "TARGET_RESIDUE_NAME_MISMATCH",
                            "severity": "hard",
                            "detail": (
                                f"{check['locked_label']} expects "
                                f"{observation['expected_residue_name']} but "
                                f"observes {observation['observed_residue_name']} "
                                f"on chain {chain_check['chain']} under "
                                f"{scheme_key}."
                            ),
                        }
                    )
    for alias in cross_numbering_aliases:
        blocking_issues.append(
            {
                "id": "CROSS_NUMBERING_ALIAS",
                "severity": "hard",
                "detail": (
                    f"{alias['left_locked_label']} and "
                    f"{alias['right_locked_label']} resolve to "
                    f"{alias['physical_site_key']}."
                ),
            }
        )
    return {
        "target_key": target.key,
        "pdb_id": target.pdb_id,
        "source_file": str(relative_path).replace("\\", "/"),
        "source_sha256_before": sha256_file(target.path),
        "source_sha256_after": sha256_file(target.path),
        "source_size_bytes": target.path.stat().st_size,
        "structure_name": structure.name,
        "entry_id": (
            entry_rows[0].get("id") if entry_rows else target.pdb_id
        ),
        "title": clean_cif_value(block.find_value("_struct.title")),
        "experimental_method": clean_cif_value(
            block.find_value("_exptl.method")
        ),
        "resolution_angstrom": floating(
            clean_cif_value(block.find_value("_refine.ls_d_res_high"))
        )
        or floating(structure.resolution),
        "r_work": floating(
            clean_cif_value(block.find_value("_refine.ls_R_factor_R_work"))
        ),
        "r_free": floating(
            clean_cif_value(block.find_value("_refine.ls_R_factor_R_free"))
        ),
        "database_status": database_status_rows,
        "revision_history": revision_rows,
        "model_count": len(structure),
        "audited_model_index": 0,
        "reference_database": ref.get("db_name"),
        "reference_accession": target.reference_accession,
        "reference_code": ref.get("db_code"),
        "target_entity_id": ref.get("entity_id"),
        "target_entity_description": (
            target_entity.get("pdbx_description") if target_entity else None
        ),
        "target_entity_mutation": (
            target_entity.get("pdbx_mutation") if target_entity else None
        ),
        "target_polymer_strands": (
            target_polymer.get("pdbx_strand_id") if target_polymer else None
        ),
        "target_sequence": {
            "raw_length": len(raw_sequence) if raw_sequence else None,
            "raw_sha256": sha256_text(raw_sequence) if raw_sequence else None,
            "canonical_length": (
                len(canonical_sequence) if canonical_sequence else None
            ),
            "canonical_sha256": (
                sha256_text(canonical_sequence)
                if canonical_sequence
                else None
            ),
        },
        "chain_inventory": chain_inventory,
        "target_chains_from_reference_alignment": target_chains,
        "candidate_target_chains": target_chains,
        "selected_chain": None,
        "chain_selection_status": "awaiting_human_gate",
        "chain_nonwater_hetero_residues": {
            chain: chain_atoms[chain]["hetero_residue_names"]
            for chain in target_chains
        },
        "reference_alignment_rows": alignments,
        "reference_sequence_differences": sequence_differences,
        "locked_reactive_residues": list(target.reactive_residues),
        "residue_checks": residue_checks,
        "cross_numbering_aliases": cross_numbering_aliases,
        "all_coordinate_cysteines_in_target_chains": cysteines,
        "blocking_issues": blocking_issues,
        "target_residue_gate_passes": target_residue_gate_passes,
        "target_residue_gate_status": (
            "pass" if target_residue_gate_passes else "blocked_pending_human_resolution"
        ),
    }


def build_audit(project_root: Path) -> dict[str, Any]:
    lock, targets = load_locked_targets(project_root)
    lock_path = project_root / "protocol/original/workflow.lock.json"
    target_results = [audit_target(target, project_root) for target in targets]
    amendment_path = project_root / AMENDMENT_PATH
    tracked_diff = git_value(
        project_root, "status", "--porcelain", "--untracked-files=no"
    )
    overall_gate_passes = all(
        target["target_residue_gate_passes"] for target in target_results
    )
    blocking_issues = [
        {
            "target_key": target["target_key"],
            **issue,
        }
        for target in target_results
        for issue in target["blocking_issues"]
    ]
    return {
        "schema_version": 1,
        "audit_type": "P1-05_read_only_target_structure_preflight",
        "generated_at_utc": utc_now(),
        "project_root": str(project_root),
        "workflow_lock": {
            "path": str(lock_path.relative_to(project_root)).replace("\\", "/"),
            "sha256": sha256_file(lock_path),
            "locked": lock.get("locked"),
            "protocol_changes_by_agent": lock.get("policy", {}).get(
                "protocol_changes_by_agent"
            ),
            "p1_05_human_gate": True,
        },
        "amendment": {
            "path": str(AMENDMENT_PATH).replace("\\", "/"),
            "exists": amendment_path.is_file(),
            "sha256": sha256_file(amendment_path)
            if amendment_path.is_file()
            else None,
        },
        "software": {
            "python": platform.python_version(),
            "gemmi": gemmi.__version__,
            "script": "scripts/admin/47_audit_p1_05_target_structures.py",
        },
        "git": {
            "commit": git_value(project_root, "rev-parse", "HEAD"),
            "branch": git_value(project_root, "branch", "--show-current"),
            "tracked_worktree_dirty": bool(tracked_diff),
            "tracked_diff_paths": tracked_diff.splitlines() if tracked_diff else [],
        },
        "safety_invariants": {
            "human_gate_required": True,
            "source_structures_modified": False,
            "hydrogens_added": False,
            "residues_changed": False,
            "atoms_or_ligands_deleted": False,
            "receptor_preparation_performed": False,
            "docking_started": False,
            "docking_authorized": False,
            "formal_p1_05_complete": False,
        },
        "targets": target_results,
        "blocking_issues": blocking_issues,
        "blocking_issue_counts": {
            issue_id: sum(
                1 for issue in blocking_issues if issue["id"] == issue_id
            )
            for issue_id in sorted({issue["id"] for issue in blocking_issues})
        },
        "overall_target_residue_gate_passes": overall_gate_passes,
        "overall_status": (
            "ready_for_human_gate_review"
            if overall_gate_passes
            else "blocked_pending_target_residue_resolution"
        ),
    }


def flattened_residue_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for target in audit["targets"]:
        for check in target["residue_checks"]:
            for chain_check in check["chain_checks"]:
                for field, scheme in (
                    ("pdb_author_interpretation", "pdb_author"),
                    ("reference_database_interpretation", "reference_database"),
                ):
                    observation = chain_check[field]
                    nearest_metal = observation.get("nearest_same_chain_metal") or {}
                    rows.append(
                        {
                            "target_key": target["target_key"],
                            "pdb_id": target["pdb_id"],
                            "reference_accession": target["reference_accession"],
                            "locked_label": check["locked_label"],
                            "chain": chain_check["chain"],
                            "numbering_scheme": scheme,
                            "expected_residue_name": observation[
                                "expected_residue_name"
                            ],
                            "observed_residue_name": observation[
                                "observed_residue_name"
                            ],
                            "residue_name_matches": observation[
                                "residue_name_matches"
                            ],
                            "author_position": observation["author_position"],
                            "entity_sequence_position": observation[
                                "entity_sequence_position"
                            ],
                            "coordinate_present": observation["coordinate_present"],
                            "sg_atom_present": observation["sg_atom_present"],
                            "nearest_metal_atom": nearest_metal.get("atom"),
                            "nearest_metal_distance_angstrom": nearest_metal.get(
                                "distance_angstrom"
                            ),
                            "physical_site_key": observation["physical_site_key"],
                        }
                    )
    return rows


def flattened_cysteine_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for target in audit["targets"]:
        for cysteine in target["all_coordinate_cysteines_in_target_chains"]:
            metal = cysteine.get("nearest_same_chain_metal") or {}
            hetero = (
                cysteine.get(
                    "nearest_same_chain_nonwater_hetero_heavy_atom"
                )
                or {}
            )
            rows.append(
                {
                    "target_key": target["target_key"],
                    "pdb_id": target["pdb_id"],
                    "reference_accession": target["reference_accession"],
                    "chain": cysteine["chain"],
                    "author_position": cysteine["author_position"],
                    "reference_position": cysteine["reference_position"],
                    "sg_atom_present": cysteine["sg_atom_present"],
                    "nearest_metal_atom": metal.get("atom"),
                    "nearest_metal_distance_angstrom": metal.get(
                        "distance_angstrom"
                    ),
                    "nearest_nonwater_hetero_atom": hetero.get("atom"),
                    "nearest_nonwater_hetero_distance_angstrom": hetero.get(
                        "distance_angstrom"
                    ),
                }
            )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def human_gate_markdown(audit: dict[str, Any], run_id: str) -> str:
    target_lines = []
    for target in audit["targets"]:
        target_lines.append(
            f"- `{target['pdb_id']}` / `{target['target_key']}`: "
            f"`{target['target_residue_gate_status']}`"
        )
        for check in target["residue_checks"]:
            author = check["all_chains_match_under_pdb_author_numbering"]
            reference = check[
                "all_chains_match_under_reference_database_numbering"
            ]
            target_lines.append(
                f"  - `{check['locked_label']}`: PDB author numbering match="
                f"`{str(author).lower()}`; {target['reference_accession']} "
                f"numbering match=`{str(reference).lower()}`."
            )
        for alias in target["cross_numbering_aliases"]:
            target_lines.append(
                "  - WARNING: "
                f"`{alias['left_locked_label']}` ({alias['left_numbering_scheme']}) "
                f"and `{alias['right_locked_label']}` "
                f"({alias['right_numbering_scheme']}) resolve to the same "
                f"`{alias['physical_site_key']}` coordinate."
            )

    return "\n".join(
        [
            "# P1-05 target-structure human gate",
            "",
            f"- Audit run: `{run_id}`",
            f"- Overall status: `{audit['overall_status']}`",
            "- Formal P1-05 completion: `false`",
            "- Receptor preparation performed: `false`",
            "- Docking started: `false`",
            "- Docking authorized: `false`",
            "",
            "## Read-only findings",
            "",
            "Blocking issue IDs: "
            + (
                ", ".join(
                    f"`{issue_id}`"
                    for issue_id in sorted(audit["blocking_issue_counts"])
                )
                if audit["blocking_issue_counts"]
                else "none"
            ),
            "",
            *target_lines,
            "",
            "## Required reviewer decisions before any receptor preparation",
            "",
            "- [ ] Confirm the intended numbering system for every locked reactive "
            "residue (PDB author numbering versus reference-sequence numbering).",
            "- [ ] Confirm whether the two HPV16 E6 labels are intended to identify "
            "two distinct physical residues; the audit reports any cross-numbering "
            "alias explicitly.",
            "- [ ] Resolve and document every expected/observed residue-name mismatch. "
            "The agent must not substitute another cysteine without an approved "
            "protocol amendment.",
            "- [ ] Approve the biological chain(s) used for each target.",
            "- [ ] Decide how the 4XR8 engineered E6 construct mutations are handled "
            "and how results are mapped back to the intended biological sequence.",
            "- [ ] Approve treatment of Zn ions, IDO1 heme, crystallographic ligands, "
            "waters, missing atoms/residues, protonation, tautomers and charge states.",
            "- [ ] Approve the legal open-source covalent-docking engine, covalent "
            "reaction mapping, grid center/dimensions and benchmark protocol.",
            "- [ ] Record reviewer identity, date, evidence and approval or rejection.",
            "",
            "## Sign-off",
            "",
            "- Reviewer: ____________________",
            "- Date: ________________________",
            "- Decision: approve / reject / revise",
            "- Approved residue mapping and chain(s): ______________________________",
            "- Evidence or amendment identifier: ___________________________________",
            "- Notes: ______________________________________________________________",
            "",
            "This checklist records a human gate; it does not amend "
            "`protocol/original/workflow.lock.json`.",
            "",
        ]
    )


def write_outputs(
    project_root: Path, output_dir: Path, run_id: str, audit: dict[str, Any]
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    staging_dir = output_dir.with_name(output_dir.name + ".building")
    if staging_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse an incomplete staging directory: {staging_dir}"
        )
    staging_dir.mkdir(parents=True)
    snapshots = staging_dir / "input_snapshots"
    snapshots.mkdir()

    for target in audit["targets"]:
        source = project_root / target["source_file"]
        destination = snapshots / source.name
        shutil.copy2(source, destination)
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"Snapshot checksum mismatch: {source}")

    audit_path = staging_dir / "target_structure_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_tsv(
        staging_dir / "target_residue_checks.tsv",
        flattened_residue_rows(audit),
    )
    write_tsv(
        staging_dir / "all_coordinate_cysteines.tsv",
        flattened_cysteine_rows(audit),
    )
    (staging_dir / "human_gate_checklist.md").write_text(
        human_gate_markdown(audit, run_id), encoding="utf-8"
    )
    (staging_dir / "README.md").write_text(
        "\n".join(
            [
                "# P1-05 read-only target-structure preflight",
                "",
                f"Run ID: `{run_id}`",
                "",
                "This package audits locked residue labels against mmCIF author "
                "numbering and reference-sequence mapping. It does not prepare "
                "receptors or authorize docking.",
                "",
                "Review `human_gate_checklist.md` and "
                "`target_structure_audit.json` before any P1-05 preparation.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    artifact_paths = sorted(
        path for path in staging_dir.rglob("*") if path.is_file()
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "artifacts": [
            {
                "path": str(path.relative_to(staging_dir)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for target in audit["targets"]:
        source = project_root / target["source_file"]
        if sha256_file(source) != target["source_sha256_before"]:
            raise RuntimeError(f"Source structure changed during audit: {source}")
    staging_dir.replace(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="CervixAgent project root (default: current directory)",
    )
    parser.add_argument(
        "--run-id",
        help="Auditable output ID; defaults to a UTC timestamp",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Parse and validate without writing an output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ_read_only_preflight"
    )
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"Invalid run ID: {run_id!r}")

    audit = build_audit(project_root)
    summary = {
        "overall_status": audit["overall_status"],
        "formal_p1_05_complete": audit["safety_invariants"][
            "formal_p1_05_complete"
        ],
        "targets": {
            target["target_key"]: target["target_residue_gate_status"]
            for target in audit["targets"]
        },
    }
    if args.validate_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if audit["git"]["tracked_worktree_dirty"]:
        raise RuntimeError(
            "Refusing formal output while tracked project files are dirty: "
            + "; ".join(audit["git"]["tracked_diff_paths"])
        )
    output_dir = (
        project_root
        / "data/processed/p1_05_target_structure_audits"
        / run_id
    )
    write_outputs(project_root, output_dir, run_id, audit)
    print(json.dumps({**summary, "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
