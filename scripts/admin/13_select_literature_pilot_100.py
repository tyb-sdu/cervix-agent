from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
LITERATURE_ROOT = PROJECT_ROOT / "data" / "literature"
MANIFEST = (
    PROJECT_ROOT
    / "manifests"
    / "literature_20260724_project_local"
    / "literature_files_20260724_project_local.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "literature" / "pilot_100"

QUOTAS = {
    "01_HPV16_E6": {"xml": 8, "pdf": 12},
    "02_IDO1": {"xml": 8, "pdf": 12},
    "03_天然产物数据库": {"xml": 6, "pdf": 9},
    "04_共价与天然产物方法": {"xml": 8, "pdf": 12},
    "05_TC-1模型": {"xml": 4, "pdf": 6},
    "06_宫颈癌与免疫": {"xml": 6, "pdf": 9},
}

FORCED_PDF_FAMILIES = {
    "PMID_20131845__NO_PMCID": "PAINS foundational paper",
    "PMID_33398154__PMC8316984": "Reactive-cysteine profiling reference",
    "PMID_41083141__PMC12781138": "Click-chemistry covalent-drug review",
}


def stable_key(value: str) -> str:
    return hashlib.sha256(
        ("CervixAgent-pilot-100-20260724:" + value).encode("utf-8")
    ).hexdigest()


def family_identifier(relative_path: str) -> str:
    return str(Path(relative_path).parent.as_posix())


def identifier_from_family(family: str, prefix: str) -> str:
    match = re.search(rf"(?:^|/){prefix}_(\d+)", family)
    return match.group(1) if match else ""


def choose_primary_pdf(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = []
    for row in rows:
        if row["extension"] != ".pdf":
            continue
        lowered = row["file_name"].lower()
        if any(
            marker in lowered
            for marker in (
                "supplement",
                "supporting",
                "_si_",
                "appendix",
            )
        ):
            continue
        if lowered.startswith("manual_fulltext"):
            priority = 0
        elif lowered == "article.pdf":
            priority = 1
        else:
            priority = 2
        candidates.append((priority, row["file_name"], row))
    return min(candidates, default=(None, None, None))[2]


def choose_primary_xml(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if row["extension"] == ".xml"
        and "supplement" not in row["file_name"].lower()
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            0 if row["file_name"].lower() == "article.xml" else 1,
            row["file_name"],
        ),
    )


def main() -> int:
    if not LITERATURE_ROOT.is_dir():
        raise SystemExit(f"Missing literature root: {LITERATURE_ROOT}")
    if not MANIFEST.is_file():
        raise SystemExit(f"Missing source manifest: {MANIFEST}")

    rows_by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["extension"] not in {".pdf", ".xml"}:
                continue
            rows_by_family[family_identifier(row["relative_path"])].append(row)

    families_by_topic: dict[str, list[dict[str, object]]] = defaultdict(list)
    for family, family_rows in rows_by_family.items():
        topic = family.split("/", 1)[0]
        if topic not in QUOTAS:
            continue
        xml = choose_primary_xml(family_rows)
        pdf = choose_primary_pdf(family_rows)
        if xml is None and pdf is None:
            continue
        families_by_topic[topic].append(
            {
                "family": family,
                "topic": topic,
                "xml": xml,
                "pdf": pdf,
                "manual_pdf": bool(
                    pdf and pdf["file_name"].lower().startswith("manual_fulltext")
                ),
            }
        )

    selected: list[dict[str, object]] = []
    selected_families: set[str] = set()

    def add(
        family_entry: dict[str, object],
        route: str,
        reason: str,
    ) -> None:
        family = str(family_entry["family"])
        if family in selected_families:
            return
        primary = family_entry[route]
        if not isinstance(primary, dict):
            return
        paired_route = "pdf" if route == "xml" else "xml"
        paired = family_entry[paired_route]
        selected_families.add(family)
        selected.append(
            {
                "family": family,
                "topic": family_entry["topic"],
                "route": route,
                "primary": primary,
                "paired": paired if isinstance(paired, dict) else None,
                "manual_pdf": bool(family_entry["manual_pdf"]),
                "reason": reason,
            }
        )

    for topic, quota in QUOTAS.items():
        topic_families = sorted(
            families_by_topic[topic],
            key=lambda item: stable_key(str(item["family"])),
        )

        forced_here = [
            item
            for item in topic_families
            if Path(str(item["family"])).name in FORCED_PDF_FAMILIES
        ]
        for item in forced_here:
            family_name = Path(str(item["family"])).name
            add(item, "pdf", FORCED_PDF_FAMILIES[family_name])

        already_xml = sum(
            1
            for item in selected
            if item["topic"] == topic and item["route"] == "xml"
        )
        for item in topic_families:
            if already_xml >= quota["xml"]:
                break
            before = len(selected)
            add(item, "xml", "Deterministic stratified XML sample")
            if len(selected) > before:
                already_xml += 1

        already_pdf = sum(
            1
            for item in selected
            if item["topic"] == topic and item["route"] == "pdf"
        )
        desired_manual = min(quota["pdf"], max(2, quota["pdf"] // 3))
        current_manual = sum(
            1
            for item in selected
            if item["topic"] == topic
            and item["route"] == "pdf"
            and item["manual_pdf"]
        )
        for item in topic_families:
            if already_pdf >= quota["pdf"] or current_manual >= desired_manual:
                break
            if not item["manual_pdf"]:
                continue
            before = len(selected)
            add(item, "pdf", "Deterministic manual-PDF layout sample")
            if len(selected) > before:
                already_pdf += 1
                current_manual += 1

        for item in topic_families:
            if already_pdf >= quota["pdf"]:
                break
            before = len(selected)
            add(item, "pdf", "Deterministic stratified PDF sample")
            if len(selected) > before:
                already_pdf += 1

        if already_xml != quota["xml"] or already_pdf != quota["pdf"]:
            raise RuntimeError(
                f"Quota failure for {topic}: xml={already_xml}, pdf={already_pdf}"
            )

    selected.sort(
        key=lambda item: (
            str(item["topic"]),
            0 if item["route"] == "xml" else 1,
            str(item["family"]),
        )
    )

    if len(selected) != 100 or len(selected_families) != 100:
        raise RuntimeError(
            f"Expected 100 unique families, got {len(selected)} rows and "
            f"{len(selected_families)} families"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []
    for index, item in enumerate(selected, start=1):
        primary = item["primary"]
        paired = item["paired"]
        assert isinstance(primary, dict)
        primary_absolute = LITERATURE_ROOT / primary["relative_path"]
        if not primary_absolute.is_file():
            raise RuntimeError(f"Selected source missing: {primary_absolute}")
        output_rows.append(
            {
                "selection_id": f"PILOT-{index:03d}",
                "document_family_id": item["family"],
                "topic_folder": item["topic"],
                "pmid": identifier_from_family(str(item["family"]), "PMID"),
                "pmcid": identifier_from_family(str(item["family"]), "PMC"),
                "primary_format": item["route"],
                "parser_route": (
                    "jats_xml" if item["route"] == "xml" else "docling_pdf"
                ),
                "primary_relative_path": primary["relative_path"],
                "primary_size_bytes": int(primary["size_bytes"]),
                "primary_sha256": primary["sha256"],
                "paired_reference_relative_path": (
                    paired["relative_path"] if isinstance(paired, dict) else ""
                ),
                "selection_reason": item["reason"],
                "parse_status": "pending",
            }
        )

    csv_path = OUTPUT_ROOT / "pilot_100_manifest.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    os.replace(temporary_csv, csv_path)

    route_counts = Counter(row["primary_format"] for row in output_rows)
    topic_counts = Counter(row["topic_folder"] for row in output_rows)
    manual_pdf_count = sum(
        1
        for item in selected
        if item["route"] == "pdf" and item["manual_pdf"]
    )
    paired_count = sum(
        1 for row in output_rows if row["paired_reference_relative_path"]
    )
    summary = {
        "schema_version": 1,
        "status": "ready_for_parsing",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "literature_root": str(LITERATURE_ROOT),
        "source_manifest": str(MANIFEST),
        "selection_count": len(output_rows),
        "unique_document_family_count": len(selected_families),
        "topic_counts": dict(sorted(topic_counts.items())),
        "primary_format_counts": dict(sorted(route_counts.items())),
        "manual_pdf_count": manual_pdf_count,
        "paired_reference_count": paired_count,
        "forced_reference_count": sum(
            1
            for item in selected
            if item["reason"] in FORCED_PDF_FAMILIES.values()
        ),
        "selection_policy": {
            "deterministic": True,
            "random_seed_equivalent": "CervixAgent-pilot-100-20260724",
            "one_primary_input_per_document_family": True,
            "supplementary_pdfs_excluded_as_primary": True,
            "source_files_copied": False,
            "source_files_modified": False,
        },
        "manifest_csv": csv_path.name,
        "manifest_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    summary_path = OUTPUT_ROOT / "pilot_100_summary.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_summary, summary_path)
    (OUTPUT_ROOT / "pilot_100_manifest.sha256").write_text(
        f"{summary['manifest_csv_sha256']}  {csv_path.name}\n",
        encoding="ascii",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
