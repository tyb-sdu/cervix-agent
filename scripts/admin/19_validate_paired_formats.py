from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rapidfuzz.fuzz import ratio


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
PILOT_ROOT = PROJECT_ROOT / "data" / "processed" / "literature" / "pilot_100"
LITERATURE_ROOT = PROJECT_ROOT / "data" / "literature"
PARSER_SCRIPT = PROJECT_ROOT / "scripts" / "admin" / "14_parse_literature_pilot.py"


def load_parser_module() -> Any:
    spec = importlib.util.spec_from_file_location("pilot_parser", PARSER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load parser module: {PARSER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normal_tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", value)
    }


def text_from_payload(payload: dict[str, Any]) -> str:
    document = payload["document"]
    return document.get("full_text") or document.get("markdown") or ""


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    latest_run_id = (PILOT_ROOT / "latest_full_run.txt").read_text(
        encoding="ascii"
    ).strip()
    run_root = PILOT_ROOT / "runs" / latest_run_id
    parser_module = load_parser_module()

    with (PILOT_ROOT / "pilot_100_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        manifest_rows = list(csv.DictReader(stream))

    # One deterministic paired family from each topic: enough to test both
    # routes without silently turning the pilot into a second full parse.
    selected: list[dict[str, str]] = []
    for topic in sorted({row["topic_folder"] for row in manifest_rows}):
        candidates = sorted(
            (
                row
                for row in manifest_rows
                if row["topic_folder"] == topic
                and row["paired_reference_relative_path"]
            ),
            key=lambda row: row["selection_id"],
        )
        if candidates:
            selected.append(candidates[0])

    ledger = json.loads((run_root / "ledger.json").read_text(encoding="utf-8"))
    primary_by_id = {
        row["selection_id"]: row
        for row in ledger
        if row["status"] == "parsed"
    }
    title_rows: dict[str, dict[str, str]] = {}
    with (run_root / "title_enrichment.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        title_rows = {row["selection_id"]: row for row in csv.DictReader(stream)}

    checks: list[dict[str, Any]] = []
    for row in selected:
        selection_id = row["selection_id"]
        primary_ledger = primary_by_id[selection_id]
        primary_payload = json.loads(
            (PROJECT_ROOT / primary_ledger["output_relative_path"]).read_text(
                encoding="utf-8"
            )
        )
        paired_source = LITERATURE_ROOT / row["paired_reference_relative_path"]
        paired_format = paired_source.suffix.lower().lstrip(".")
        if paired_format == "xml":
            paired_payload = parser_module.parse_jats_xml(paired_source)
        elif paired_format == "pdf":
            paired_payload = parser_module.parse_pdf_docling(
                paired_source, parser_module.source_metadata_title(paired_source)
            )
        else:
            raise RuntimeError(f"Unsupported paired format: {paired_source}")

        primary_text = text_from_payload(primary_payload)
        paired_text = text_from_payload(paired_payload)
        primary_tokens = normal_tokens(primary_text)
        paired_tokens = normal_tokens(paired_text)
        union = primary_tokens | paired_tokens
        jaccard = (
            round(len(primary_tokens & paired_tokens) / len(union), 4)
            if union
            else 0.0
        )
        canonical_title = title_rows[selection_id]["canonical_title"]
        paired_title = paired_payload["document"].get("title", "")
        title_similarity = round(ratio(canonical_title, paired_title), 1)
        text_ratio = round(
            min(len(primary_text), len(paired_text))
            / max(len(primary_text), len(paired_text)),
            4,
        ) if primary_text and paired_text else 0.0
        passed = (
            len(paired_text) >= 1000
            and title_similarity >= 90.0
            and jaccard >= 0.02
        )
        checks.append(
            {
                "selection_id": selection_id,
                "topic_folder": row["topic_folder"],
                "primary_format": row["primary_format"],
                "paired_format": paired_format,
                "primary_source_relative_path": row["primary_relative_path"],
                "paired_source_relative_path": row["paired_reference_relative_path"],
                "paired_source_sha256": sha256_file(paired_source),
                "canonical_title": canonical_title,
                "paired_extracted_title": paired_title,
                "title_similarity_percent": title_similarity,
                "primary_text_characters": len(primary_text),
                "paired_text_characters": len(paired_text),
                "text_length_ratio": text_ratio,
                "token_jaccard": jaccard,
                "passed": passed,
            }
        )

    csv_path = run_root / "paired_format_validation_6.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    summary = {
        "schema_version": 1,
        "run_id": latest_run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "One paired document family per topic; source documents unchanged.",
        "check_count": len(checks),
        "pass_count": sum(item["passed"] for item in checks),
        "fail_count": sum(not item["passed"] for item in checks),
        "criteria": {
            "paired_text_minimum_characters": 1000,
            "title_similarity_minimum_percent": 90.0,
            "token_jaccard_minimum": 0.02,
        },
        "csv": csv_path.name,
        "status": "passed" if all(item["passed"] for item in checks) else "review_required",
        "note": "This is a parser-consistency check, not a claim that publisher and manuscript versions have identical scientific content.",
    }
    write_json_atomic(run_root / "paired_format_validation_6.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 3


if __name__ == "__main__":
    sys.exit(main())
