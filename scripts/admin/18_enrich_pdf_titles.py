from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rapidfuzz.fuzz import ratio


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
PILOT_ROOT = PROJECT_ROOT / "data" / "processed" / "literature" / "pilot_100"
LITERATURE_ROOT = PROJECT_ROOT / "data" / "literature"

GENERIC_TITLES = {
    "<!-- image -->",
    "research article",
    "research-article",
    "article",
    "original article",
    "review article",
}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def usable_title(value: str) -> bool:
    normalized = normalize(value)
    return (
        len(normalized) >= 20
        and len(normalized) <= 500
        and normalized.lower() not in GENERIC_TITLES
        and "<!--" not in normalized
    )


def source_metadata_title(source_relative_path: str) -> str:
    metadata_path = LITERATURE_ROOT / source_relative_path
    metadata_path = metadata_path.parent / "metadata.json"
    if not metadata_path.is_file():
        return ""
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    title = payload.get("title", "")
    return normalize(title) if isinstance(title, str) else ""


def markdown_fallback(markdown: str) -> str:
    for raw_line in markdown.splitlines():
        candidate = raw_line.strip().lstrip("#").strip()
        candidate = re.sub(r"<[^>]+>", "", candidate).strip()
        if not usable_title(candidate):
            continue
        if candidate.lower() in {"abstract", "introduction", "keywords"}:
            continue
        return candidate
    return ""


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    latest_run_id = (PILOT_ROOT / "latest_full_run.txt").read_text(
        encoding="ascii"
    ).strip()
    run_root = PILOT_ROOT / "runs" / latest_run_id
    ledger = json.loads((run_root / "ledger.json").read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []

    for entry in ledger:
        output_path = PROJECT_ROOT / entry["output_relative_path"]
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        parsed_title = normalize(payload["document"].get("title", ""))
        metadata_title = source_metadata_title(entry["source_relative_path"])
        markdown_title = markdown_fallback(payload["document"].get("markdown", ""))

        if usable_title(metadata_title):
            canonical_title = metadata_title
            title_source = "source_metadata"
            confidence = "high"
        elif usable_title(parsed_title):
            canonical_title = parsed_title
            title_source = "parser"
            confidence = "medium"
        elif usable_title(markdown_title):
            canonical_title = markdown_title
            title_source = "markdown_fallback"
            confidence = "low"
        else:
            canonical_title = ""
            title_source = "unresolved"
            confidence = "none"

        parser_title_usable = usable_title(parsed_title)
        similarity = (
            round(ratio(parsed_title, metadata_title), 1)
            if parsed_title and metadata_title
            else None
        )
        results.append(
            {
                "selection_id": entry["selection_id"],
                "topic_folder": entry["topic_folder"],
                "primary_format": entry["primary_format"],
                "source_relative_path": entry["source_relative_path"],
                "parser_title": parsed_title,
                "parser_title_usable": parser_title_usable,
                "metadata_title": metadata_title,
                "markdown_fallback_title": markdown_title,
                "canonical_title": canonical_title,
                "title_source": title_source,
                "confidence": confidence,
                "parser_metadata_similarity": similarity,
                "needs_manual_title_check": title_source == "unresolved",
            }
        )

    result_by_id = {row["selection_id"]: row for row in results}
    csv_path = run_root / "title_enrichment.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    original_spot_path = run_root / "manual_spot_check_20.csv"
    with original_spot_path.open("r", encoding="utf-8-sig", newline="") as stream:
        spot_rows = list(csv.DictReader(stream))
    enriched_spot_rows = []
    for row in spot_rows:
        enrichment = result_by_id[row["selection_id"]]
        enriched_spot_rows.append(
            {
                **row,
                "canonical_title": enrichment["canonical_title"],
                "canonical_title_source": enrichment["title_source"],
                "auto_title_status": (
                    "needs_manual_check"
                    if enrichment["needs_manual_title_check"]
                    else "metadata_or_parser_resolved"
                ),
            }
        )
    spot_path = run_root / "manual_spot_check_20_enriched.csv"
    with spot_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(enriched_spot_rows[0]))
        writer.writeheader()
        writer.writerows(enriched_spot_rows)

    counts = Counter(row["title_source"] for row in results)
    invalid_parser_titles = [
        row for row in results if not row["parser_title_usable"]
    ]
    unresolved = [row for row in results if row["title_source"] == "unresolved"]
    summary = {
        "schema_version": 1,
        "run_id": latest_run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(results),
        "title_source_counts": dict(counts),
        "parser_title_invalid_count": len(invalid_parser_titles),
        "unresolved_canonical_title_count": len(unresolved),
        "manual_title_check_count": sum(
            row["needs_manual_title_check"] for row in results
        ),
        "title_enrichment_csv": csv_path.name,
        "enriched_manual_spot_check_csv": spot_path.name,
        "status": (
            "ready_for_manual_spot_check"
            if not unresolved
            else "review_required"
        ),
        "notes": [
            "Original parsed JSON files were not overwritten.",
            "Future RAG ingestion must use canonical_title from title_enrichment.csv.",
        ],
    }
    write_json_atomic(run_root / "title_enrichment_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "ready_for_manual_spot_check" else 3


if __name__ == "__main__":
    sys.exit(main())
