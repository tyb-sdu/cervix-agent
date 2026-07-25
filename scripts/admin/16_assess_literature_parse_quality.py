from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
PILOT_ROOT = PROJECT_ROOT / "data" / "processed" / "literature" / "pilot_100"

GENERIC_TITLES = {
    "<!-- image -->",
    "research article",
    "research-article",
    "article",
    "original article",
    "review article",
}


def title_is_generic(value: str) -> bool:
    normalized = " ".join(value.split()).strip().lower()
    return normalized in GENERIC_TITLES or "<!--" in normalized


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.mean(values), 3),
        "maximum": max(values),
    }


def main() -> int:
    latest_run_id = (PILOT_ROOT / "latest_full_run.txt").read_text(
        encoding="ascii"
    ).strip()
    run_root = PILOT_ROOT / "runs" / latest_run_id
    ledger = json.loads((run_root / "ledger.json").read_text(encoding="utf-8"))
    run_summary = json.loads(
        (run_root / "summary.json").read_text(encoding="utf-8")
    )

    issues: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    content_hashes: defaultdict[str, list[str]] = defaultdict(list)
    for entry in ledger:
        selection_id = entry["selection_id"]
        if entry["status"] != "parsed":
            issues.append(
                {
                    "selection_id": selection_id,
                    "severity": "error",
                    "issue": "parse_failed",
                    "detail": entry.get("error", ""),
                }
            )
            continue
        output_path = PROJECT_ROOT / entry["output_relative_path"]
        if not output_path.is_file():
            issues.append(
                {
                    "selection_id": selection_id,
                    "severity": "error",
                    "issue": "output_missing",
                    "detail": str(output_path),
                }
            )
            continue
        if sha256(output_path) != entry["output_sha256"]:
            issues.append(
                {
                    "selection_id": selection_id,
                    "severity": "error",
                    "issue": "output_hash_mismatch",
                    "detail": str(output_path),
                }
            )
            continue

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        title = payload["document"].get("title", "")
        full_text = (
            payload["document"].get("full_text")
            or payload["document"].get("markdown")
            or ""
        )
        full_text_hash = hashlib.sha256(
            full_text.encode("utf-8")
        ).hexdigest()
        content_hashes[full_text_hash].append(selection_id)
        metric_rows.append(
            {
                "selection_id": selection_id,
                "topic_folder": entry["topic_folder"],
                "primary_format": entry["primary_format"],
                "title": title,
                "title_characters": metrics.get("title_characters", 0),
                "full_text_characters": metrics.get(
                    "full_text_characters", 0
                ),
                "section_or_heading_count": metrics.get(
                    "section_count", metrics.get("heading_count", 0)
                ),
                "page_count": metrics.get("page_count"),
                "duration_seconds": entry["duration_seconds"],
                "source_relative_path": entry["source_relative_path"],
                "output_relative_path": entry["output_relative_path"],
            }
        )

        if not title.strip():
            issues.append(
                {
                    "selection_id": selection_id,
                    "severity": "warning",
                    "issue": "title_missing",
                    "detail": "",
                }
            )
        elif title_is_generic(title):
            issues.append(
                {
                    "selection_id": selection_id,
                    "severity": "warning",
                    "issue": "title_generic_placeholder",
                    "detail": title,
                }
            )
        if len(title) > 500:
            issues.append(
                {
                    "selection_id": selection_id,
                    "severity": "warning",
                    "issue": "title_suspiciously_long",
                    "detail": str(len(title)),
                }
            )
        if len(full_text) < 1000:
            issues.append(
                {
                    "selection_id": selection_id,
                    "severity": "warning",
                    "issue": "extracted_text_below_1000_characters",
                    "detail": str(len(full_text)),
                }
            )
        if metrics.get("section_count") == 0:
            issues.append(
                {
                    "selection_id": selection_id,
                    "severity": "warning",
                    "issue": "jats_body_sections_missing",
                    "detail": "",
                }
            )

    for digest, selection_ids in content_hashes.items():
        if len(selection_ids) > 1:
            issues.append(
                {
                    "selection_id": ",".join(selection_ids),
                    "severity": "warning",
                    "issue": "duplicate_extracted_content",
                    "detail": digest,
                }
            )

    format_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    topic_counts: Counter[str] = Counter()
    for row in metric_rows:
        format_groups[row["primary_format"]].append(row)
        topic_counts[row["topic_folder"]] += 1

    format_statistics = {}
    for file_format, rows in sorted(format_groups.items()):
        format_statistics[file_format] = {
            "documents": len(rows),
            "full_text_characters": distribution(
                [float(row["full_text_characters"]) for row in rows]
            ),
            "duration_seconds": distribution(
                [float(row["duration_seconds"]) for row in rows]
            ),
        }

    # Prepare a deterministic 20-document human spot-check queue: one XML and
    # one PDF per topic, then the slowest or shortest remaining PDFs.
    manual_check: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    topics = sorted(topic_counts)
    for topic in topics:
        for file_format in ("xml", "pdf"):
            candidates = sorted(
                (
                    row
                    for row in metric_rows
                    if row["topic_folder"] == topic
                    and row["primary_format"] == file_format
                ),
                key=lambda row: row["selection_id"],
            )
            if candidates:
                manual_check.append(candidates[0])
                chosen_ids.add(candidates[0]["selection_id"])

    remaining = sorted(
        (
            row
            for row in metric_rows
            if row["selection_id"] not in chosen_ids
        ),
        key=lambda row: (
            0 if row["primary_format"] == "pdf" else 1,
            -float(row["duration_seconds"]),
            int(row["full_text_characters"]),
            row["selection_id"],
        ),
    )
    for row in remaining:
        if len(manual_check) >= 20:
            break
        manual_check.append(row)
        chosen_ids.add(row["selection_id"])

    metric_path = run_root / "quality_metrics.csv"
    with metric_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    manual_path = run_root / "manual_spot_check_20.csv"
    manual_fields = list(manual_check[0]) + [
        "title_correct",
        "body_readable",
        "columns_in_order",
        "tables_usable",
        "citation_traceable",
        "review_notes",
    ]
    with manual_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manual_fields)
        writer.writeheader()
        for row in manual_check:
            writer.writerow(
                {
                    **row,
                    "title_correct": "",
                    "body_readable": "",
                    "columns_in_order": "",
                    "tables_usable": "",
                    "citation_traceable": "",
                    "review_notes": "",
                }
            )

    severity_counts = Counter(issue["severity"] for issue in issues)
    assessment = {
        "schema_version": 1,
        "assessment_id": f"{latest_run_id}_quality",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_summary": run_summary,
        "verified_output_count": len(metric_rows),
        "topic_counts": dict(sorted(topic_counts.items())),
        "format_statistics": format_statistics,
        "issue_count": len(issues),
        "severity_counts": dict(severity_counts),
        "issues": issues,
        "manual_spot_check_count": len(manual_check),
        "manual_spot_check_csv": manual_path.name,
        "quality_metrics_csv": metric_path.name,
        "status": (
            "ready_for_manual_spot_check"
            if run_summary["status"] == "completed"
            and len(metric_rows) == 100
            and severity_counts.get("error", 0) == 0
            else "review_required"
        ),
    }
    output_path = run_root / "quality_assessment.json"
    write_json_atomic(output_path, assessment)
    (run_root / "quality_assessment.sha256").write_text(
        f"{sha256(output_path)}  {output_path.name}\n",
        encoding="ascii",
    )
    print(json.dumps(assessment, ensure_ascii=False, indent=2))
    return 0 if assessment["status"] == "ready_for_manual_spot_check" else 3


if __name__ == "__main__":
    sys.exit(main())
