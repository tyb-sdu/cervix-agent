from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
BASE = PROJECT_ROOT / "data" / "processed" / "literature" / "manual_merge_20260726"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def candidate_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines()[:80]:
        line = compact(re.sub(r"<[^>]+>", "", line).lstrip("#").strip())
        if 20 <= len(line) <= 400 and line.lower() not in {"abstract", "introduction", "keywords"}:
            return line
    return fallback


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    preflight_id = (BASE / "latest_preflight_run.txt").read_text(encoding="ascii").strip()
    preflight_root = BASE / "runs" / preflight_id
    preflight_ledger = json.loads((preflight_root / "ledger.json").read_text(encoding="utf-8"))
    candidates = [
        item for item in preflight_ledger
        if item["status"] == "parsed"
        and item["extension"] == ".pdf"
        and item["role"] == "fulltext"
        and item["metrics"].get("full_text_characters", 0) >= 1000
    ]
    if not candidates:
        raise RuntimeError("No eligible full-text PDFs from the preflight run")

    run_id = "docling_fulltext_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = BASE / "runs" / run_id
    output_root = run_root / "documents"
    output_root.mkdir(parents=True, exist_ok=False)
    converter = DocumentConverter()
    ledger: list[dict[str, Any]] = []

    for index, item in enumerate(candidates, start=1):
        source = PROJECT_ROOT / item["source_relative_path"]
        started = time.perf_counter()
        result: dict[str, Any] = {
            "item_id": item["item_id"],
            "record": item["record"],
            "category": item["category"],
            "role": item["role"],
            "source_relative_path": item["source_relative_path"],
            "source_sha256": item["source_sha256"],
            "status": "failed",
        }
        try:
            if sha256_file(source) != item["source_sha256"]:
                raise RuntimeError("Source SHA-256 differs from preflight ledger")
            converted = converter.convert(source)
            markdown = converted.document.export_to_markdown().strip()
            pages = getattr(converted.document, "pages", None)
            page_count = len(pages) if pages is not None else None
            heading_count = sum(1 for line in markdown.splitlines() if line.lstrip().startswith("#"))
            title = candidate_title(markdown, item["record"])
            warnings = []
            if len(markdown) < 1000:
                warnings.append("docling_text_below_1000_characters")
            if title == item["record"]:
                warnings.append("docling_title_fallback_to_record")
            payload = {
                "schema_version": 1,
                "source": {
                    "relative_path": item["source_relative_path"],
                    "sha256": item["source_sha256"],
                    "source_modified": False,
                },
                "document": {"title": title, "markdown": markdown},
                "metrics": {
                    "full_text_characters": len(markdown),
                    "page_count": page_count,
                    "heading_count": heading_count,
                    "table_marker_count": markdown.count("<table"),
                    "text_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
                },
                "parser": {"route": "docling", "preflight_run": preflight_id},
                "warnings": warnings,
            }
            output_path = output_root / f"{item['item_id']}.json"
            write_json(output_path, payload)
            result.update(
                {
                    "status": "parsed",
                    "warnings": warnings,
                    "metrics": payload["metrics"],
                    "output_relative_path": str(output_path.relative_to(PROJECT_ROOT)),
                    "output_sha256": sha256_file(output_path),
                }
            )
        except Exception as exc:
            result["error"] = repr(exc)
            result["traceback"] = traceback.format_exc()
        result["duration_seconds"] = round(time.perf_counter() - started, 3)
        ledger.append(result)
        print(f"{index}/{len(candidates)} {result['item_id']} {result['status']} {result['duration_seconds']}s", flush=True)

    status_counts = Counter(item["status"] for item in ledger)
    warning_counts = Counter(warning for item in ledger for warning in item.get("warnings", []))
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "preflight_run": preflight_id,
        "selected_fulltext_pdf_count": len(candidates),
        "status_counts": dict(status_counts),
        "warning_counts": dict(warning_counts),
        "source_files_modified": False,
        "indexing_status": "not_indexed",
    }
    write_json(run_root / "ledger.json", ledger)
    write_json(run_root / "summary.json", summary)
    (BASE / "latest_docling_fulltext_run.txt").write_text(run_id + "\n", encoding="ascii")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status_counts.get("failed", 0) == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
