from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
BASE = PROJECT_ROOT / "data" / "processed" / "literature" / "manual_merge_20260726"
REPORT = PROJECT_ROOT / "reports" / "literature_ingest" / "manual_merge_20260726_parse_quality.json"


def load_latest(name: str) -> tuple[str, Path]:
    run_id = (BASE / name).read_text(encoding="ascii").strip()
    return run_id, BASE / "runs" / run_id


def main() -> None:
    preflight_id, preflight_root = load_latest("latest_preflight_run.txt")
    docling_id, docling_root = load_latest("latest_docling_fulltext_run.txt")
    preflight = json.loads((preflight_root / "ledger.json").read_text(encoding="utf-8"))
    docling = json.loads((docling_root / "ledger.json").read_text(encoding="utf-8"))
    docling_by_id = {item["item_id"]: item for item in docling}

    ready = []
    flags = []
    for item in docling:
        if item["status"] != "parsed":
            flags.append({"item_id": item["item_id"], "severity": "error", "issue": "docling_parse_failed"})
            continue
        output = PROJECT_ROOT / item["output_relative_path"]
        payload = json.loads(output.read_text(encoding="utf-8"))
        text_size = payload["metrics"]["full_text_characters"]
        title = payload["document"]["title"]
        item_flags = []
        if text_size < 1000:
            item_flags.append("docling_text_below_1000_characters")
        if title == item["record"]:
            item_flags.append("docling_title_fallback_to_record")
        if item_flags:
            for issue in item_flags:
                flags.append({"item_id": item["item_id"], "severity": "warning", "issue": issue})
        else:
            ready.append(item)

    long_si = [
        item for item in preflight
        if item["status"] == "parsed"
        and item["role"] != "fulltext"
        and item["metrics"].get("full_text_characters", 0) >= 1000
    ]
    short_si = [
        item for item in preflight
        if item["status"] == "parsed"
        and item["role"] != "fulltext"
        and item["metrics"].get("full_text_characters", 0) < 1000
    ]
    legacy_docs = [item for item in preflight if item["status"] == "manual_review_required"]
    text_sizes = [item["metrics"]["full_text_characters"] for item in ready]
    report = {
        "schema_version": 1,
        "preflight_run": preflight_id,
        "docling_fulltext_run": docling_id,
        "quality_gate": {
            "main_fulltexts_ready_for_incremental_chunking": len(ready),
            "main_fulltexts_with_quality_flags": len(docling) - len(ready),
            "long_text_supplements_pending_structured_conversion": len(long_si),
            "short_or_figure_supplements_excluded_from_initial_text_rag": len(short_si),
            "legacy_doc_supplements_pending_conversion": len(legacy_docs),
            "do_not_index_structures": True,
        },
        "ready_fulltext_character_distribution": {
            "minimum": min(text_sizes) if text_sizes else 0,
            "median": int(median(text_sizes)) if text_sizes else 0,
            "maximum": max(text_sizes) if text_sizes else 0,
        },
        "ready_by_category": dict(Counter(item["category"] for item in ready)),
        "quality_flags": flags,
        "source_files_modified": False,
        "indexing_recommendation": "Create a separate incremental chunk manifest only from the ready main fulltexts; retain all SI records and resolve them in a subsequent SI conversion pass.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
