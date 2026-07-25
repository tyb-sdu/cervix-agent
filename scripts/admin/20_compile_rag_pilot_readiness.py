from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
PILOT_ROOT = PROJECT_ROOT / "data" / "processed" / "literature" / "pilot_100"


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


def main() -> int:
    run_id = (PILOT_ROOT / "latest_full_run.txt").read_text(
        encoding="ascii"
    ).strip()
    run_root = PILOT_ROOT / "runs" / run_id
    parse_summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    title_summary = json.loads(
        (run_root / "title_enrichment_summary.json").read_text(encoding="utf-8")
    )
    paired_summary = json.loads(
        (run_root / "paired_format_validation_6.json").read_text(encoding="utf-8")
    )
    with (run_root / "title_enrichment.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        title_rows = list(csv.DictReader(stream))
    with (run_root / "manual_spot_check_20_enriched.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        manual_rows = list(csv.DictReader(stream))

    gates = {
        "all_selected_documents_parsed": parse_summary.get("status") == "completed"
        and parse_summary.get("selected_count") == 100,
        "canonical_titles_complete": title_summary.get(
            "unresolved_canonical_title_count"
        ) == 0 and len(title_rows) == 100,
        "paired_format_check_passed": paired_summary.get("status") == "passed",
        "manual_structural_review_queue_prepared": len(manual_rows) == 20,
        "source_files_preserved": parse_summary.get("source_files_modified") is False,
    }
    readiness = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "RAG pilot structural readiness only; not clinical, chemical, or biological validation.",
        "gates": gates,
        "automated_gate_status": "passed" if all(gates.values()) else "review_required",
        "next_authorized_technical_action": (
            "Create a versioned chunk manifest from the 100 parsed records, using "
            "canonical_title from title_enrichment.csv; do not index any raw parser title."
        ),
        "human_review_remaining": {
            "required_rows": len(manual_rows),
            "worksheet": "manual_spot_check_20_enriched.csv",
            "checks": [
                "正文是否可读且没有明显页眉页脚/双栏串行问题",
                "关键表格、图题和引文能否追溯回原文",
                "标题与原文是否一致",
                "若用于结论，是否需要标注研究类型和证据强度",
            ],
            "note": "该人工抽检用于质量控制；不需要对每篇文献作医学结论。",
        },
        "artifacts": {
            "parse_summary": "summary.json",
            "title_enrichment": "title_enrichment.csv",
            "paired_validation": "paired_format_validation_6.csv",
            "manual_review_queue": "manual_spot_check_20_enriched.csv",
        },
    }
    output = run_root / "rag_pilot_readiness_20260725.json"
    write_json_atomic(output, readiness)
    (run_root / "rag_pilot_readiness_20260725.sha256").write_text(
        f"{sha256(output)}  {output.name}\n", encoding="ascii"
    )
    print(json.dumps(readiness, ensure_ascii=False, indent=2))
    return 0 if readiness["automated_gate_status"] == "passed" else 3


if __name__ == "__main__":
    sys.exit(main())
