from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/data2/lxj/projects/CervixAgent")
DEFAULT_REPORT_ROOT = ROOT / "reports" / "rag_acceptance"
DEFAULT_RUNS = [
    DEFAULT_REPORT_ROOT / "rag_acceptance_20260727T023620Z" / "results.json",
    DEFAULT_REPORT_ROOT / "rag_acceptance_20260727T024041Z" / "results.json",
]


def write_review_form(path: Path, cases: list[dict[str, Any]]) -> None:
    columns = [
        "case_id", "category", "question", "expected_source_family", "expected_behavior",
        "run_status", "citation_id_check_passed", "reviewer",
        "source_selection_pass_yes_no", "answer_entailment_pass_yes_no",
        "citation_traceability_pass_yes_no", "evidence_boundary_pass_yes_no",
        "final_decision_pass_fail_hold", "reviewer_notes",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            writer.writerow({key: case.get(key, "") for key in columns})


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine completed CervixAgent RAG acceptance runs")
    parser.add_argument("--runs", type=Path, nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()
    merged: list[dict[str, Any]] = []
    source_runs: list[str] = []
    seen: set[str] = set()
    for path in args.runs:
        data = json.loads(path.read_text(encoding="utf-8"))
        source_runs.append(str(path.relative_to(ROOT)))
        for case in data["cases"]:
            case_id = case["id"]
            if case_id in seen:
                raise ValueError(f"Duplicate acceptance case: {case_id}")
            seen.add(case_id)
            merged.append(case)
    merged.sort(key=lambda item: int(item["id"].split("-")[1]))
    run_id = "rag_acceptance_v1_20260727_combined"
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_runs": source_runs,
        "cases": merged,
        "summary": {
            "case_count": len(merged),
            "completed": sum(case.get("run_status") == "completed" for case in merged),
            "failed": sum(case.get("run_status") == "failed" for case in merged),
            "citation_id_checks_passed": sum(case.get("citation_id_check_passed") is True for case in merged),
            "human_review_required": True,
        },
    }
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review_form(output / "human_review_form.csv", merged)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
