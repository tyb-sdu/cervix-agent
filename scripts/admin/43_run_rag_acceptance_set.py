from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path("/data2/lxj/projects/CervixAgent")
DEFAULT_QUESTION_SET = ROOT / "tests" / "rag_acceptance" / "acceptance_questions_v1.json"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "rag_acceptance"
RAG_SCRIPT = ROOT / "scripts" / "cervixagent-rag.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_rag_module() -> Any:
    spec = importlib.util.spec_from_file_location("cervixagent_terminal_rag", RAG_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load terminal RAG script: {RAG_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_review_form(path: Path, cases: list[dict[str, Any]]) -> None:
    columns = [
        "case_id",
        "category",
        "question",
        "expected_source_family",
        "expected_behavior",
        "run_status",
        "citation_id_check_passed",
        "reviewer",
        "source_selection_pass_yes_no",
        "answer_entailment_pass_yes_no",
        "citation_traceability_pass_yes_no",
        "evidence_boundary_pass_yes_no",
        "final_decision_pass_fail_hold",
        "reviewer_notes",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            writer.writerow({key: case.get(key, "") for key in columns})


def selected_cases(question_set: dict[str, Any], ids: set[str] | None) -> list[dict[str, Any]]:
    cases = question_set["cases"]
    if not ids:
        return cases
    found = {case["id"] for case in cases}
    unknown = ids - found
    if unknown:
        raise ValueError(f"Unknown acceptance case IDs: {', '.join(sorted(unknown))}")
    return [case for case in cases if case["id"] in ids]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible CervixAgent RAG human-acceptance cases")
    parser.add_argument("--question-set", type=Path, default=DEFAULT_QUESTION_SET)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--ids", help="Comma-separated subset, for example ACPT-01,ACPT-04")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--llm-device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=700)
    args = parser.parse_args()

    question_bytes = args.question_set.read_bytes()
    question_set = json.loads(question_bytes)
    ids = {value.strip() for value in args.ids.split(",")} if args.ids else None
    cases = selected_cases(question_set, ids)
    run_id = f"rag_acceptance_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = args.report_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "results.json"
    review_path = run_dir / "human_review_form.csv"
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": utc_now(),
        "question_set": str(args.question_set.relative_to(ROOT)),
        "question_set_sha256": hashlib.sha256(question_bytes).hexdigest(),
        "parameters": {
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "device": args.device,
            "llm_device": args.llm_device,
            "max_new_tokens": args.max_new_tokens,
        },
        "review_rule": question_set["review_rule"],
        "cases": [],
    }
    rag = load_rag_module()
    for case in cases:
        started = perf_counter()
        case_result: dict[str, Any] = {**case, "started_at": utc_now()}
        try:
            answer = rag.answer(
                case["question"],
                args.top_k,
                args.candidate_k,
                args.device,
                args.llm_device,
                args.max_new_tokens,
            )
            case_result.update(
                {
                    "run_status": "completed",
                    "duration_seconds": round(perf_counter() - started, 3),
                    "answer": answer,
                    "citation_id_check_passed": answer["citation_validation"]["citation_id_check_passed"],
                }
            )
        except Exception as exc:  # Preserve a failure record and continue with other cases.
            case_result.update(
                {
                    "run_status": "failed",
                    "duration_seconds": round(perf_counter() - started, 3),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "citation_id_check_passed": "",
                }
            )
        case_result["finished_at"] = utc_now()
        report["cases"].append(case_result)
        write_json(report_path, report)
        print(f"{case['id']}: {case_result['run_status']} ({case_result['duration_seconds']} s)", flush=True)
    report["finished_at"] = utc_now()
    report["summary"] = {
        "case_count": len(report["cases"]),
        "completed": sum(case["run_status"] == "completed" for case in report["cases"]),
        "failed": sum(case["run_status"] == "failed" for case in report["cases"]),
        "human_review_required": True,
    }
    write_json(report_path, report)
    write_review_form(review_path, report["cases"])
    print(f"Results: {report_path}")
    print(f"Human review form: {review_path}")
    return 0 if report["summary"]["failed"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
