from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data2/lxj/projects/CervixAgent")
BASE = ROOT / "data" / "processed" / "literature" / "manual_merge_20260726"
ANTIWORD = ROOT / "tmp" / "antiword_20260726" / "root" / "usr" / "bin" / "antiword"
MAPPING = ROOT / "tmp" / "antiword_20260726" / "root" / "usr" / "share" / "antiword" / "UTF-8.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    preflight_id = (BASE / "latest_preflight_run.txt").read_text(encoding="ascii").strip()
    preflight = json.loads((BASE / "runs" / preflight_id / "ledger.json").read_text(encoding="utf-8"))
    candidates = [item for item in preflight if item["status"] == "manual_review_required" and item["extension"] == ".doc"]
    if not ANTIWORD.is_file() or not MAPPING.is_file():
        raise RuntimeError("Project-local antiword runtime is unavailable")
    run_id = "legacy_doc_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = BASE / "runs" / run_id
    docs = run_root / "documents"
    docs.mkdir(parents=True, exist_ok=False)
    ledger = []
    for index, item in enumerate(candidates, start=1):
        source = ROOT / item["source_relative_path"]
        started = time.perf_counter()
        result = {"item_id": item["item_id"], "record": item["record"], "category": item["category"], "role": item["role"], "source_relative_path": item["source_relative_path"], "source_sha256": item["source_sha256"], "extension": ".doc", "status": "failed"}
        try:
            if sha256(source) != item["source_sha256"]:
                raise RuntimeError("Source hash changed")
            environment = dict(os.environ)
            environment["HOME"] = str(ANTIWORD.parents[2])
            process = subprocess.run([str(ANTIWORD), "-m", "UTF-8.txt", str(source)], env=environment, capture_output=True, check=True, timeout=120)
            text = process.stdout.decode("utf-8", errors="replace").strip()
            warnings = []
            if len(text) < 1000:
                warnings.append("antiword_text_below_1000_characters")
            payload = {"schema_version": 1, "source": {"relative_path": item["source_relative_path"], "sha256": item["source_sha256"], "source_modified": False}, "document": {"title": item["record"] + " supplementary material", "text": text}, "metrics": {"full_text_characters": len(text), "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}, "parser": {"route": "project_local_antiword", "antiword_version": "0.37-17"}, "warnings": warnings}
            output = docs / f"{item['item_id']}.json"
            write(output, payload)
            result.update({"status": "parsed", "warnings": warnings, "metrics": payload["metrics"], "output_relative_path": str(output.relative_to(ROOT)), "output_sha256": sha256(output)})
        except Exception as exc:
            result["error"] = repr(exc)
        result["duration_seconds"] = round(time.perf_counter() - started, 3)
        ledger.append(result)
        print(f"{index}/{len(candidates)} {result['item_id']} {result['status']} {result['duration_seconds']}s", flush=True)
    statuses = Counter(item["status"] for item in ledger)
    warnings = Counter(value for item in ledger for value in item.get("warnings", []))
    summary = {"schema_version": 1, "run_id": run_id, "selected_legacy_doc_count": len(candidates), "status_counts": dict(statuses), "warning_counts": dict(warnings), "source_files_modified": False, "indexing_status": "not_indexed"}
    write(run_root / "ledger.json", ledger)
    write(run_root / "summary.json", summary)
    (BASE / "latest_legacy_doc_run.txt").write_text(run_id + "\n", encoding="ascii")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if statuses.get("failed", 0) == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
