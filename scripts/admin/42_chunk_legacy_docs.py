from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path("/data2/lxj/projects/CervixAgent")
BASE = ROOT / "data" / "processed" / "literature" / "manual_merge_20260726"
VERSION = "v1_char1200_overlap180_structured"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pieces(value: str):
    value = re.sub(r"\s+", " ", value).strip()
    start = 0
    while start < len(value):
        end = min(start + 1200, len(value))
        if end < len(value):
            boundary = max(value.rfind(". ", start + 180, end), value.rfind("; ", start + 180, end))
            if boundary > start:
                end = boundary + 1
        text = value[start:end].strip()
        if text:
            yield start, end, text
        if end >= len(value):
            break
        start = max(end - 180, start + 1)


def main() -> None:
    run_id = (BASE / "latest_legacy_doc_run.txt").read_text(encoding="ascii").strip()
    run = BASE / "runs" / run_id
    ledger = json.loads((run / "ledger.json").read_text(encoding="utf-8"))
    target = run / "chunks" / VERSION
    target.mkdir(parents=True, exist_ok=True)
    output = target / "chunks.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    count, per_doc = 0, Counter()
    with temporary.open("w", encoding="utf-8") as stream:
        for entry in ledger:
            if entry["status"] != "parsed":
                raise RuntimeError(f"Unparsed legacy DOC: {entry['item_id']}")
            parsed = json.loads((ROOT / entry["output_relative_path"]).read_text(encoding="utf-8"))
            for index, (start, end, text) in enumerate(pieces(parsed["document"]["text"])):
                row = {"schema_version": 1, "chunk_version": VERSION, "chunk_id": f"{entry['item_id']}-{index:05d}", "selection_id": entry["item_id"], "document_family_id": entry["record"], "canonical_title": parsed["document"]["title"], "topic_folder": entry["category"], "source_relative_path": entry["source_relative_path"], "source_sha256": entry["source_sha256"], "source_role": entry["role"], "parsed_output_relative_path": entry["output_relative_path"], "parsed_output_sha256": entry["output_sha256"], "primary_format": ".doc", "block_kind": "supplement_document", "section_title": "Supplementary material", "block_start_character": start, "block_end_character": end, "text": text, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1; per_doc[entry["item_id"]] += 1
    temporary.replace(output)
    summary = {"schema_version": 1, "source_legacy_doc_run": run_id, "document_count": len(per_doc), "chunk_count": count, "chunks_jsonl": str(output.relative_to(ROOT)), "chunks_jsonl_sha256": sha256(output), "source_files_modified": False, "indexing_status": "not_indexed"}
    (target / "chunk_manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
