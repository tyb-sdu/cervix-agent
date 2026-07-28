from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
BASE = PROJECT_ROOT / "data" / "processed" / "literature" / "manual_merge_20260726"
CHUNK_VERSION = "v1_char1200_overlap180_structured"
MAX_CHARS = 1200
OVERLAP_CHARS = 180
MIN_CHARS = 180


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def chunks(text: str) -> Iterator[tuple[int, int, str]]:
    text = norm(text)
    start = 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start + MIN_CHARS, end), text.rfind("? ", start + MIN_CHARS, end), text.rfind("! ", start + MIN_CHARS, end), text.rfind("; ", start + MIN_CHARS, end))
            if boundary > start:
                end = boundary + 1
        fragment = text[start:end].strip()
        if fragment:
            yield start, end, fragment
        if end >= len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)


def blocks(markdown: str) -> list[tuple[str, str, str]]:
    found = []
    heading, lines = "Supplementary material", []
    for line in markdown.splitlines():
        if line.strip().startswith("#"):
            content = " ".join(lines).strip()
            if content:
                found.append(("supplement_section", heading, content))
            heading, lines = line.strip().lstrip("#").strip() or "Untitled heading", []
        else:
            lines.append(line)
    content = " ".join(lines).strip()
    if content:
        found.append(("supplement_section", heading, content))
    return found or [("supplement_document", "Supplementary material", markdown)]


def main() -> None:
    run_id = (BASE / "latest_docling_long_si_run.txt").read_text(encoding="ascii").strip()
    root = BASE / "runs" / run_id
    ledger = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    destination = root / "chunks" / CHUNK_VERSION
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "chunks.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    doc_counts, kind_counts = Counter(), Counter()
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for entry in ledger:
            if entry["status"] != "parsed":
                raise RuntimeError(f"Unparsed SI: {entry['item_id']}")
            parsed_path = PROJECT_ROOT / entry["output_relative_path"]
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
            local = 0
            for kind, section, text in blocks(parsed["document"]["markdown"]):
                for start, end, fragment in chunks(text):
                    record = {
                        "schema_version": 1, "chunk_version": CHUNK_VERSION,
                        "chunk_id": f"{entry['item_id']}-{local:05d}",
                        "selection_id": entry["item_id"], "document_family_id": entry["record"],
                        "canonical_title": parsed["document"]["title"],
                        "topic_folder": entry["category"], "source_relative_path": entry["source_relative_path"],
                        "source_sha256": entry["source_sha256"], "source_role": entry["role"],
                        "parsed_output_relative_path": entry["output_relative_path"], "parsed_output_sha256": entry["output_sha256"],
                        "primary_format": entry["extension"], "block_kind": kind, "section_title": section,
                        "block_start_character": start, "block_end_character": end, "text": fragment,
                        "text_sha256": hashlib.sha256(fragment.encode("utf-8")).hexdigest(),
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    local += 1; count += 1; doc_counts[entry["item_id"]] += 1; kind_counts[kind] += 1
    temporary.replace(output)
    summary = {"schema_version": 1, "source_docling_run": run_id, "chunk_version": CHUNK_VERSION, "document_count": len(doc_counts), "chunk_count": count, "minimum_chunks_per_document": min(doc_counts.values()), "maximum_chunks_per_document": max(doc_counts.values()), "chunks_by_kind": dict(kind_counts), "chunks_jsonl": str(output.relative_to(PROJECT_ROOT)), "chunks_jsonl_sha256": sha256(output), "source_files_modified": False, "indexing_status": "not_indexed"}
    (destination / "chunk_manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
