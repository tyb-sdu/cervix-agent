from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


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


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def sentence_chunks(text: str) -> Iterator[tuple[int, int, str]]:
    text = normalized(text)
    start = 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):
            boundary = max(
                text.rfind(". ", start + MIN_CHARS, end),
                text.rfind("? ", start + MIN_CHARS, end),
                text.rfind("! ", start + MIN_CHARS, end),
                text.rfind("; ", start + MIN_CHARS, end),
            )
            if boundary > start:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            yield start, end, piece
        if end >= len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)


def markdown_blocks(markdown: str) -> list[tuple[str, str, str]]:
    blocks: list[tuple[str, str, str]] = []
    title = "Document text"
    lines: list[str] = []
    for line in markdown.splitlines():
        if line.strip().startswith("#"):
            content = " ".join(lines).strip()
            if content:
                blocks.append(("markdown_section", title, content))
            title = line.strip().lstrip("#").strip() or "Untitled heading"
            lines = []
        else:
            lines.append(line)
    content = " ".join(lines).strip()
    if content:
        blocks.append(("markdown_section", title, content))
    return blocks or [("markdown_document", "Document text", markdown)]


def main() -> None:
    run_id = (BASE / "latest_docling_fulltext_run.txt").read_text(encoding="ascii").strip()
    run_root = BASE / "runs" / run_id
    ledger = json.loads((run_root / "ledger.json").read_text(encoding="utf-8"))
    output_root = run_root / "chunks" / CHUNK_VERSION
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "chunks.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    total = 0
    document_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8") as stream:
        for entry in ledger:
            if entry["status"] != "parsed":
                raise RuntimeError(f"Unparsed main fulltext: {entry['item_id']}")
            parsed_path = PROJECT_ROOT / entry["output_relative_path"]
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
            local = 0
            for kind, section_title, block in markdown_blocks(parsed["document"]["markdown"]):
                for start, end, text in sentence_chunks(block):
                    record = {
                        "schema_version": 1,
                        "chunk_version": CHUNK_VERSION,
                        "chunk_id": f"{entry['item_id']}-{local:05d}",
                        "selection_id": entry["item_id"],
                        "document_family_id": entry["record"],
                        "canonical_title": parsed["document"]["title"],
                        "topic_folder": entry["category"],
                        "source_relative_path": entry["source_relative_path"],
                        "source_sha256": entry["source_sha256"],
                        "parsed_output_relative_path": entry["output_relative_path"],
                        "parsed_output_sha256": entry["output_sha256"],
                        "primary_format": ".pdf",
                        "block_kind": kind,
                        "section_title": section_title,
                        "block_start_character": start,
                        "block_end_character": end,
                        "text": text,
                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    local += 1
                    total += 1
                    document_counts[entry["item_id"]] += 1
                    kind_counts[kind] += 1
    temporary.replace(output)
    summary = {
        "schema_version": 1,
        "source_docling_run": run_id,
        "chunk_version": CHUNK_VERSION,
        "document_count": len(document_counts),
        "chunk_count": total,
        "minimum_chunks_per_document": min(document_counts.values()),
        "maximum_chunks_per_document": max(document_counts.values()),
        "chunks_by_kind": dict(kind_counts),
        "chunks_jsonl": str(output.relative_to(PROJECT_ROOT)),
        "chunks_jsonl_sha256": sha256(output),
        "source_files_modified": False,
        "indexing_status": "not_indexed",
    }
    (output_root / "chunk_manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
