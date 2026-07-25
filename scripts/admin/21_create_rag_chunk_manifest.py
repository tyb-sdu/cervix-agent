from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
PILOT_ROOT = PROJECT_ROOT / "data" / "processed" / "literature" / "pilot_100"
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
    if not text:
        return
    start = 0
    length = len(text)
    while start < length:
        end = min(start + MAX_CHARS, length)
        if end < length:
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
        if end >= length:
            break
        start = max(end - OVERLAP_CHARS, start + 1)


def xml_blocks(document: dict[str, Any]) -> list[tuple[str, str, str]]:
    blocks: list[tuple[str, str, str]] = []
    abstract = normalized(document.get("abstract", ""))
    if abstract:
        blocks.append(("abstract", "Abstract", abstract))
    for section in document.get("sections", []):
        text = normalized(section.get("text", ""))
        if text:
            blocks.append(("section", section.get("title", "Untitled section"), text))
    for table in document.get("tables", []):
        text = normalized(" ".join((table.get("caption", ""), table.get("text", ""))))
        if text:
            blocks.append(("table", table.get("label", "Table"), text))
    for figure in document.get("figures", []):
        text = normalized(figure.get("caption", ""))
        if text:
            blocks.append(("figure_caption", figure.get("label", "Figure"), text))
    return blocks


def pdf_blocks(document: dict[str, Any]) -> list[tuple[str, str, str]]:
    markdown = document.get("markdown", "")
    blocks: list[tuple[str, str, str]] = []
    current_title = "Document text"
    current_lines: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            content = " ".join(current_lines).strip()
            if content:
                blocks.append(("markdown_section", current_title, content))
            current_title = stripped.lstrip("#").strip() or "Untitled heading"
            current_lines = []
        else:
            current_lines.append(line)
    content = " ".join(current_lines).strip()
    if content:
        blocks.append(("markdown_section", current_title, content))
    return blocks or [("markdown_document", "Document text", markdown)]


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    run_id = (PILOT_ROOT / "latest_full_run.txt").read_text(encoding="ascii").strip()
    run_root = PILOT_ROOT / "runs" / run_id
    readiness = json.loads(
        (run_root / "rag_pilot_readiness_20260725.json").read_text(encoding="utf-8")
    )
    if readiness["automated_gate_status"] != "passed":
        raise RuntimeError("RAG pilot readiness gates have not passed")

    ledger = json.loads((run_root / "ledger.json").read_text(encoding="utf-8"))
    with (run_root / "title_enrichment.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        title_by_selection = {
            row["selection_id"]: row["canonical_title"]
            for row in csv.DictReader(stream)
        }

    output_root = run_root / "chunks" / CHUNK_VERSION
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "chunks.jsonl"
    temporary_path = output_path.with_suffix(".jsonl.tmp")
    chunk_count_by_kind: Counter[str] = Counter()
    document_chunk_counts: Counter[str] = Counter()
    count = 0
    with temporary_path.open("w", encoding="utf-8") as stream:
        for entry in ledger:
            if entry["status"] != "parsed":
                raise RuntimeError(f"Unparsed record: {entry['selection_id']}")
            payload_path = PROJECT_ROOT / entry["output_relative_path"]
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            document = payload["document"]
            if entry["primary_format"] == "xml":
                blocks = xml_blocks(document)
            elif entry["primary_format"] == "pdf":
                blocks = pdf_blocks(document)
            else:
                raise RuntimeError(f"Unsupported format {entry['primary_format']}")
            local_index = 0
            for block_kind, section_title, block_text in blocks:
                for start, end, text in sentence_chunks(block_text):
                    chunk_id = f"{entry['selection_id']}-{local_index:05d}"
                    record = {
                        "schema_version": 1,
                        "chunk_version": CHUNK_VERSION,
                        "chunk_id": chunk_id,
                        "selection_id": entry["selection_id"],
                        "document_family_id": entry["document_family_id"],
                        "canonical_title": title_by_selection[entry["selection_id"]],
                        "topic_folder": entry["topic_folder"],
                        "source_relative_path": entry["source_relative_path"],
                        "source_sha256": entry["actual_sha256"],
                        "parsed_output_relative_path": entry["output_relative_path"],
                        "parsed_output_sha256": entry["output_sha256"],
                        "primary_format": entry["primary_format"],
                        "block_kind": block_kind,
                        "section_title": section_title,
                        "block_start_character": start,
                        "block_end_character": end,
                        "text": text,
                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    local_index += 1
                    count += 1
                    chunk_count_by_kind[block_kind] += 1
                    document_chunk_counts[entry["selection_id"]] += 1
    os.replace(temporary_path, output_path)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chunk_version": CHUNK_VERSION,
        "chunking": {
            "max_characters": MAX_CHARS,
            "overlap_characters": OVERLAP_CHARS,
            "minimum_search_for_boundary_characters": MIN_CHARS,
        },
        "document_count": len(document_chunk_counts),
        "chunk_count": count,
        "chunks_by_kind": dict(sorted(chunk_count_by_kind.items())),
        "minimum_chunks_per_document": min(document_chunk_counts.values()),
        "maximum_chunks_per_document": max(document_chunk_counts.values()),
        "chunks_jsonl": output_path.name,
        "chunks_jsonl_sha256": sha256(output_path),
        "source_files_modified": False,
        "indexing_contract": "Embed the text field only; retain every other field as Qdrant payload metadata.",
        "status": "ready_for_embedding_model_selection",
    }
    write_json_atomic(output_root / "chunk_manifest_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
