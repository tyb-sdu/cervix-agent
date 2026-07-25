from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from lxml import etree


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
LITERATURE_ROOT = PROJECT_ROOT / "data" / "literature"
PILOT_ROOT = PROJECT_ROOT / "data" / "processed" / "literature" / "pilot_100"
PILOT_MANIFEST = PILOT_ROOT / "pilot_100_manifest.csv"
MODEL_ROOT = PROJECT_ROOT / "models"

GENERIC_TITLES = {
    "<!-- image -->",
    "research article",
    "research-article",
    "article",
    "original article",
    "review article",
}

os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "huggingface"))
os.environ.setdefault(
    "TRANSFORMERS_CACHE", str(MODEL_ROOT / "huggingface" / "hub")
)
os.environ.setdefault("TORCH_HOME", str(MODEL_ROOT / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(MODEL_ROOT / "cache"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def usable_title(value: str) -> bool:
    normalized = normalize_text(value)
    return (
        20 <= len(normalized) <= 500
        and normalized.lower() not in GENERIC_TITLES
        and "<!--" not in normalized
    )


def source_metadata_title(source: Path) -> str:
    """Read the catalogued article title without changing source records."""
    metadata_path = source.parent / "metadata.json"
    if not metadata_path.is_file():
        return ""
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    title = payload.get("title", "")
    return normalize_text(title) if isinstance(title, str) else ""


def normalized_text(node: etree._Element | None) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", " ".join(node.itertext())).strip()


def xpath_first_text(root: etree._Element, expression: str) -> str:
    values = root.xpath(expression)
    if not values:
        return ""
    value = values[0]
    if isinstance(value, etree._Element):
        return normalized_text(value)
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_jats_xml(source: Path) -> dict[str, Any]:
    parser = etree.XMLParser(
        recover=True,
        resolve_entities=False,
        no_network=True,
        huge_tree=True,
    )
    tree = etree.parse(str(source), parser)
    root = tree.getroot()
    title = xpath_first_text(root, "//*[local-name()='article-title'][1]")
    journal = xpath_first_text(root, "//*[local-name()='journal-title'][1]")
    abstract_nodes = root.xpath("//*[local-name()='abstract'][1]")
    abstract = normalized_text(abstract_nodes[0]) if abstract_nodes else ""

    identifiers: dict[str, str] = {}
    for node in root.xpath("//*[local-name()='article-id']"):
        identifier_type = (
            node.get("pub-id-type")
            or node.get("specific-use")
            or "unknown"
        )
        value = normalized_text(node)
        if value and identifier_type not in identifiers:
            identifiers[identifier_type] = value

    sections: list[dict[str, Any]] = []
    for index, section in enumerate(
        root.xpath("//*[local-name()='body']//*[local-name()='sec']"),
        start=1,
    ):
        title_nodes = section.xpath("./*[local-name()='title'][1]")
        section_title = (
            normalized_text(title_nodes[0])
            if title_nodes
            else f"Untitled section {index}"
        )
        paragraphs = [
            normalized_text(node)
            for node in section.xpath("./*[local-name()='p']")
            if normalized_text(node)
        ]
        if not paragraphs:
            paragraphs = [
                normalized_text(node)
                for node in section.xpath(
                    "./*[local-name()='sec']/*[local-name()='p']"
                )
                if normalized_text(node)
            ]
        sections.append(
            {
                "index": index,
                "title": section_title,
                "paragraphs": paragraphs,
                "text": "\n\n".join(paragraphs),
            }
        )

    figures = []
    for index, figure in enumerate(
        root.xpath("//*[local-name()='fig']"), start=1
    ):
        label_nodes = figure.xpath("./*[local-name()='label'][1]")
        caption_nodes = figure.xpath(
            "./*[local-name()='caption'][1]"
        )
        figures.append(
            {
                "index": index,
                "label": normalized_text(label_nodes[0]) if label_nodes else "",
                "caption": (
                    normalized_text(caption_nodes[0]) if caption_nodes else ""
                ),
            }
        )

    tables = []
    for index, table in enumerate(
        root.xpath("//*[local-name()='table-wrap']"), start=1
    ):
        label_nodes = table.xpath("./*[local-name()='label'][1]")
        caption_nodes = table.xpath("./*[local-name()='caption'][1]")
        table_nodes = table.xpath(".//*[local-name()='table'][1]")
        tables.append(
            {
                "index": index,
                "label": normalized_text(label_nodes[0]) if label_nodes else "",
                "caption": (
                    normalized_text(caption_nodes[0]) if caption_nodes else ""
                ),
                "text": normalized_text(table_nodes[0]) if table_nodes else "",
            }
        )

    references = [
        normalized_text(node)
        for node in root.xpath(
            "//*[local-name()='ref-list']/*[local-name()='ref']"
        )
        if normalized_text(node)
    ]
    body_text = "\n\n".join(section["text"] for section in sections)
    combined_text = "\n\n".join(
        text for text in (title, abstract, body_text) if text
    )
    return {
        "document": {
            "title": title,
            "journal": journal,
            "identifiers": identifiers,
            "abstract": abstract,
            "sections": sections,
            "figures": figures,
            "tables": tables,
            "references": references,
            "full_text": combined_text,
        },
        "metrics": {
            "title_characters": len(title),
            "abstract_characters": len(abstract),
            "body_characters": len(body_text),
            "full_text_characters": len(combined_text),
            "section_count": len(sections),
            "figure_count": len(figures),
            "table_count": len(tables),
            "reference_count": len(references),
        },
    }


def parse_pdf_docling(source: Path, metadata_title: str = "") -> dict[str, Any]:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(source)
    markdown = result.document.export_to_markdown()
    markdown = markdown.strip()
    markdown_title = ""
    for line in markdown.splitlines():
        candidate = line.strip().lstrip("#").strip()
        candidate = re.sub(r"<[^>]+>", "", candidate).strip()
        if usable_title(candidate) and candidate.lower() not in {
            "abstract", "introduction", "keywords"
        }:
            markdown_title = candidate
            break
    title = metadata_title if usable_title(metadata_title) else markdown_title
    page_count: int | None = None
    pages = getattr(result.document, "pages", None)
    if pages is not None:
        try:
            page_count = len(pages)
        except TypeError:
            page_count = None
    heading_count = sum(
        1 for line in markdown.splitlines() if line.lstrip().startswith("#")
    )
    table_count = markdown.count("<table")
    return {
        "document": {
            "title": title,
            "title_source": (
                "source_metadata"
                if usable_title(metadata_title)
                else "markdown_fallback"
            ),
            "markdown": markdown,
        },
        "metrics": {
            "title_characters": len(title),
            "full_text_characters": len(markdown),
            "page_count": page_count,
            "heading_count": heading_count,
            "table_marker_count": table_count,
        },
    }


def select_rows(mode: str) -> list[dict[str, str]]:
    with PILOT_MANIFEST.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if mode == "full":
        return rows
    chosen: list[dict[str, str]] = []
    topics = sorted({row["topic_folder"] for row in rows})
    for topic in topics:
        for file_format in ("xml", "pdf"):
            matches = [
                row
                for row in rows
                if row["topic_folder"] == topic
                and row["primary_format"] == file_format
            ]
            if not matches:
                raise RuntimeError(
                    f"No {file_format} sample available for topic {topic}"
                )
            chosen.append(matches[0])
    return chosen


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()

    selected_rows = select_rows(args.mode)
    run_id = (
        f"parse_{args.mode}_{len(selected_rows)}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_root = PILOT_ROOT / "runs" / run_id
    documents_root = run_root / "documents"
    documents_root.mkdir(parents=True, exist_ok=False)

    ledger: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        selection_id = row["selection_id"]
        source = LITERATURE_ROOT / row["primary_relative_path"]
        started = time.perf_counter()
        entry: dict[str, Any] = {
            "selection_id": selection_id,
            "topic_folder": row["topic_folder"],
            "document_family_id": row["document_family_id"],
            "primary_format": row["primary_format"],
            "parser_route": row["parser_route"],
            "source_relative_path": row["primary_relative_path"],
            "expected_sha256": row["primary_sha256"],
            "status": "failed",
        }
        try:
            actual_sha256 = sha256_file(source)
            entry["actual_sha256"] = actual_sha256
            if actual_sha256 != row["primary_sha256"]:
                raise RuntimeError("Source SHA-256 differs from pilot manifest")

            if row["primary_format"] == "xml":
                parsed = parse_jats_xml(source)
            elif row["primary_format"] == "pdf":
                parsed = parse_pdf_docling(
                    source, source_metadata_title(source)
                )
            else:
                raise RuntimeError(
                    f"Unsupported primary format: {row['primary_format']}"
                )

            output_payload = {
                "schema_version": 1,
                "selection": row,
                "source": {
                    "root": str(LITERATURE_ROOT),
                    "relative_path": row["primary_relative_path"],
                    "sha256": actual_sha256,
                    "source_modified": False,
                },
                "parser": {
                    "route": row["parser_route"],
                    "lxml_version": version("lxml"),
                    "docling_version": (
                        version("docling")
                        if row["primary_format"] == "pdf"
                        else None
                    ),
                },
                **parsed,
            }
            output_path = documents_root / f"{selection_id}.json"
            write_json_atomic(output_path, output_payload)
            entry["output_relative_path"] = str(
                output_path.relative_to(PROJECT_ROOT)
            )
            entry["output_sha256"] = sha256_file(output_path)
            entry["metrics"] = parsed["metrics"]
            entry["status"] = "parsed"
        except Exception as exc:
            entry["error"] = repr(exc)
            entry["traceback"] = traceback.format_exc()
        finally:
            entry["duration_seconds"] = round(
                time.perf_counter() - started, 3
            )
            ledger.append(entry)
            print(
                f"{index}/{len(selected_rows)} {selection_id} "
                f"{entry['primary_format']} {entry['status']} "
                f"{entry['duration_seconds']}s",
                flush=True,
            )

    status_counts = Counter(item["status"] for item in ledger)
    quality_flags = []
    for entry in ledger:
        if entry["status"] != "parsed":
            continue
        metrics = entry["metrics"]
        if metrics.get("full_text_characters", 0) < 1000:
            quality_flags.append(
                {
                    "selection_id": entry["selection_id"],
                    "flag": "extracted_text_below_1000_characters",
                }
            )
        if metrics.get("title_characters", 0) == 0:
            quality_flags.append(
                {
                    "selection_id": entry["selection_id"],
                    "flag": "title_missing",
                }
            )

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "mode": args.mode,
        "status": (
            "completed"
            if status_counts.get("failed", 0) == 0
            else "review_required"
        ),
        "started_from_manifest": str(PILOT_MANIFEST),
        "source_root": str(LITERATURE_ROOT),
        "output_root": str(run_root),
        "selected_count": len(selected_rows),
        "status_counts": dict(status_counts),
        "quality_flag_count": len(quality_flags),
        "quality_flags": quality_flags,
        "source_files_modified": False,
        "model_cache_root": str(MODEL_ROOT),
    }
    write_json_atomic(run_root / "ledger.json", ledger)
    write_json_atomic(run_root / "summary.json", summary)
    (PILOT_ROOT / f"latest_{args.mode}_run.txt").write_text(
        run_id + "\n", encoding="ascii"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 3


if __name__ == "__main__":
    sys.exit(main())
