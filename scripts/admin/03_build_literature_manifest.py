from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


LITERATURE_ROOT = Path("/data2/lxj/CervixAgent文献")
OUTPUT_ROOT = Path(
    "/data2/lxj/projects/CervixAgent/manifests/literature_20260724"
)
EXPECTED_FILES = 2970
EXPECTED_BYTES = 6_097_732_334


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(extension: str) -> str:
    return {
        ".pdf": "pdf",
        ".xml": "xml",
        ".nxml": "jats_xml",
        ".docx": "docx",
        ".doc": "doc",
        ".xlsx": "xlsx",
        ".xls": "xls",
        ".csv": "csv",
        ".tsv": "tsv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".ndjson": "ndjson",
        ".pdb": "pdb",
        ".cif": "mmcif",
        ".mmcif": "mmcif",
        ".sdf": "sdf",
        ".mol": "mol",
        ".mol2": "mol2",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".tif": "image",
        ".tiff": "image",
        ".zip": "archive",
        ".gz": "archive",
        ".tar": "archive",
    }.get(extension, "other")


def main() -> int:
    if not LITERATURE_ROOT.is_dir():
        raise SystemExit(f"Missing literature root: {LITERATURE_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    files = sorted(
        (item for item in LITERATURE_ROOT.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(LITERATURE_ROOT).as_posix(),
    )

    extension_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    zero_byte_files: list[str] = []
    total_bytes = 0

    for index, path in enumerate(files, start=1):
        relative_path = path.relative_to(LITERATURE_ROOT).as_posix()
        stat = path.stat()
        extension = path.suffix.lower()
        topic = relative_path.split("/", 1)[0] if "/" in relative_path else "_root"
        digest = sha256_file(path)
        total_bytes += stat.st_size
        extension_counts[extension or "[no_extension]"] += 1
        topic_counts[topic] += 1
        if stat.st_size == 0:
            zero_byte_files.append(relative_path)
        rows.append(
            {
                "relative_path": relative_path,
                "file_name": path.name,
                "extension": extension,
                "file_type": classify(extension),
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": digest,
                "topic_folder": topic,
                "parse_status": "pending",
                "document_family_id": "",
            }
        )
        if index % 100 == 0 or index == len(files):
            print(
                f"hashed_files={index}/{len(files)} hashed_bytes={total_bytes}",
                flush=True,
            )

    csv_path = OUTPUT_ROOT / "literature_files_20260724.csv"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    fieldnames = [
        "relative_path",
        "file_name",
        "extension",
        "file_type",
        "size_bytes",
        "mtime_utc",
        "sha256",
        "topic_folder",
        "parse_status",
        "document_family_id",
    ]
    with csv_temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_temporary, csv_path)

    manifest_sha256 = sha256_file(csv_path)
    summary = {
        "schema_version": 1,
        "status": (
            "accepted"
            if len(files) == EXPECTED_FILES
            and total_bytes == EXPECTED_BYTES
            and not zero_byte_files
            else "review_required"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "literature_root": str(LITERATURE_ROOT),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "expected_file_count": EXPECTED_FILES,
        "expected_total_bytes": EXPECTED_BYTES,
        "file_count_matches": len(files) == EXPECTED_FILES,
        "total_bytes_matches": total_bytes == EXPECTED_BYTES,
        "zero_byte_file_count": len(zero_byte_files),
        "zero_byte_files": zero_byte_files,
        "topic_counts": dict(sorted(topic_counts.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
        "manifest_csv": csv_path.name,
        "manifest_csv_sha256": manifest_sha256,
        "hash_algorithm": "SHA-256",
        "notes": [
            "The literature source directory was read only; no source file was modified.",
            "parse_status and document_family_id are placeholders for the next parsing stage."
        ],
    }
    summary_path = OUTPUT_ROOT / "literature_acceptance_20260724.json"
    summary_temporary = summary_path.with_suffix(".json.tmp")
    summary_temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(summary_temporary, summary_path)

    checksum_path = OUTPUT_ROOT / "literature_manifest_20260724.sha256"
    checksum_path.write_text(
        f"{manifest_sha256}  {csv_path.name}\n", encoding="ascii"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "accepted" else 3


if __name__ == "__main__":
    sys.exit(main())
