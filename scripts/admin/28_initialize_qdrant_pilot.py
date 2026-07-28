from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
PILOT_ROOT = PROJECT_ROOT / "data" / "processed" / "literature" / "pilot_100"
VECTORSTORE_ROOT = PROJECT_ROOT / "vectorstores" / "qdrant"
CONFIG_ROOT = PROJECT_ROOT / "configs" / "rag"
REPORT_ROOT = PROJECT_ROOT / "reports" / "qdrant"

COLLECTION = "cervixagent_literature_pilot_text_qwen3vl8b_4096_v1"
VECTOR_SIZE = 4096
EXPECTED_CHUNKS = 7797


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    chunk_root = (
        PILOT_ROOT
        / "runs"
        / "parse_full_100_20260724T040510Z"
        / "chunks"
        / "v1_char1200_overlap180_structured"
    )
    chunk_summary_path = chunk_root / "chunk_manifest_summary.json"
    chunk_summary = json.loads(chunk_summary_path.read_text(encoding="utf-8"))
    if chunk_summary["chunk_count"] != EXPECTED_CHUNKS:
        raise RuntimeError("Unexpected pilot chunk count; refusing to create index")
    if chunk_summary["chunking"]["max_characters"] != 1200:
        raise RuntimeError("Unexpected chunking configuration")

    VECTORSTORE_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(VECTORSTORE_ROOT))
    exists = client.collection_exists(COLLECTION)
    if not exists:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    info = client.get_collection(COLLECTION)
    vector_config = info.config.params.vectors
    actual_size = getattr(vector_config, "size", None)
    actual_distance = str(getattr(vector_config, "distance", ""))
    if actual_size != VECTOR_SIZE or "Cosine" not in actual_distance:
        raise RuntimeError(
            f"Existing collection configuration mismatch: {actual_size}, {actual_distance}"
        )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection": COLLECTION,
        "qdrant_mode": "local_persistent",
        "qdrant_path": str(VECTORSTORE_ROOT),
        "vector_size": VECTOR_SIZE,
        "distance": "cosine",
        "embedding_model": "Qwen/Qwen3-VL-Embedding-8B",
        "embedding_model_path": str(
            PROJECT_ROOT / "models" / "weights" / "qwen3-vl-embedding-8b"
        ),
        "source_chunk_manifest": str(chunk_root / "chunks.jsonl"),
        "source_chunk_manifest_sha256": chunk_summary["chunks_jsonl_sha256"],
        "expected_chunk_count": EXPECTED_CHUNKS,
        "point_count_before_indexing": info.points_count,
        "source_files_modified": False,
        "indexing_policy": {
            "vectorize": "text field only",
            "store_as_payload": [
                "chunk_id",
                "canonical_title",
                "topic_folder",
                "source_relative_path",
                "source_sha256",
                "section_title",
                "text",
                "text_sha256",
                "chunk_version",
            ],
            "idempotency_key": "chunk_id + text_sha256 + embedding_model + chunk_version",
        },
    }
    config_path = CONFIG_ROOT / "qdrant_pilot_text_v1.json"
    write_json_atomic(config_path, manifest)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection_created_now": not exists,
        "collection": COLLECTION,
        "vector_size": actual_size,
        "distance": actual_distance,
        "point_count": info.points_count,
        "status": "ready_for_embedding_indexing",
        "config": str(config_path),
    }
    write_json_atomic(REPORT_ROOT / "pilot_text_v1_initialization.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
