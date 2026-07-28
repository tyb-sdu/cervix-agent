from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag" / "qdrant_pilot_text_v1.json"
RUN_ROOT = PROJECT_ROOT / "runs" / "qdrant_indexing"
STATE_PATH = RUN_ROOT / "pilot_text_v1_state.json"
REPORT_ROOT = PROJECT_ROOT / "reports" / "qdrant"
NAMESPACE = uuid.UUID("65a298df-c1fc-4962-85a7-50e8932a4d1e")
DEFAULT_BATCH_SIZE = 4
DOCUMENT_PROMPT = "Represent the scientific literature passage for retrieval."


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            if not record.get("text", "").strip():
                raise RuntimeError(f"Empty text in chunk line {line_number}")
            chunks.append(record)
    return chunks


def make_point(record: dict[str, Any], vector: list[float]) -> PointStruct:
    point_uuid = str(uuid.uuid5(NAMESPACE, record["chunk_id"]))
    payload = {
        key: record[key]
        for key in (
            "chunk_id",
            "selection_id",
            "document_family_id",
            "canonical_title",
            "topic_folder",
            "source_relative_path",
            "source_sha256",
            "parsed_output_relative_path",
            "parsed_output_sha256",
            "primary_format",
            "block_kind",
            "section_title",
            "block_start_character",
            "block_end_character",
            "text",
            "text_sha256",
            "chunk_version",
        )
    }
    return PointStruct(id=point_uuid, vector=vector, payload=payload)


def initial_state(config: dict[str, Any], chunks: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": f"pilot_text_qwen3vl8b_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "started_at": utc_now(),
        "status": "running",
        "collection": config["collection"],
        "chunk_manifest": config["source_chunk_manifest"],
        "chunk_manifest_sha256": config["source_chunk_manifest_sha256"],
        "total_chunks": len(chunks),
        "next_chunk_index": 0,
        "batch_size": batch_size,
        "embedding_model": config["embedding_model"],
        "embedding_model_path": config["embedding_model_path"],
        "document_prompt": DOCUMENT_PROMPT,
        "source_files_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    chunk_path = Path(config["source_chunk_manifest"])
    if sha256(chunk_path) != config["source_chunk_manifest_sha256"]:
        raise RuntimeError("Chunk manifest hash differs from Qdrant configuration")
    chunks = load_chunks(chunk_path)
    if len(chunks) != config["expected_chunk_count"]:
        raise RuntimeError("Chunk count differs from Qdrant configuration")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        safe_resume = (
            state.get("status") == "running"
            and state.get("chunk_manifest_sha256") == config["source_chunk_manifest_sha256"]
            and state.get("total_chunks") == len(chunks)
            and state.get("batch_size") == args.batch_size
        )
        if not safe_resume:
            raise RuntimeError(
                "Existing indexing state is not safely resumable; inspect it before restarting."
            )
    else:
        state = initial_state(config, chunks, args.batch_size)
        write_json_atomic(STATE_PATH, state)

    model_path = Path(config["embedding_model_path"])
    if not model_path.exists():
        raise RuntimeError(f"Embedding model path missing: {model_path}")
    client = QdrantClient(path=config["qdrant_path"])
    if not client.collection_exists(config["collection"]):
        raise RuntimeError("Configured Qdrant collection does not exist")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback for pilot indexing")
    started = perf_counter()
    model = SentenceTransformer(str(model_path), device="cuda:0")
    state["model_loaded_at"] = utc_now()
    write_json_atomic(STATE_PATH, state)

    for start in range(int(state["next_chunk_index"]), len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        texts = [record["text"] for record in batch]
        vectors = model.encode(
            texts,
            prompt=DOCUMENT_PROMPT,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )
        if vectors.shape != (len(batch), int(config["vector_size"])):
            raise RuntimeError(f"Unexpected embedding matrix shape: {vectors.shape}")
        client.upsert(
            collection_name=config["collection"],
            points=[
                make_point(record, vector.tolist())
                for record, vector in zip(batch, vectors, strict=True)
            ],
            wait=True,
        )
        state["next_chunk_index"] = start + len(batch)
        state["updated_at"] = utc_now()
        state["elapsed_seconds"] = round(perf_counter() - started, 3)
        write_json_atomic(STATE_PATH, state)
        if state["next_chunk_index"] % 100 == 0 or state["next_chunk_index"] == len(chunks):
            print(
                f"Indexed {state['next_chunk_index']}/{len(chunks)} chunks",
                flush=True,
            )

    info = client.get_collection(config["collection"])
    point_count = info.points_count
    state.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "point_count": point_count,
            "elapsed_seconds": round(perf_counter() - started, 3),
        }
    )
    write_json_atomic(STATE_PATH, state)
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "collection": config["collection"],
        "expected_chunk_count": len(chunks),
        "point_count": point_count,
        "embedding_model": config["embedding_model"],
        "vector_size": config["vector_size"],
        "document_prompt": DOCUMENT_PROMPT,
        "state": str(STATE_PATH),
        "status": "passed" if point_count == len(chunks) else "review_required",
    }
    write_json_atomic(REPORT_ROOT / "pilot_text_v1_indexing.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    client.close()
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    sys.exit(main())
