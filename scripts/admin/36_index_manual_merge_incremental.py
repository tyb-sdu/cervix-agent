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
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag" / "qdrant_manual_merge_20260726_incremental.json"
NAMESPACE = uuid.UUID("65a298df-c1fc-4962-85a7-50e8932a4d1e")
DOCUMENT_PROMPT = "Represent the scientific literature passage for retrieval."


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        chunks = [json.loads(line) for line in stream if line.strip()]
    if not all(item.get("text", "").strip() for item in chunks):
        raise RuntimeError("Chunk manifest contains empty text")
    return chunks


def point(record: dict[str, Any], vector: list[float]) -> PointStruct:
    payload_keys = (
        "chunk_id", "selection_id", "document_family_id", "canonical_title",
        "topic_folder", "source_relative_path", "source_sha256",
        "parsed_output_relative_path", "parsed_output_sha256", "primary_format",
        "block_kind", "section_title", "block_start_character",
        "block_end_character", "text", "text_sha256", "chunk_version",
    )
    return PointStruct(
        id=str(uuid.uuid5(NAMESPACE, record["chunk_id"])),
        vector=vector,
        payload={key: record[key] for key in payload_keys},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    chunk_path = Path(config["source_chunk_manifest"])
    if sha256(chunk_path) != config["source_chunk_manifest_sha256"]:
        raise RuntimeError("Chunk manifest checksum differs from approved configuration")
    chunks = load_chunks(chunk_path)
    if len(chunks) != config["expected_incremental_chunk_count"]:
        raise RuntimeError("Unexpected incremental chunk count")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU indexing")
    if args.device.startswith("cuda:"):
        device_id = int(args.device.split(":", 1)[1])
        if device_id >= torch.cuda.device_count():
            raise RuntimeError(f"Requested device {args.device} does not exist")

    client = QdrantClient(path=config["qdrant_path"])
    if not client.collection_exists(config["collection"]):
        raise RuntimeError("Pilot collection is missing; refusing to create a different collection")
    state_path = Path(config["state_path"])
    report_path = Path(config["report_path"])
    current_count = client.get_collection(config["collection"]).points_count
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        compatible = (
            state.get("source_chunk_manifest_sha256") == config["source_chunk_manifest_sha256"]
            and state.get("total_chunks") == len(chunks)
            and state.get("batch_size") == args.batch_size
            and state.get("device") == args.device
            and state.get("collection") == config["collection"]
        )
        if not compatible:
            raise RuntimeError("Existing incremental state is incompatible; inspect before retrying")
        if state.get("status") == "completed":
            print(json.dumps({"status": "already_completed", "point_count": current_count}, ensure_ascii=False))
            return 0
    else:
        state = {
            "schema_version": 1,
            "run_id": "manual_merge_20260726_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "status": "running",
            "started_at": now(),
            "collection": config["collection"],
            "source_chunk_manifest": config["source_chunk_manifest"],
            "source_chunk_manifest_sha256": config["source_chunk_manifest_sha256"],
            "total_chunks": len(chunks),
            "next_chunk_index": 0,
            "batch_size": args.batch_size,
            "device": args.device,
            "point_count_before": current_count,
            "source_files_modified": False,
        }
        write_json(state_path, state)

    started = perf_counter()
    model = SentenceTransformer(str(config["embedding_model_path"]), device=args.device)
    state["model_loaded_at"] = now()
    write_json(state_path, state)
    for start in range(int(state["next_chunk_index"]), len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        vectors = model.encode(
            [item["text"] for item in batch],
            prompt=DOCUMENT_PROMPT,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )
        if vectors.shape != (len(batch), int(config["vector_size"])):
            raise RuntimeError(f"Unexpected vector shape {vectors.shape}")
        client.upsert(
            collection_name=config["collection"],
            points=[point(item, vector.tolist()) for item, vector in zip(batch, vectors, strict=True)],
            wait=True,
        )
        state["next_chunk_index"] = start + len(batch)
        state["updated_at"] = now()
        state["elapsed_seconds"] = round(perf_counter() - started, 3)
        write_json(state_path, state)
        if state["next_chunk_index"] % 100 == 0 or state["next_chunk_index"] == len(chunks):
            print(f"Indexed {state['next_chunk_index']}/{len(chunks)} chunks", flush=True)

    after_count = client.get_collection(config["collection"]).points_count
    state.update({"status": "completed", "completed_at": now(), "point_count_after": after_count, "elapsed_seconds": round(perf_counter() - started, 3)})
    write_json(state_path, state)
    report = {
        "schema_version": 1,
        "collection": config["collection"],
        "incremental_chunks": len(chunks),
        "point_count_before": state["point_count_before"],
        "point_count_after": after_count,
        "expected_minimum_point_count_after": state["point_count_before"] + len(chunks),
        "embedding_model": config["embedding_model"],
        "device": args.device,
        "status": "passed" if after_count >= state["point_count_before"] + len(chunks) else "review_required",
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    client.close()
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    sys.exit(main())
