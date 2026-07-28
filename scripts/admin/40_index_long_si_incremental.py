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

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer


ROOT = Path("/data2/lxj/projects/CervixAgent")
CONFIG = ROOT / "configs" / "rag" / "qdrant_manual_merge_20260726_long_si.json"
NAMESPACE = uuid.UUID("65a298df-c1fc-4962-85a7-50e8932a4d1e")
PROMPT = "Represent the scientific literature passage for retrieval."


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=4)
    args.add_argument("--device", default="cuda:1")
    args.add_argument("--config", type=Path, default=CONFIG)
    options = args.parse_args()
    config = json.loads(options.config.read_text(encoding="utf-8"))
    manifest = Path(config["source_chunk_manifest"])
    if digest(manifest) != config["source_chunk_manifest_sha256"]:
        raise RuntimeError("SI chunk manifest checksum mismatch")
    chunks = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(chunks) != config["expected_incremental_chunk_count"]:
        raise RuntimeError("Unexpected SI chunk count")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    client = QdrantClient(path=config["qdrant_path"])
    if not client.collection_exists(config["collection"]):
        raise RuntimeError("Target collection missing")
    state_path, report_path = Path(config["state_path"]), Path(config["report_path"])
    current = client.get_collection(config["collection"]).points_count
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "completed":
            print(json.dumps({"status": "already_completed", "point_count": current}))
            return 0
        if state.get("source_chunk_manifest_sha256") != config["source_chunk_manifest_sha256"]:
            raise RuntimeError("Existing SI index state incompatible")
    else:
        state = {"schema_version": 1, "run_id": "manual_merge_20260726_long_si_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "collection": config["collection"], "source_chunk_manifest_sha256": config["source_chunk_manifest_sha256"], "total_chunks": len(chunks), "next_chunk_index": 0, "batch_size": options.batch_size, "device": options.device, "point_count_before": current, "source_files_modified": False}
        write(state_path, state)
    started = perf_counter()
    model = SentenceTransformer(config["embedding_model_path"], device=options.device)
    keys = ("chunk_id", "selection_id", "document_family_id", "canonical_title", "topic_folder", "source_relative_path", "source_sha256", "source_role", "parsed_output_relative_path", "parsed_output_sha256", "primary_format", "block_kind", "section_title", "block_start_character", "block_end_character", "text", "text_sha256", "chunk_version")
    for start in range(state["next_chunk_index"], len(chunks), options.batch_size):
        batch = chunks[start:start + options.batch_size]
        matrix = model.encode([item["text"] for item in batch], prompt=PROMPT, normalize_embeddings=True, convert_to_numpy=True, batch_size=options.batch_size, show_progress_bar=False)
        if matrix.shape != (len(batch), config["vector_size"]):
            raise RuntimeError(f"Unexpected vector shape {matrix.shape}")
        points = [PointStruct(id=str(uuid.uuid5(NAMESPACE, item["chunk_id"])), vector=vector.tolist(), payload={key: item[key] for key in keys if key in item}) for item, vector in zip(batch, matrix, strict=True)]
        client.upsert(collection_name=config["collection"], points=points, wait=True)
        state.update({"next_chunk_index": start + len(batch), "updated_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(perf_counter() - started, 3)})
        write(state_path, state)
        if state["next_chunk_index"] % 100 == 0 or state["next_chunk_index"] == len(chunks):
            print(f"Indexed {state['next_chunk_index']}/{len(chunks)} SI chunks", flush=True)
    after = client.get_collection(config["collection"]).points_count
    state.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(), "point_count_after": after})
    write(state_path, state)
    report = {"schema_version": 1, "collection": config["collection"], "incremental_chunks": len(chunks), "point_count_before": state["point_count_before"], "point_count_after": after, "expected_minimum_point_count_after": state["point_count_before"] + len(chunks), "embedding_model": config["embedding_model"], "device": options.device, "status": "passed" if after >= state["point_count_before"] + len(chunks) else "review_required"}
    write(report_path, report)
    print(json.dumps(report, ensure_ascii=False))
    client.close()
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    sys.exit(main())
