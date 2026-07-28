from __future__ import annotations

import importlib.util
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag" / "qdrant_pilot_text_v1.json"
REPORT_ROOT = PROJECT_ROOT / "reports" / "retrieval_validation"
QUERY_PROMPT = "Represent the scientific literature question for retrieval."
RERANK_PROMPT = "Retrieve evidence relevant to the scientific literature question."
TOP_K_RECALL = 40
TOP_K_RERANK = 8

QUESTIONS = [
    "HPV16 E6 的 Cys51 和 Cys58 与共价抑制策略之间有哪些文献证据？",
    "IDO1 在宫颈癌免疫微环境中发挥什么作用，现有文献证据是什么？",
    "TC-1 小鼠模型在 HPV 相关肿瘤免疫研究中如何使用？",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_official_reranker(model_root: Path, torch: Any) -> Any:
    script_path = model_root / "scripts" / "qwen3_vl_reranker.py"
    spec = importlib.util.spec_from_file_location("qwen3_vl_reranker", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reranker helper: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Qwen3VLReranker(str(model_root), torch_dtype=torch.bfloat16)


def selected_payload(point: Any) -> dict[str, Any]:
    payload = point.payload or {}
    return {
        "chunk_id": payload.get("chunk_id"),
        "canonical_title": payload.get("canonical_title"),
        "topic_folder": payload.get("topic_folder"),
        "section_title": payload.get("section_title"),
        "source_relative_path": payload.get("source_relative_path"),
        "text": payload.get("text"),
        "vector_score": float(point.score),
    }


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=config["qdrant_path"])
    info = client.get_collection(config["collection"])
    if info.points_count != config["expected_chunk_count"]:
        raise RuntimeError(
            f"Index incomplete: {info.points_count}/{config['expected_chunk_count']} points"
        )

    import torch

    embedding_path = Path(config["embedding_model_path"])
    reranker_path = PROJECT_ROOT / "models" / "weights" / "qwen3-vl-reranker-8b"
    started = perf_counter()
    embedding_model = SentenceTransformer(str(embedding_path), device="cuda:0")
    validation: list[dict[str, Any]] = []
    recalls: list[list[dict[str, Any]]] = []
    for question in QUESTIONS:
        vector = embedding_model.encode(
            question,
            prompt=QUERY_PROMPT,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        response = client.query_points(
            collection_name=config["collection"],
            query=vector.tolist(),
            limit=TOP_K_RECALL,
            with_payload=True,
        )
        recalls.append([selected_payload(point) for point in response.points])
    del embedding_model, vector
    gc.collect()
    torch.cuda.empty_cache()

    reranker = load_official_reranker(reranker_path, torch)
    for question, candidates in zip(QUESTIONS, recalls, strict=True):
        scores = reranker.process(
            {
                "instruction": RERANK_PROMPT,
                "query": {"text": question},
                "documents": [{"text": item["text"]} for item in candidates],
            }
        )
        reranked = sorted(
            (
                {**candidate, "reranker_score": float(score)}
                for candidate, score in zip(candidates, scores, strict=True)
            ),
            key=lambda item: item["reranker_score"],
            reverse=True,
        )[:TOP_K_RERANK]
        validation.append(
            {
                "question": question,
                "retrieval_count": len(candidates),
                "reranked_count": len(reranked),
                "top_evidence": reranked,
            }
        )
    del reranker
    torch.cuda.empty_cache()
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "collection": config["collection"],
        "point_count": info.points_count,
        "expected_point_count": config["expected_chunk_count"],
        "embedding_model": config["embedding_model"],
        "reranker_model": "Qwen/Qwen3-VL-Reranker-8B",
        "query_prompt": QUERY_PROMPT,
        "reranker_prompt": RERANK_PROMPT,
        "top_k_recall": TOP_K_RECALL,
        "top_k_rerank": TOP_K_RERANK,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "results": validation,
        "status": "ready_for_human_evidence_review",
        "note": "This verifies retrieval and reranking mechanics only; scientific conclusions still require evidence review.",
    }
    output = REPORT_ROOT / f"pilot_retrieval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "results": "written_to_report"}, ensure_ascii=False, indent=2))
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
