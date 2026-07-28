from __future__ import annotations

import gc
import importlib.util
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
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag" / "qdrant_manual_merge_20260726_incremental.json"
REPORT_ROOT = PROJECT_ROOT / "reports" / "retrieval_validation"
QUERY_PROMPT = "Represent the scientific literature question for retrieval."
RERANK_PROMPT = "Retrieve evidence relevant to the scientific literature question."
TOP_K_RECALL = 40
TOP_K_RERANK = 8

QUESTIONS = [
    "What structural evidence explains how HPV16 E6, E6AP and p53 assemble into a degradation complex?",
    "Which newly indexed study describes a small-molecule inhibitor of the HPV E6-p53 interaction?",
    "What is the evidence for a binding pocket on HPV16 E6 that could support rational inhibitor design?",
    "Which structural study discusses the ordered domain of E6AP in an HPV16 E6-p53 complex?",
    "What determinants of HPV16 E6 protein stability are reported in the literature?",
    "How have reactive peptide or covalent peptide approaches been used to target HPV16 E6?",
    "What role do IDO1-expressing LAMP3-positive dendritic cells play in cervical cancer immunity?",
    "What preclinical evidence exists for combined IDO1 and CXCR2 inhibition in cervical cancer?",
    "How are IDO1 and Notch1 linked to radiation response in cervical cancer stem cells?",
    "What does cervical-cancer evidence say about IDO1 and infiltrating CD8-positive T cells?",
    "What is the prognostic relevance of IDO1 expression in cervical cancer?",
    "How did a therapeutic DNA vaccine and gemcitabine perform in an HPV-associated tumor model?",
    "How is the TC-1 mouse model used for HPV-related cancer immunization and immunotherapy?",
    "What components were used in the E7 vaccination, carboplatin/paclitaxel and intravaginal CpG tri-therapy?",
    "What evidence supports mucosal rAd5 therapeutic vaccination against HPV16-positive tumors?",
    "What is reported for mRNA-LNP vaccines encoding HPV16 E6 and E7 in early-intervention tumor models?",
    "How does click chemistry support natural-product-inspired covalent drug discovery?",
    "Which natural-product mechanisms involve Michael acceptor molecules?",
    "What are the main advances and design considerations in covalent drug discovery?",
    "What is the chemical rationale for covalent modifiers based on hetero-Michael addition reactions?",
]


def load_reranker(model_root: Path, torch: Any) -> Any:
    helper = model_root / "scripts" / "qwen3_vl_reranker.py"
    spec = importlib.util.spec_from_file_location("qwen3_vl_reranker", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reranker helper: {helper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Qwen3VLReranker(str(model_root), torch_dtype=torch.bfloat16)


def compact(point: Any) -> dict[str, Any]:
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


def is_manual(item: dict[str, Any]) -> bool:
    return str(item.get("chunk_id", "")).startswith("MANUAL-20260726-")


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    client = QdrantClient(path=config["qdrant_path"])
    count = client.get_collection(config["collection"]).points_count
    expected = 10055
    if count < expected:
        raise RuntimeError(f"Incremental index incomplete: {count}/{expected}")
    import torch
    started = perf_counter()
    embedding = SentenceTransformer(str(config["embedding_model_path"]), device="cuda:0")
    recalls: list[list[dict[str, Any]]] = []
    for question in QUESTIONS:
        vector = embedding.encode(question, prompt=QUERY_PROMPT, normalize_embeddings=True, convert_to_numpy=True)
        response = client.query_points(collection_name=config["collection"], query=vector.tolist(), limit=TOP_K_RECALL, with_payload=True)
        recalls.append([compact(point) for point in response.points])
    del embedding
    gc.collect()
    torch.cuda.empty_cache()

    reranker = load_reranker(PROJECT_ROOT / "models" / "weights" / "qwen3-vl-reranker-8b", torch)
    results = []
    for question, candidates in zip(QUESTIONS, recalls, strict=True):
        scores = reranker.process({"instruction": RERANK_PROMPT, "query": {"text": question}, "documents": [{"text": item["text"]} for item in candidates]})
        ranked = sorted(({**candidate, "reranker_score": float(score)} for candidate, score in zip(candidates, scores, strict=True)), key=lambda item: item["reranker_score"], reverse=True)
        top = ranked[:TOP_K_RERANK]
        results.append({
            "question": question,
            "manual_in_recall_at_40": any(is_manual(item) for item in candidates),
            "manual_in_rerank_at_8": any(is_manual(item) for item in top),
            "top_manual_recall": next((item for item in candidates if is_manual(item)), None),
            "top_evidence": top,
        })
    del reranker
    torch.cuda.empty_cache()
    recall_hits = sum(result["manual_in_recall_at_40"] for result in results)
    rerank_hits = sum(result["manual_in_rerank_at_8"] for result in results)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection": config["collection"],
        "point_count": count,
        "question_count": len(QUESTIONS),
        "top_k_recall": TOP_K_RECALL,
        "top_k_rerank": TOP_K_RERANK,
        "manual_recall_at_40_hits": recall_hits,
        "manual_rerank_at_8_hits": rerank_hits,
        "embedding_model": config["embedding_model"],
        "reranker_model": "Qwen/Qwen3-VL-Reranker-8B",
        "elapsed_seconds": round(perf_counter() - started, 3),
        "results": results,
        "status": "passed" if recall_hits >= 18 and rerank_hits >= 16 else "review_required",
        "scope": "Retrieval mechanics and source traceability only; it does not establish scientific conclusions.",
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    path = REPORT_ROOT / "manual_merge_20260726_retrieval_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))
    client.close()
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    sys.exit(main())
