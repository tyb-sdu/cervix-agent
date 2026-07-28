from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
WEIGHTS_ROOT = PROJECT_ROOT / "models" / "weights"
REPORT_ROOT = PROJECT_ROOT / "reports" / "vl_smoke_tests"

EMBEDDING_MODEL = WEIGHTS_ROOT / "qwen3-vl-embedding-8b"
RERANKER_MODEL = WEIGHTS_ROOT / "qwen3-vl-reranker-8b"
LLM_MODEL = WEIGHTS_ROOT / "qwen3-vl-8b-instruct"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reset_gpu(torch, device_id: int) -> None:
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def gpu_metrics(torch, device_id: int) -> dict[str, float | str]:
    device = torch.device(f"cuda:{device_id}")
    return {
        "device": torch.cuda.get_device_name(device_id),
        "allocated_gib": round(torch.cuda.memory_allocated(device) / 2**30, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(device) / 2**30, 3),
    }


def make_test_image(path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (768, 384), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 744, 360), outline="black", width=4)
    draw.text((60, 110), "CervixAgent multimodal smoke test", fill="black")
    draw.text((60, 190), "HPV16 E6 | IDO1 | literature evidence", fill="black")
    image.save(path)


def result_record(name: str, started: float, status: str, **extra: object) -> dict:
    return {
        "component": name,
        "status": status,
        "duration_seconds": round(perf_counter() - started, 3),
        **extra,
    }


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = f"qwen3_vl_smoke_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = REPORT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    image_path = run_dir / "multimodal_smoke_test.png"
    make_test_image(image_path)

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Expected two CUDA GPUs for the planned smoke test")
    for path in (EMBEDDING_MODEL, RERANKER_MODEL, LLM_MODEL):
        if not path.exists():
            raise RuntimeError(f"Missing downloaded model: {path}")

    report: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": now(),
        "offline_mode": True,
        "model_paths": {
            "embedding": str(EMBEDDING_MODEL),
            "reranker": str(RERANKER_MODEL),
            "llm": str(LLM_MODEL),
        },
        "gpu_plan": {
            "embedding_and_reranker": 0,
            "llm": 1,
            "models_loaded_concurrently": False,
        },
        "components": [],
    }
    components: list[dict] = report["components"]  # type: ignore[assignment]
    text_query = "Which literature evidence discusses HPV16 E6 and IDO1?"
    text_document = (
        "This test record concerns literature retrieval for HPV16 E6, IDO1, "
        "cervical cancer immunity, and natural products."
    )

    started = perf_counter()
    try:
        from sentence_transformers import SentenceTransformer

        reset_gpu(torch, 0)
        model = SentenceTransformer(str(EMBEDDING_MODEL), device="cuda:0")
        vectors = model.encode(
            [text_query, text_document],
            normalize_embeddings=True,
            convert_to_numpy=True,
            prompt="Represent the user's input for scientific literature retrieval.",
        )
        components.append(
            result_record(
                "embedding_text",
                started,
                "passed",
                output_shape=list(vectors.shape),
                gpu=gpu_metrics(torch, 0),
            )
        )
        del vectors, model
        reset_gpu(torch, 0)
    except Exception as exc:
        components.append(
            result_record(
                "embedding_text", started, "failed", error=repr(exc), traceback=traceback.format_exc()
            )
        )

    started = perf_counter()
    try:
        reset_gpu(torch, 0)
        script_path = RERANKER_MODEL / "scripts" / "qwen3_vl_reranker.py"
        spec = importlib.util.spec_from_file_location("qwen3_vl_reranker", script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load official reranker helper: {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.Qwen3VLReranker(
            str(RERANKER_MODEL), torch_dtype=torch.bfloat16
        )
        scores = model.process(
            {
                "instruction": "Retrieve evidence relevant to the scientific literature question.",
                "query": {"text": text_query},
                "documents": [
                    {"text": text_document},
                    {"text": "An unrelated methods note."},
                ],
            }
        )
        components.append(
            result_record(
                "reranker_text",
                started,
                "passed",
                score_count=len(scores),
                scores=[float(value) for value in scores],
                gpu=gpu_metrics(torch, 0),
            )
        )
        del scores, model, module
        reset_gpu(torch, 0)
    except Exception as exc:
        components.append(
            result_record(
                "reranker_text", started, "failed", error=repr(exc), traceback=traceback.format_exc()
            )
        )

    started = perf_counter()
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        reset_gpu(torch, 1)
        processor = AutoProcessor.from_pretrained(str(LLM_MODEL), local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(LLM_MODEL),
            torch_dtype=torch.bfloat16,
            device_map={"": 1},
            local_files_only=True,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {
                        "type": "text",
                        "text": "Read the image. Reply with the exact phrase after the vertical bars.",
                    },
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda:1")
        generated = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        response_ids = generated[:, inputs.input_ids.shape[-1]:]
        response = processor.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        components.append(
            result_record(
                "llm_multimodal",
                started,
                "passed",
                response=response,
                gpu=gpu_metrics(torch, 1),
            )
        )
        del generated, inputs, model, processor
        reset_gpu(torch, 1)
    except Exception as exc:
        components.append(
            result_record(
                "llm_multimodal", started, "failed", error=repr(exc), traceback=traceback.format_exc()
            )
        )

    report["finished_at"] = now()
    report["status"] = "passed" if all(
        component["status"] == "passed" for component in components
    ) else "review_required"
    output = run_dir / "smoke_test_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    sys.exit(main())
