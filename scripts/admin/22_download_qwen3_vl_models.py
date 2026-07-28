from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path("/data2/lxj/projects/CervixAgent")
MODEL_ROOT = PROJECT_ROOT / "models"
CACHE_ROOT = MODEL_ROOT / "huggingface"
WEIGHTS_ROOT = MODEL_ROOT / "weights"
RUN_ROOT = PROJECT_ROOT / "runs" / "model_downloads"

MODELS = [
    {
        "role": "embedding",
        "repo_id": "Qwen/Qwen3-VL-Embedding-8B",
        "local_name": "qwen3-vl-embedding-8b",
    },
    {
        "role": "reranker",
        "repo_id": "Qwen/Qwen3-VL-Reranker-8B",
        "local_name": "qwen3-vl-reranker-8b",
    },
    {
        "role": "llm",
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
        "local_name": "qwen3-vl-8b-instruct",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = f"qwen3_vl_download_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    state_path = run_dir / "download_state.json"
    state = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "cache_root": str(CACHE_ROOT),
        "weights_root": str(WEIGHTS_ROOT),
        "models": [],
        "status": "running",
        "resumable": True,
    }
    write_json(state_path, state)

    try:
        for spec in MODELS:
            entry = {**spec, "status": "downloading", "started_at": utc_now()}
            state["models"].append(entry)
            write_json(state_path, state)
            print(f"Downloading {spec['repo_id']}...", flush=True)
            snapshot_path = Path(
                snapshot_download(
                    repo_id=spec["repo_id"],
                    cache_dir=str(CACHE_ROOT),
                    max_workers=4,
                )
            )
            stable_path = WEIGHTS_ROOT / spec["local_name"]
            if stable_path.is_symlink() or stable_path.is_file():
                stable_path.unlink()
            elif stable_path.exists():
                raise RuntimeError(
                    f"Refusing to replace non-symlink directory: {stable_path}"
                )
            stable_path.symlink_to(snapshot_path)
            entry.update(
                {
                    "status": "completed",
                    "completed_at": utc_now(),
                    "snapshot_path": str(snapshot_path),
                    "stable_path": str(stable_path),
                    "size_bytes": directory_size(snapshot_path),
                }
            )
            write_json(state_path, state)
            print(
                f"Completed {spec['repo_id']}: {entry['size_bytes']} bytes",
                flush=True,
            )
    except Exception as exc:
        state["status"] = "failed_or_interrupted"
        state["finished_at"] = utc_now()
        state["error"] = repr(exc)
        write_json(state_path, state)
        raise

    state["status"] = "completed"
    state["finished_at"] = utc_now()
    state["total_model_bytes"] = sum(
        int(item.get("size_bytes", 0)) for item in state["models"]
    )
    write_json(state_path, state)
    (RUN_ROOT / "latest_qwen3_vl_download.txt").write_text(
        run_id + "\n", encoding="ascii"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
