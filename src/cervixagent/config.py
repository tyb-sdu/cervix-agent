from __future__ import annotations

import hashlib
import json
from importlib import resources
from typing import Any


def load_workflow() -> dict[str, Any]:
    """Load the packaged, protocol-faithful workflow definition."""
    resource = resources.files("cervixagent").joinpath("resources/workflow.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_public_sources() -> dict[str, Any]:
    """Load the allow-listed public data sources used by the terminal agent."""
    resource = resources.files("cervixagent").joinpath("resources/public_sources.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_compound_ingestion_schema() -> dict[str, Any]:
    """Load the locked P1-02 ingestion contract (not the P1-04 filter rules)."""
    resource = resources.files("cervixagent").joinpath(
        "resources/compound_ingestion_schema.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def workflow_checksum(workflow: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(workflow).encode("utf-8")).hexdigest()
