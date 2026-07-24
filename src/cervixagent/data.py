from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .config import load_public_sources
from .project import load_project


CHUNK_SIZE = 1024 * 1024
USER_AGENT = "CervixAgent/0.1 (public-research-data downloader)"


class DataError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(
    url: str,
    destination: Path,
    *,
    force: bool = False,
    progress: Callable[[int], None] | None = None,
) -> tuple[str, int, bool]:
    """Download atomically and return (sha256, bytes, downloaded_now)."""
    destination = destination.resolve()
    if destination.exists() and not force:
        return sha256_file(destination), destination.stat().st_size, False

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                total += len(chunk)
                if progress is not None:
                    progress(total)
        if total == 0:
            raise DataError(f"下载结果为空：{url}")
        partial.replace(destination)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        partial.unlink(missing_ok=True)
        raise DataError(f"下载失败 {url}：{exc}") from exc
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), total, True


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_download(source_key: str, source: dict[str, Any], path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise DataError(f"{source_key} 文件不存在或为空：{path}")
    if source["kind"] == "structure":
        prefix = path.read_bytes()[:256].lstrip()
        if not prefix.startswith(b"data_"):
            raise DataError(f"{source_key} 不是可识别的 PDBx/mmCIF 文件：{path}")
    if path.suffix.lower() == ".zip" and not zipfile.is_zipfile(path):
        raise DataError(f"{source_key} 不是有效 ZIP 文件：{path}")


def source_keys(selection: str) -> list[str]:
    registry = load_public_sources()
    sources = registry["sources"]
    groups = registry["groups"]
    if selection in groups:
        return list(groups[selection])
    if selection in sources:
        return [selection]
    allowed = ", ".join(sorted([*groups, *sources]))
    raise DataError(f"未知数据源 {selection!r}；可选值：{allowed}")


def fetch_sources(
    project_path: Path,
    selection: str,
    *,
    force: bool = False,
    progress: Callable[[str, int], None] | None = None,
) -> list[dict[str, Any]]:
    root = project_path.expanduser().resolve()
    load_project(root)
    registry = load_public_sources()
    manifest_path = root / "data/raw/download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "files": {}}

    results: list[dict[str, Any]] = []
    for key in source_keys(selection):
        source = registry["sources"][key]
        destination = root / source["relative_path"]
        callback = None
        if progress is not None:
            callback = lambda byte_count, current=key: progress(current, byte_count)
        digest, byte_count, downloaded = download_url(
            source["url"], destination, force=force, progress=callback
        )
        _validate_download(key, source, destination)
        previous_time = manifest.get("files", {}).get(key, {}).get("downloaded_at")
        record = {
            "source_key": key,
            "kind": source["kind"],
            "record_id": source["record_id"],
            "url": source["url"],
            "landing_page": source["landing_page"],
            "version": source["version"],
            "license": source["license"],
            "relative_path": source["relative_path"],
            "bytes": byte_count,
            "sha256": digest,
            "downloaded_at": utc_now() if downloaded else previous_time,
            "verified_at": utc_now(),
            "status": "downloaded" if downloaded else "verified_existing",
        }
        manifest.setdefault("files", {})[key] = record
        manifest["updated_at"] = utc_now()
        _write_json_atomic(manifest_path, manifest)
        results.append(record)
    return results


def iter_lotus_smiles(path: Path) -> Iterator[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("\t") if "\t" in stripped else stripped.split(maxsplit=1)
            smiles = parts[0].strip().strip('"')
            source_id = parts[1].strip().strip('"') if len(parts) > 1 else f"line-{line_number}"
            if line_number == 1 and "smiles" in smiles.lower():
                continue
            if smiles:
                yield source_id, smiles


def _normalise_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _choose_field(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    normalised = {_normalise_header(field): field for field in fieldnames}
    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]
    for key, original in normalised.items():
        if any(candidate in key for candidate in candidates):
            return original
    return None


def _prepend(first: str, remaining: Iterable[str]) -> Iterator[str]:
    yield first
    yield from remaining


def iter_coconut_smiles(path: Path) -> Iterator[tuple[str, str]]:
    if not zipfile.is_zipfile(path):
        raise DataError(f"COCONUT 输入不是有效 ZIP：{path}")
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
            and Path(name).suffix.lower() in {".tsv", ".csv", ".txt"}
        ]
        if not members:
            raise DataError(f"COCONUT ZIP 中没有 TSV/CSV/TXT 文件：{path}")
        member = sorted(members, key=lambda name: ("drug" not in name.lower(), len(name)))[0]
        with archive.open(member) as binary:
            text_stream = io.TextIOWrapper(
                binary, encoding="utf-8-sig", errors="replace", newline=""
            )
            first_line = text_stream.readline()
            delimiter = "\t" if first_line.count("\t") >= first_line.count(",") else ","
            rows = csv.DictReader(_prepend(first_line, text_stream), delimiter=delimiter)
            fieldnames = list(rows.fieldnames or [])
            smiles_field = _choose_field(
                fieldnames,
                ("canonicalsmiles", "smiles", "uniquesmiles", "originalsmiles"),
            )
            id_field = _choose_field(
                fieldnames,
                ("coconutid", "identifier", "id", "inchikey"),
            )
            if smiles_field is None:
                raise DataError(f"无法识别 COCONUT SMILES 列；实际表头：{fieldnames}")
            for row_number, row in enumerate(rows, start=2):
                smiles = (row.get(smiles_field) or "").strip()
                source_id = (row.get(id_field) or "").strip() if id_field else ""
                if smiles:
                    yield source_id or f"row-{row_number}", smiles


def _reservoir_sample(
    records: Iterable[tuple[str, str]], size: int, rng: random.Random
) -> list[tuple[str, str]]:
    reservoir: list[tuple[str, str]] = []
    for index, record in enumerate(records):
        if index < size:
            reservoir.append(record)
        else:
            replacement = rng.randint(0, index)
            if replacement < size:
                reservoir[replacement] = record
    rng.shuffle(reservoir)
    return reservoir


def build_test_dataset(
    project_path: Path, *, size: int = 500, seed: int = 20260715
) -> dict[str, Any]:
    if size < 2:
        raise DataError("测试集大小必须至少为 2")
    root = project_path.expanduser().resolve()
    load_project(root)
    registry = load_public_sources()["sources"]
    coconut_path = root / registry["coconut_drug_discovery"]["relative_path"]
    lotus_path = root / registry["lotus_smiles"]["relative_path"]
    missing = [str(path) for path in (coconut_path, lotus_path) if not path.exists()]
    if missing:
        raise DataError(
            "缺少天然产物原始文件，请先运行 data fetch natural-products："
            + ", ".join(missing)
        )

    rng = random.Random(seed)
    reservoir_size = max(size * 3, 1000)
    candidates = {
        "COCONUT": _reservoir_sample(
            iter_coconut_smiles(coconut_path), reservoir_size, rng
        ),
        "LOTUS": _reservoir_sample(iter_lotus_smiles(lotus_path), reservoir_size, rng),
    }
    selected: list[tuple[str, str, str]] = []
    seen_smiles: set[str] = set()
    while len(selected) < size and any(candidates.values()):
        made_progress = False
        for source_name in ("COCONUT", "LOTUS"):
            bucket = candidates[source_name]
            while bucket:
                source_id, smiles = bucket.pop()
                if smiles not in seen_smiles:
                    seen_smiles.add(smiles)
                    selected.append((source_name, source_id, smiles))
                    made_progress = True
                    break
            if len(selected) >= size:
                break
        if not made_progress:
            break
    if len(selected) < size:
        raise DataError(
            f"公开数据中只得到 {len(selected)} 条不重复 SMILES，无法生成 {size} 条测试集"
        )

    output_path = root / f"data/processed/test_compounds_{size}.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["record_id", "source", "source_id", "smiles"])
        for index, (source_name, source_id, smiles) in enumerate(selected, start=1):
            writer.writerow([f"TEST-{index:04d}", source_name, source_id, smiles])
    temporary.replace(output_path)

    source_counts = {
        source_name: sum(1 for source, _, _ in selected if source == source_name)
        for source_name in ("COCONUT", "LOTUS")
    }
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "purpose": "工程联调测试集，不是原方案的正式筛选库或科学结论",
        "chemical_validation": "not_performed; RDKit 尚未接入",
        "selection": "固定随机种子的水库抽样；按原始 SMILES 精确去重；尽量平衡两个来源",
        "seed": seed,
        "record_count": len(selected),
        "source_counts": source_counts,
        "inputs": {
            "COCONUT": {
                "relative_path": str(coconut_path.relative_to(root)),
                "sha256": sha256_file(coconut_path),
            },
            "LOTUS": {
                "relative_path": str(lotus_path.relative_to(root)),
                "sha256": sha256_file(lotus_path),
            },
        },
        "output": {
            "relative_path": str(output_path.relative_to(root)),
            "sha256": sha256_file(output_path),
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    _write_json_atomic(manifest_path, manifest)
    return manifest
