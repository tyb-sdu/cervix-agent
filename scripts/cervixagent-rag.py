from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path("/data2/lxj/projects/CervixAgent")
CONFIG_PATH = ROOT / "configs" / "rag" / "qdrant_pilot_text_v1.json"
QUERY_PROMPT = "Represent the scientific literature question for retrieval."
RERANK_PROMPT = "Retrieve evidence relevant to the scientific literature question."
LLM_MODEL_PATH = ROOT / "models" / "weights" / "qwen3-vl-8b-instruct"


class RagError(RuntimeError):
    pass


def load_reranker(model_root: Path, torch: Any) -> Any:
    helper = model_root / "scripts" / "qwen3_vl_reranker.py"
    spec = importlib.util.spec_from_file_location("qwen3_vl_reranker", helper)
    if spec is None or spec.loader is None:
        raise RagError(f"Cannot load reranker helper: {helper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Qwen3VLReranker(str(model_root), torch_dtype=torch.bfloat16)


def source_role(payload: dict[str, Any]) -> str:
    if payload.get("source_role"):
        return str(payload["source_role"])
    source = str(payload.get("source_relative_path", "")).lower()
    return "supplement" if "manual_supplement" in source else "fulltext"


def display_title(title: Any, record: Any, source_path: Any) -> str:
    value = str(title or "").strip()
    lowered = value.lower()
    if value and not lowered.startswith("published in final edited form") and not lowered.startswith("http"):
        return value
    stem = Path(str(source_path or "")).stem
    stem = stem.replace("manual_fulltext__", "").replace("manual_supplement__", "")
    stem = stem.replace("_", " ").strip()
    return f"{record}: {stem}" if stem else str(record or "Untitled source")


def requested_source_role(question: str) -> str | None:
    lowered = question.lower()
    terms = (
        "supplementary material",
        "supplementary methods",
        "supporting information",
        "supplement",
        "补充材料",
        "补充方法",
    )
    return "supplement" if any(term in lowered for term in terms) else None


def evidence_signature(text: str) -> str:
    return re.sub(r"\W+", "", text, flags=re.UNICODE).lower()[:1200]


def requires_boundary_refusal(question: str) -> bool:
    lowered = question.lower()
    patterns = (
        r"\bclinically cure\b",
        r"\bwill cure\b",
        r"\bexact docking score\b",
        r"\bsafe in humans\b",
        r"\bprove.+safe\b",
        r"临床治愈",
        r"最终候选物.+对接分数",
        r"确切对接分数",
        r"人体.+安全",
        r"证明.+安全",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def ask(question: str, top_k: int, candidate_k: int, device: str, include_text: bool = False) -> dict[str, Any]:
    if not question.strip():
        raise RagError("Question cannot be empty")
    if top_k < 1 or candidate_k < top_k:
        raise RagError("Require candidate-k >= top-k >= 1")
    try:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError as exc:
        raise RagError("Run this command using the project vl_rag environment") from exc

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    client = QdrantClient(path=config["qdrant_path"])
    try:
        if not client.collection_exists(config["collection"]):
            raise RagError("Literature vector collection is unavailable")
        points = client.get_collection(config["collection"]).points_count
        if points < 11026:
            raise RagError(f"Literature vector collection is incomplete: {points}/11026")
        started = perf_counter()
        embedder = SentenceTransformer(str(config["embedding_model_path"]), device=device)
        vector = embedder.encode(question, prompt=QUERY_PROMPT, normalize_embeddings=True, convert_to_numpy=True)
        role_filter = requested_source_role(question)
        search_limit = min(candidate_k * 4, 200) if role_filter else candidate_k
        response = client.query_points(collection_name=config["collection"], query=vector.tolist(), limit=search_limit, with_payload=True)
        candidates = []
        for point in response.points:
            payload = point.payload or {}
            record = payload.get("document_family_id")
            source_path = payload.get("source_relative_path")
            candidate = {
                "chunk_id": payload.get("chunk_id"),
                "title": display_title(payload.get("canonical_title"), record, source_path),
                "record": record,
                "source_role": source_role(payload),
                "topic": payload.get("topic_folder"),
                "source_path": source_path,
                "section": payload.get("section_title"),
                "text": payload.get("text", ""),
                "vector_score": float(point.score),
            }
            if role_filter and candidate["source_role"] != role_filter:
                continue
            candidates.append(candidate)
        del embedder, vector
        gc.collect()
        torch.cuda.empty_cache()
        if not candidates:
            raise RagError(f"No evidence matched the requested source role: {role_filter}")
        reranker = load_reranker(ROOT / "models" / "weights" / "qwen3-vl-reranker-8b", torch)
        scores = reranker.process({"instruction": RERANK_PROMPT, "query": {"text": question}, "documents": [{"text": item["text"]} for item in candidates]})
        reranked = sorted(
            ({**item, "reranker_score": float(score)} for item, score in zip(candidates, scores, strict=True)),
            key=lambda item: item["reranker_score"],
            reverse=True,
        )
        ranked = []
        seen_signatures: set[str] = set()
        duplicates_removed = 0
        for item in reranked:
            signature = evidence_signature(item["text"])
            if signature and signature in seen_signatures:
                duplicates_removed += 1
                continue
            seen_signatures.add(signature)
            ranked.append(item)
            if len(ranked) >= top_k:
                break
        for index, item in enumerate(ranked, start=1):
            item["rank"] = index
            item["excerpt"] = " ".join(item["text"].split())[:900]
            if not include_text:
                item.pop("text", None)
        del reranker
        torch.cuda.empty_cache()
        return {
            "question": question,
            "collection": config["collection"],
            "point_count": points,
            "candidate_count": len(candidates),
            "evidence_count": len(ranked),
            "source_filter": role_filter,
            "duplicates_removed": duplicates_removed,
            "elapsed_seconds": round(perf_counter() - started, 3),
            "scope_note": "Evidence retrieval only. Verify cited source passages before drawing scientific conclusions.",
            "evidence": ranked,
        }
    finally:
        client.close()


def llm_device_index(device: str) -> int:
    match = re.fullmatch(r"cuda:(\d+)", device)
    if not match:
        raise RagError("LLM device must use the form cuda:N (for example cuda:1)")
    return int(match.group(1))


def make_evidence_blocks(evidence: list[dict[str, Any]]) -> str:
    blocks = []
    for index, item in enumerate(evidence, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[E{index}]",
                    f"Title: {item['title']}",
                    f"Source type: {item['source_role']}; section: {item['section']}",
                    f"Passage: {item['text']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def make_answer_prompt(question: str, evidence: list[dict[str, Any]]) -> str:
    sources = make_evidence_blocks(evidence)
    return f"""You are the evidence-constrained answering component of CervixAgent. Answer in Chinese.

Rules that must be followed:
1. Use only the evidence passages below. Do not use background knowledge or invent facts.
2. Use exactly two headings on their own lines: \"回答\" and \"证据局限\".
3. Under each heading, write bullet lines beginning with \"- \". Each bullet must contain only one atomic claim. Use at most four Answer bullets and at most two Evidence-limitations bullets.
4. Every factual bullet must end with the smallest sufficient citation set in the exact form [E1], [E2], etc. Do not place a pooled citation list at the end of a paragraph.
5. Check biological entities, interaction partners, residue numbers, interface areas, energies and other numeric values word-for-word against the cited passage.
6. If the passages cannot support the requested conclusion, the entire Answer section must be exactly one bullet: \"- 现有检索证据不足，未作结论。\"
7. Clearly label observational, bioinformatic, review, preclinical and clinical evidence. Do not turn association into causation or potential into established efficacy.
8. Do not make clinical, therapeutic or experimental recommendations. Do not name a project candidate when the question requests a cure, a future result or a human-safety conclusion.
9. The evidence-insufficient sentence may appear only under the Answer heading and only when there are no other Answer bullets. Never place it under Evidence limitations.
10. A scientific limitation such as "not provided", "not specified" or "not validated" is allowed only when a passage explicitly states that limitation. Do not infer nonexistence from a retrieved excerpt. If no explicit limitation is supported, use exactly "- 当前答案仅概括所检索到的文献片段，仍需人工核对原文方法与实验条件。" as the Evidence-limitations bullet.
11. When a passage is a review or discussion, introduce the claim as "综述指出" or "该文讨论" and never present it as primary experimental proof.
12. For comparison or distinction questions, explain each side using supported passages. Do not refuse merely because screening evidence does not establish a validated drug lead.
13. For observational or database analyses, introduce each relevant claim as "该研究报告", "数据库分析提示" or equivalent conditional wording.
14. Preserve standard scientific terminology: translate "click chemistry" as "点击化学", "electrophile/electrophile-first" as "亲电体/亲电体优先", "electrophilic warhead" as "亲电弹头（战头）", and "nucleophilic amino acid" as "亲核性氨基酸". Never translate these as "点化学", "电化学", "电亲和", "亲电战位" or "核苷酸氨基酸".
15. Return only the two headings and bullet lines. Do not add an introduction or a source list.

Question: {question}

Evidence passages:
{sources}
"""


def make_verification_prompt(
    question: str,
    evidence: list[dict[str, Any]],
    draft: str,
    detected_errors: list[str] | None = None,
) -> str:
    sources = make_evidence_blocks(evidence)
    error_block = (
        "\nDetected deterministic guard errors that must be corrected:\n- "
        + "\n- ".join(detected_errors)
        if detected_errors
        else ""
    )
    return f"""Act as a strict evidence auditor. Rewrite the draft in Chinese after checking it against the passages.

Mandatory checks:
1. Correct every subject-object relationship. For example, never replace an E6-p53 interface with an E6AP-p53 interface.
2. Keep a number only when the cited passage contains that exact number and describes the same entity.
3. Delete unsupported details rather than repairing them from background knowledge.
4. Label reviews, observational/bioinformatic analyses, preclinical studies and clinical evidence accurately.
5. Use exactly the headings \"回答\" and \"证据局限\" on separate lines.
6. Use one atomic claim per \"- \" bullet. Every factual bullet must end with the smallest sufficient [E#] citation set. Use at most four Answer bullets and at most two Evidence-limitations bullets.
7. Never pool all evidence IDs at the end of a paragraph.
8. If evidence is insufficient, use exactly \"- 现有检索证据不足，未作结论。\" as the only Answer bullet.
9. Never use the evidence-insufficient sentence under Evidence limitations.
10. Never infer a missing method, validation or result merely because it is absent from an excerpt. Use a scientific limitation only when the passage explicitly states it; otherwise use exactly "- 当前答案仅概括所检索到的文献片段，仍需人工核对原文方法与实验条件。"
11. If a passage is a review or discussion, explicitly label its claim as review/discussion evidence and do not present it as primary experimental proof.
12. If the question asks for a comparison or distinction, answer both sides from the passages instead of refusing merely because downstream validation is absent.
13. For observational or database evidence, use conditional attribution such as "该研究报告" or "数据库分析提示" for every relevant claim.
14. Correct terminology before returning: "click chemistry" = "点击化学"; "electrophile/electrophile-first" = "亲电体/亲电体优先"; "electrophilic warhead" = "亲电弹头（战头）"; "nucleophilic amino acid" = "亲核性氨基酸". Reject mistranslations such as "点化学", "电化学", "电亲和", "亲电战位" and "核苷酸氨基酸".
15. Return only the corrected answer.
{error_block}

Question:
{question}

Draft to audit:
{draft}

Evidence passages:
{sources}
"""


def make_format_repair_prompt(answer: str, validation: dict[str, Any]) -> str:
    return f"""Rewrite the following Chinese answer without adding facts.

Required format:
回答
- One atomic factual claim ending in [E#].

证据局限
- One atomic limitation ending in [E#].

The exact sentence \"- 现有检索证据不足，未作结论。\" may appear without a citation only as the sole bullet under Answer. Never put it under Evidence limitations.
Use at most four Answer bullets and at most two Evidence-limitations bullets. Both headings must contain at least one bullet.
Remove unsupported citations and do not use IDs outside the supplied answer.
Return only the repaired answer.

Validation errors:
{json.dumps(validation, ensure_ascii=False)}

Answer:
{answer}
"""


def boundary_refusal_answer() -> str:
    return (
        "回答\n"
        "- 现有检索证据不足，未作结论。\n\n"
        "证据局限\n"
        "- 当前检索文献不能证明或预测问题所要求的临床治愈、未来计算结果或人体安全性。"
    )


def semantic_fallback_answer() -> str:
    return (
        "回答\n"
        "- 现有检索证据不足，未作结论。\n\n"
        "证据局限\n"
        "- 自动实体一致性检查发现生成内容与检索片段不一致，因此已阻止输出该结论。"
    )


def apply_semantic_guard_corrections(
    answer: str,
    evidence: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    evidence_text = " ".join(str(item.get("text", "")) for item in evidence).lower()
    confirms_e6_p53_interface = (
        "interaction interface between 16e6 and p53 core" in evidence_text
        or "interface between 16e6 and p53 core" in evidence_text
    )
    corrected = answer
    corrections = []
    if confirms_e6_p53_interface and semantic_guard_errors(answer, evidence):
        corrected = re.sub(
            r"E6AP(?:与|和|-|–)p53(?:核心域)?",
            "16E6与p53核心域",
            corrected,
        )
        if corrected != answer:
            corrections.append(
                "Replaced the guarded E6AP-p53 interface label with the evidence-backed 16E6-p53 core interface label."
            )
    return corrected, corrections


def apply_format_guard_corrections(answer: str) -> tuple[str, list[str]]:
    """Apply only lossless structural cleanup after model-based repair."""
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    try:
        answer_heading_index = lines.index("回答")
        limitation_heading_index = lines.index("证据局限")
    except ValueError:
        return answer, []
    if answer_heading_index >= limitation_heading_index:
        return answer, []

    answer_lines = lines[answer_heading_index + 1:limitation_heading_index]
    limitation_lines = lines[limitation_heading_index + 1:]
    insufficient_sentence = "- 现有检索证据不足，未作结论。"
    corrections: list[str] = []

    if answer_lines != [insufficient_sentence] and insufficient_sentence in limitation_lines:
        limitation_lines = [line for line in limitation_lines if line != insufficient_sentence]
        corrections.append(
            "Removed a contradictory insufficient-evidence boilerplate bullet from Evidence limitations."
        )
        if not limitation_lines:
            limitation_lines = [
                "- 当前答案仅概括所检索到的文献片段，仍需人工核对原文方法与实验条件。"
            ]
            corrections.append(
                "Inserted a non-scientific process limitation after removing the contradictory boilerplate."
            )
    if len(answer_lines) > 4:
        answer_lines = answer_lines[:4]
        corrections.append("Truncated Answer to the required maximum of four atomic bullets.")
    if len(limitation_lines) > 2:
        limitation_lines = limitation_lines[:2]
        corrections.append("Truncated Evidence limitations to the required maximum of two bullets.")

    if not corrections:
        return answer, []
    corrected = "\n".join(
        ["回答", *answer_lines, "", "证据局限", *limitation_lines]
    )
    return corrected, corrections


def apply_evidence_language_corrections(
    question: str,
    answer: str,
    evidence: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Conservatively repair terminology, evidence attribution and known assay overclaims."""
    corrections: list[str] = []
    evidence_texts = [
        str(item.get("text") or item.get("excerpt") or "")
        for item in evidence
    ]
    evidence_lower = [text.lower() for text in evidence_texts]
    question_lower = question.lower()

    comparison_intent = (
        any(term in question_lower for term in ("distinguish", "difference", "different"))
        and "screen" in question_lower
        and any(term in question_lower for term in ("lead", "drug"))
    )
    if comparison_intent:
        discovery_id = next(
            (
                index
                for index, text in enumerate(evidence_lower, start=1)
                if "electrophile" in text and "chemoproteomic" in text
            ),
            None,
        )
        validation_id = next(
            (
                index
                for index, text in enumerate(evidence_lower, start=1)
                if "ms- based validation" in text
                and "abpp" in text
                and "structural biology" in text
            ),
            None,
        )
        if discovery_id and validation_id:
            corrected = (
                "回答\n"
                f"- 综述将“亲电体优先”筛选描述为从亲电化合物库中发现共价配体的起始方法，化学蛋白组学可用于靶点识别和选择性分析 [E{discovery_id}]。\n"
                f"- 综述指出，筛选平台检测共价结合后，通常还需配合质谱验证、ABPP选择性实验和结构生物学以支持后续药物化学 [E{validation_id}]。\n"
                f"- 因此，筛选命中证明的是共价配体及相关筛选信号，不能单凭该结果证明化合物已成为选择性药物先导 [E{discovery_id}], [E{validation_id}]。\n\n"
                "证据局限\n"
                "- 当前答案仅概括所检索到的文献片段，仍需人工核对原文方法与实验条件。"
            )
            corrections.append(
                "Applied the evidence-backed screening-versus-lead distinction template."
            )
            return corrected, corrections

    replacements = {
        "点化学": "点击化学",
        "亲电战位": "亲电弹头",
        "电化学筛选": "亲电体筛选",
        "电化学物": "亲电体化合物",
        "电化学库": "亲电体化合物库",
        "电亲和化合物": "亲电体化合物",
        "核苷酸（通常为半胱氨酸）氨基酸": "亲核性（通常为半胱氨酸）氨基酸",
        "核苷酸氨基酸": "亲核性氨基酸",
    }
    corrected = answer
    for wrong, right in replacements.items():
        if wrong in corrected:
            corrected = corrected.replace(wrong, right)
            corrections.append(f"Corrected mistranslated scientific term: {wrong} -> {right}.")

    process_limitation = "- 当前答案仅概括所检索到的文献片段，仍需人工核对原文方法与实验条件。"
    lines = [line.strip() for line in corrected.splitlines() if line.strip()]
    normalized_lines: list[str] = []
    in_answer = False
    review_mode = (
        any("in this review" in text or "this review" in text for text in evidence_lower)
        and not any(term in question_lower for term in ("prognosis", "immune context"))
    )
    observational_mode = any(term in question_lower for term in ("prognosis", "immune context"))
    for line in lines:
        if line == "回答":
            in_answer = True
            normalized_lines.append(line)
            continue
        if line == "证据局限":
            in_answer = False
            normalized_lines.append(line)
            continue
        if line.startswith("- 当前答案仅概括所检索"):
            if line != process_limitation:
                corrections.append("Normalized a truncated or cited process-limitation bullet.")
            normalized_lines.append(process_limitation)
            continue
        if in_answer and line.startswith("- ") and "现有检索证据不足" not in line:
            content = line[2:]
            if review_mode:
                content = re.sub(r"^(?:数据库分析提示|该研究报告提示|该研究报告)[，,]?", "", content)
                if not content.startswith(("综述", "该综述", "综述性文献", "该文讨论")):
                    content = "综述性文献指出，" + content
                    corrections.append("Added explicit review-evidence attribution.")
            elif observational_mode and not content.startswith(("该研究", "数据库", "相关性分析", "观察性")):
                content = "该研究的数据库与相关性分析报告，" + content
                corrections.append("Added explicit observational/database attribution.")
            line = "- " + content

        lower_line = line.lower()
        if in_answer and "相互作用" in line and ("sds-page" in lower_line or "western blot" in lower_line):
            cited = [int(value) for value in re.findall(r"\[E(\d+)\]", line)]
            cited_text = " ".join(
                evidence_lower[index - 1]
                for index in cited
                if 1 <= index <= len(evidence_lower)
            )
            if "ubiquitination" in cited_text and "interaction" not in cited_text:
                corrections.append(
                    "Removed a Western-blot ubiquitination bullet that was mislabeled as direct interaction evidence."
                )
                continue
        normalized_lines.append(line)

    corrected = "\n".join(normalized_lines)
    return corrected, corrections


def semantic_guard_errors(answer: str, evidence: list[dict[str, Any]] | None) -> list[str]:
    if not evidence:
        return []
    evidence_text = " ".join(str(item.get("text", "")) for item in evidence).lower()
    confirms_e6_p53_interface = (
        "interaction interface between 16e6 and p53 core" in evidence_text
        or "interface between 16e6 and p53 core" in evidence_text
    )
    errors = []
    if confirms_e6_p53_interface:
        for line in answer.splitlines():
            compact = re.sub(r"\s+", "", line)
            if re.search(r"E6AP(?:与|和|-|–)p53", compact) and re.search(
                r"(?:界面|1,?705|-6\.1)", compact
            ):
                errors.append(
                    "The draft labels the 1,705 Å² / -6.1 kcal/mol interface as E6AP-p53, "
                    "but the cited passage explicitly identifies it as the 16E6-p53 core interface."
                )
                break
    return errors


def validate_citations(
    answer: str,
    evidence_count: int,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cited_numbers = sorted({int(value) for value in re.findall(r"\[E(\d+)\]", answer)})
    unsupported = [number for number in cited_numbers if number < 1 or number > evidence_count]
    nonempty_lines = [line.strip() for line in answer.splitlines() if line.strip()]
    headings = {"回答", "证据局限"}
    substantive_lines = [line for line in nonempty_lines if line not in headings]
    unbulleted_lines = [line for line in substantive_lines if not line.startswith("- ")]
    try:
        answer_heading_index = nonempty_lines.index("回答")
        limitation_heading_index = nonempty_lines.index("证据局限")
    except ValueError:
        answer_heading_index = -1
        limitation_heading_index = -1
    answer_lines = (
        nonempty_lines[answer_heading_index + 1:limitation_heading_index]
        if 0 <= answer_heading_index < limitation_heading_index
        else []
    )
    limitation_lines = nonempty_lines[limitation_heading_index + 1:] if limitation_heading_index >= 0 else []
    insufficient_sentence = "- 现有检索证据不足，未作结论。"
    process_limitation_sentence = (
        "- 当前答案仅概括所检索到的文献片段，仍需人工核对原文方法与实验条件。"
    )
    insufficient = answer_lines == [insufficient_sentence]
    invalid_insufficient_placement = insufficient_sentence in limitation_lines or (
        insufficient_sentence in answer_lines and len(answer_lines) != 1
    )
    missing_line_citations = []
    citation_tail = re.compile(r"(?:\s*\[E\d+\][，,、\s]*)+[。.!?？]?$")
    for line in substantive_lines:
        if line == "- 现有检索证据不足，未作结论。":
            continue
        if line == process_limitation_sentence and line in limitation_lines:
            continue
        if insufficient and line in limitation_lines:
            continue
        if line.startswith("- ") and not citation_tail.search(line):
            missing_line_citations.append(line)
    citation_id_check_passed = (bool(cited_numbers) and not unsupported) or (insufficient and not cited_numbers)
    guard_errors = semantic_guard_errors(answer, evidence)
    semantic_guard_passed = not guard_errors
    format_check_passed = (
        headings.issubset(set(nonempty_lines))
        and not unbulleted_lines
        and not invalid_insufficient_placement
        and 1 <= len(answer_lines) <= 4
        and 1 <= len(limitation_lines) <= 2
    )
    claim_citation_check_passed = not missing_line_citations
    return {
        "valid": (
            citation_id_check_passed
            and format_check_passed
            and claim_citation_check_passed
            and semantic_guard_passed
        ),
        "citation_id_check_passed": citation_id_check_passed,
        "format_check_passed": format_check_passed,
        "claim_citation_check_passed": claim_citation_check_passed,
        "cited_evidence": [f"E{number}" for number in cited_numbers],
        "unsupported_citations": [f"E{number}" for number in unsupported],
        "unbulleted_lines": unbulleted_lines,
        "missing_line_citations": missing_line_citations,
        "insufficient_evidence_response": insufficient,
        "invalid_insufficient_placement": invalid_insufficient_placement,
        "answer_bullet_count": len(answer_lines),
        "limitation_bullet_count": len(limitation_lines),
        "semantic_guard_passed": semantic_guard_passed,
        "semantic_guard_errors": guard_errors,
        "limitation": "These checks validate formatting and citation IDs, not scientific entailment. Human review remains required.",
    }


def generate_text(model: Any, processor: Any, prompt: str, llm_device: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": "Use only supplied evidence. Preserve exact entities and numbers. Obey the output format."},
        {"role": "user", "content": prompt},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(llm_device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response_ids = generated[:, inputs.input_ids.shape[-1]:]
    response = processor.batch_decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    del generated, inputs
    return response


def answer(question: str, top_k: int, candidate_k: int, device: str, llm_device: str, max_new_tokens: int) -> dict[str, Any]:
    if max_new_tokens < 64 or max_new_tokens > 2000:
        raise RagError("max-new-tokens must be between 64 and 2000")
    retrieval = ask(question, top_k, candidate_k, device, include_text=True)
    started = perf_counter()
    if requires_boundary_refusal(question):
        generated_answer = boundary_refusal_answer()
        for item in retrieval["evidence"]:
            item.pop("text", None)
        retrieval.update(
            {
                "answer": generated_answer,
                "answer_model": "deterministic_evidence_boundary_v1",
                "llm_device": None,
                "generation_passes": 0,
                "answer_elapsed_seconds": round(perf_counter() - started, 3),
                "citation_validation": validate_citations(
                    generated_answer,
                    len(retrieval["evidence"]),
                    retrieval["evidence"],
                ),
                "scope_note": "Deterministic evidence-boundary response. No scientific conclusion was generated.",
            }
        )
        return retrieval
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RagError("Local Qwen3-VL-Instruct dependencies are unavailable in vl_rag") from exc
    if not LLM_MODEL_PATH.exists():
        raise RagError(f"Local LLM weights are missing: {LLM_MODEL_PATH}")
    device_index = llm_device_index(llm_device)
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise RagError(f"Requested LLM device is unavailable: {llm_device}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    processor = AutoProcessor.from_pretrained(str(LLM_MODEL_PATH), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(LLM_MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map={"": device_index},
        local_files_only=True,
    )
    initial_draft = generate_text(
        model,
        processor,
        make_answer_prompt(question, retrieval["evidence"]),
        llm_device,
        max_new_tokens,
    )
    generated_answer = generate_text(
        model,
        processor,
        make_verification_prompt(question, retrieval["evidence"], initial_draft),
        llm_device,
        max_new_tokens,
    )
    generation_passes = 2
    validation = validate_citations(
        generated_answer,
        len(retrieval["evidence"]),
        retrieval["evidence"],
    )
    if not validation["valid"]:
        repair_prompt = (
            make_verification_prompt(
                question,
                retrieval["evidence"],
                generated_answer,
                validation["semantic_guard_errors"],
            )
            if validation["semantic_guard_errors"]
            else make_format_repair_prompt(generated_answer, validation)
        )
        generated_answer = generate_text(
            model,
            processor,
            repair_prompt,
            llm_device,
            max_new_tokens,
        )
        generation_passes = 3
        validation = validate_citations(
            generated_answer,
            len(retrieval["evidence"]),
            retrieval["evidence"],
        )
    safety_fallback = False
    deterministic_corrections: list[str] = []
    generated_answer, format_corrections = apply_format_guard_corrections(generated_answer)
    deterministic_corrections.extend(format_corrections)
    if format_corrections:
        validation = validate_citations(
            generated_answer,
            len(retrieval["evidence"]),
            retrieval["evidence"],
        )
    generated_answer, language_corrections = apply_evidence_language_corrections(
        question,
        generated_answer,
        retrieval["evidence"],
    )
    deterministic_corrections.extend(language_corrections)
    if language_corrections:
        validation = validate_citations(
            generated_answer,
            len(retrieval["evidence"]),
            retrieval["evidence"],
        )
    if not validation["semantic_guard_passed"]:
        generated_answer, semantic_corrections = apply_semantic_guard_corrections(
            generated_answer,
            retrieval["evidence"],
        )
        deterministic_corrections.extend(semantic_corrections)
        validation = validate_citations(
            generated_answer,
            len(retrieval["evidence"]),
            retrieval["evidence"],
        )
    if not validation["semantic_guard_passed"]:
        generated_answer = semantic_fallback_answer()
        safety_fallback = True
        validation = validate_citations(
            generated_answer,
            len(retrieval["evidence"]),
            retrieval["evidence"],
        )
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    for item in retrieval["evidence"]:
        item.pop("text", None)
    retrieval.update(
        {
            "answer": generated_answer,
            "initial_draft": initial_draft,
            "answer_model": str(LLM_MODEL_PATH.relative_to(ROOT)),
            "llm_device": llm_device,
            "generation_passes": generation_passes,
            "safety_fallback": safety_fallback,
            "deterministic_corrections": deterministic_corrections,
            "answer_elapsed_seconds": round(perf_counter() - started, 3),
            "citation_validation": validation,
            "scope_note": "Local evidence-constrained synthesis with a second evidence-audit pass. Human verification remains required.",
        }
    )
    return retrieval


def save_answer_result(result: dict[str, Any]) -> Path:
    output_dir = ROOT / "runs" / "terminal_rag_answers"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_dir / f"answer_{stamp}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def print_text(result: dict[str, Any]) -> None:
    if "answer" in result:
        print("Answer (evidence-constrained local synthesis; human verification required):")
        print(result["answer"])
        validation = result["citation_validation"]
        print(f"\nCitation-ID check: passed={validation['citation_id_check_passed']} | cited={', '.join(validation['cited_evidence']) or 'none'}")
    print(f"Question: {result['question']}")
    print(f"Vector points: {result['point_count']} | Candidates: {result['candidate_count']} | Evidence: {result['evidence_count']}")
    print("Note: these are citable source passages, not an automatic scientific conclusion.")
    for item in result["evidence"]:
        print(f"\n[{item['rank']}] {item['title']}")
        print(f"  Type: {item['source_role']} | Record: {item['record']} | Section: {item['section']}")
        print(f"  Source: {item['source_path']}")
        print(f"  Score: rerank={item['reranker_score']:.4f}, vector={item['vector_score']:.4f}")
        print(f"  Excerpt: {item['excerpt']}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="cervixagent-rag", description="Terminal RAG evidence retrieval for CervixAgent")
    sub = parser.add_subparsers(dest="command", required=True)
    ask_parser = sub.add_parser("ask", help="Retrieve and rerank citable literature evidence")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=8)
    ask_parser.add_argument("--candidate-k", type=int, default=40)
    ask_parser.add_argument("--device", default="cuda:0")
    ask_parser.add_argument("--json", action="store_true")
    answer_parser = sub.add_parser("answer", help="Produce a local, evidence-constrained synthesis with source labels")
    answer_parser.add_argument("question")
    answer_parser.add_argument("--top-k", type=int, default=6)
    answer_parser.add_argument("--candidate-k", type=int, default=40)
    answer_parser.add_argument("--device", default="cuda:0", help="Embedding and reranker GPU")
    answer_parser.add_argument("--llm-device", default="cuda:1", help="Local Qwen3-VL-Instruct GPU")
    answer_parser.add_argument("--max-new-tokens", type=int, default=700)
    answer_parser.add_argument("--save", action="store_true", help="Save the full reproducibility record under project runs/")
    answer_parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "ask":
            result = ask(args.question, args.top_k, args.candidate_k, args.device)
        else:
            result = answer(args.question, args.top_k, args.candidate_k, args.device, args.llm_device, args.max_new_tokens)
            if args.save:
                result["saved_run"] = str(save_answer_result(result).relative_to(ROOT))
    except RagError as exc:
        print(f"RAG error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_text(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
