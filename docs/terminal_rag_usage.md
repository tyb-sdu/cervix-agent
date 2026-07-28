# CervixAgent terminal RAG

Run all commands on the server inside the project directory:

```bash
cd /data2/lxj/projects/CervixAgent
./scripts/cervixagent-rag ask "What structural evidence explains how HPV16 E6, E6AP and p53 assemble into a degradation complex?"
```

Useful options:

```bash
# Return five final evidence passages after reranking 30 candidates
./scripts/cervixagent-rag ask "What is the prognostic relevance of IDO1 expression in cervical cancer?" --top-k 5 --candidate-k 30

# Machine-readable result for later reports or agent orchestration
./scripts/cervixagent-rag ask "What supplementary methods describe dynamic light scattering for E6 and E6AP complexes?" --json
```

The command returns ranked evidence with the paper or SI type, record ID, section, source path, scores and a text excerpt. It does not generate a scientific conclusion. Open the cited source passage before using it in a protocol, report or decision.

The first query normally takes about 20-30 seconds because the embedding and reranking models are loaded locally. No external API call is made.

## Evidence-constrained local answer

`answer` performs retrieval and reranking on GPU 0, then the server-local
Qwen3-VL-8B-Instruct produces a Chinese synthesis on GPU 1 using only those passages.
Supplementary-material questions are automatically restricted to sources whose
`source_role` is `supplement`. Near-duplicate passages are removed before generation.

It is not open-ended chat. The answer must contain `回答` and `证据局限`, use at most
four atomic answer bullets and two limitation bullets, and attach an evidence ID such as
`[E1]` to every factual claim. A second model pass audits the draft against the retrieved
evidence; a third pass is used only when citation, format or semantic checks fail.
Unsupported clinical cure claims, future docking scores and human-safety conclusions are
refused deterministically instead of being guessed.

The final language guards also:

- label review/discussion evidence and observational/database evidence explicitly;
- prevent a Western-blot ubiquitination result from being described as direct binding evidence;
- distinguish an electrophile-screening hit from proof of a selective drug lead;
- normalize scientific terms including `点击化学`, `亲电体`, `亲电弹头` and `亲核性氨基酸`;
- replace unsupported inferred-absence limitations with a neutral human-verification notice.

```bash
cd /data2/lxj/projects/CervixAgent
./scripts/cervixagent-rag answer "How do HPV16 E6, E6AP and p53 assemble into a degradation complex?" --save
```

`--save` writes the question, model path, draft/final answer, source passages, validation
fields, generation-pass count and versioned retrieval record under
`runs/terminal_rag_answers/`; without it, the user question is not saved.

The automatic checks cover citation-ID validity, required format, claim-level citation
presence and selected known entity-consistency risks. They do not replace a calculation
chemistry or biology reviewer: a researcher must still verify scientific entailment and the
original passage before using an answer in a protocol, report or decision.

## Acceptance record

The final script SHA256 is:

```text
08ee454a336161d0524611368987e920da439426be459ecbc17bbd220fee981b
```

The final acceptance package is stored under:

```text
reports/rag_acceptance/rag_acceptance_final_v4_20260728/
```

All 15 fixed questions completed and passed the automatic validator. The accompanying
human review checks whether the displayed passages support each answer; it remains a
quality-control record rather than a substitute for domain-expert review of the original
papers, supplementary files, structures or calculations.
