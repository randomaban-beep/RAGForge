# RAGForge: resume evidence

## Suggested resume bullets

- Built a hybrid RAG workbench using BM25, MiniLM embeddings, and reciprocal rank fusion; achieved **76.31% Recall@5** and **0.669 nDCG@5** on **300 BEIR/SciFact test queries across 5,183 documents**.
- Implemented an evaluation pipeline comparing **5 retrieval configurations**, with per-query rankings, 95% bootstrap intervals, latency percentiles, **31 automated tests**, and CI gates for retrieval regressions.

## Measured comparison you can discuss

Hybrid retrieval versus the dense-only baseline:
- nDCG@5: 0.6293 -> 0.6688 (+6.3% relative change).
- Recall@5: 73.79% -> 76.31%.
- Median retrieval latency: 13.97 ms -> 44.83 ms.
- p95 retrieval latency: 19.29 ms -> 80.50 ms.

This compares configurations in the new evaluation harness, not a measured before/after of the
original application. The retrieval-only CPU run uses one measured pass after warmup, k=5,
candidate_k=20, and pinned model revisions. It is not deployed or concurrent latency.
Keep the latency tradeoff in the discussion. The cross-encoder ablation was slower and did not improve nDCG over hybrid retrieval in this run; do not claim that reranking improved this system.

Paired 95% bootstrap interval for the hybrid-minus-dense nDCG difference: [0.0140, 0.0650] (2,000 query resamples; seed 42).

## Evidence

- [Benchmark table](../reports/scifact/retrieval.md)
- [Per-query results, confidence intervals, configuration, and provenance](../reports/scifact/retrieval.json)
- [Evaluation protocol](EVALUATION.md)
- [Test and integration validation](../reports/validation.json)

The dataset is the public BEIR/SciFact retrieval test split, not the hidden original scientific
claim-verification leaderboard test split. No model was trained or fine-tuned for this project.
These measurements do not establish general performance on enterprise data.

## Claims to avoid

Do not write "90% answer accuracy", "zero hallucinations", "production-scale", "served thousands of
users", or a fabricated cost reduction. The synthetic demo has 90% abstention-decision agreement,
but it still fails two of four unanswerable/adversarial cases. Citation ID validity is not semantic
citation accuracy. Live OpenAI generation and judging were not run because no API key was configured.

## Interview walkthrough

1. Load the sample handbook and show a source-backed evidence excerpt.
2. Explain the lexical baseline, semantic retriever, fusion, and reranker.
3. Compare all five rows of the public benchmark, including the latency cost.
4. Show one failed query and explain what a recall failure versus an answer failure means.
5. Explain the CI gate and why you would collect domain-specific held-out labels before deployment.
