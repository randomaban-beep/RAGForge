# Retrieval benchmark

Dataset: scifact. 300 queries; 5183 documents.

Retrieval-only, single-process CPU measurements. No LLM calls. Index construction and model downloads are excluded from query latency.

| Mode | Recall@5 | MRR@5 | nDCG@5 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| tfidf | 0.6499 | 0.5363 | 0.5569 | 4.34 | 5.69 |
| bm25 | 0.7209 | 0.6257 | 0.6427 | 29.18 | 58.73 |
| dense | 0.7379 | 0.5997 | 0.6293 | 13.97 | 19.29 |
| hybrid | 0.7631 | 0.6434 | 0.6688 | 44.83 | 80.50 |
| hybrid_rerank | 0.7442 | 0.6529 | 0.6669 | 2197.15 | 2354.57 |

See retrieval.json for per-query rankings, paired bootstrap intervals, dataset hashes, package versions, source hash, and run conditions.
These results do not measure factual correctness, hallucination rate, concurrent throughput, or deployed service latency.
