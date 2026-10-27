# Retrieval benchmark

Dataset: demo. 16 queries; 12 documents.

Retrieval-only, single-process CPU measurements. No LLM calls. Index construction and model downloads are excluded from query latency.

| Mode | Recall@5 | MRR@5 | nDCG@5 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| tfidf | 1.0000 | 0.9219 | 0.9414 | 1.14 | 2.38 |
| bm25 | 1.0000 | 0.9271 | 0.9457 | 0.17 | 0.32 |

See retrieval.json for per-query rankings, paired bootstrap intervals, dataset hashes, package versions, source hash, and run conditions.
These results do not measure factual correctness, hallucination rate, concurrent throughput, or deployed service latency.
