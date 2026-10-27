# Industry signals and project decisions

Research checked 27 October 2026. These are examples of public engineering role requirements,
not a representative recruiter survey or a promise of hiring outcomes. Roles cited include
senior positions; use their technical themes, not their seniority, as portfolio guidance.

| Primary source | Signal | Concrete evidence in RAGForge |
|---|---|---|
| [Anthropic Applied AI Engineer, Enterprise](https://job-boards.greenhouse.io/anthropic/jobs/5057647008) | Building evaluation suites, Python/TypeScript production applications, and explaining architecture choices | Automated retrieval and answer evaluation; inspectable reports; documented tradeoffs |
| [OpenAI Retrieval & Search engineer](https://openai.com/careers/research-engineer-retrieval-and-search-applied-engineering-san-francisco/) | Search indices, retrieval algorithms, production ML, end-to-end ownership | TF-IDF/BM25/dense/hybrid/reranker ablation using the same retrieval code as the demo |
| [MLflow RAG evaluation](https://www.mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/) | Separately assess retrieval relevance, context sufficiency, and groundedness | Retrieval judgments stay separate from answer overlap and optional groundedness/citation-support judgments |
| [Microsoft RAG evaluators](https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/rag-evaluators) | Ground-truth retrieval metrics and response evaluation answer different questions | Labeled retrieval benchmark; explicit metric definitions and limitations |
| [BEIR project](https://github.com/beir-cellar/beir) | Standard public IR datasets support repeatable comparisons | Full SciFact test queries, corpus hashes, per-query rankings, bootstrap intervals |

## What to demonstrate to a reviewer

1. Run the sample locally without credentials, inspect an answer and its cited source.
2. Show the public benchmark and explain why recall, rank quality, and latency matter separately.
3. Show an actual failure and the regression gate that prevents an accidental quality drop.
4. Explain that retrieval quality is necessary but not sufficient for trustworthy generation.
5. Discuss the extra CPU cost of reranking, session isolation, bounded inputs, and missing production controls.

The engineering inference from these sources is that measured, explainable behavior is more
credible than a list of framework names. This is why the project removes the extra LLM routing
call and generic chat path, consolidates ingestion, and adds evaluation before adding more agents.

## Deliberate limits

This is a local portfolio workbench, not a production service. It has no authentication,
tenant authorization, distributed vector service, load testing, or deployment SLO.
The in-memory exact vector index makes small-corpus evaluation transparent; it does not establish
internet-scale search capability. A vector database should be introduced with a workload and
persistence requirement, then benchmarked against this reference implementation.
