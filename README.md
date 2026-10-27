# RAGForge

**A document Q&A and retrieval evaluation workbench with reproducible performance evidence.**

Upload documents, inspect retrieved evidence and citations, and compare retrieval strategies using a
public labeled benchmark. The default demo runs locally without an API key. Neural retrieval and
OpenAI generation are optional.

## Run the working demo

Python 3.11 or 3.12, from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

On Windows, you can also double-click **run-demo.cmd** after extracting/cloning the project; it creates a local environment and installs the runtime dependencies.

Choose **Load sample handbook**, then ask **How often are API keys rotated?**
Inspect the cited source and query trace. The Evaluation reports tab displays committed benchmark
results, latency, failure cases, and downloadable JSON. Upload PDF, DOCX, TXT, or Markdown to replace
the sample. Each browser session owns its own in-memory knowledge base.

The default **extractive** mode returns a matching evidence sentence. It is a runnable search/excerpt
demo, not an LLM or a claim of semantic answer accuracy. Ask self-contained questions; conversation
history is displayed but not used to rewrite follow-ups.

## Retrieval and generation options

| Mode | Purpose | Additional requirements |
|---|---|---|
| TF-IDF | Fast lexical baseline | None |
| BM25 | Default keyword retrieval with positive IDF for small corpora | None |
| Dense | MiniLM semantic embeddings and exact cosine search | requirements-neural.txt; model download |
| Hybrid | BM25 + dense reciprocal rank fusion | Neural dependencies |
| Hybrid + reranking | CrossEncoder rescores the top 20 fused candidates | Neural dependencies |
| OpenAI answers | Structured answers, source IDs, abstention, token usage | OPENAI_API_KEY; paid provider calls |

```bash
pip install -r requirements-neural.txt
```

For OpenAI answers, copy .env.example to .env and set your own key locally. The default model is
gpt-5-mini, configurable with OPENAI_MODEL. Select openai and rebuild the sample/upload index.
Questions and retrieved evidence are sent to the provider. Citation IDs are validated against the
actual retrieved set; this does not guarantee that every claim is supported.

## Architecture

```text
PDF / DOCX / TXT / MD
  -> in-memory parser -> stable source IDs -> overlapping word chunks
  -> per-session index
  -> TF-IDF | BM25 | dense | hybrid RRF -> optional CrossEncoder
  -> evidence excerpts | structured LLM answer
  -> citations + stage timings + token usage

Public corpus + relevance judgments
  -> same retrieval implementation
  -> per-query Recall / MRR / nDCG + p50 / p95 + confidence intervals
  -> versioned report + CI regression gate
```

The local exact vector index is intentionally easy to inspect and benchmark on a small corpus.
It replaces an in-memory Qdrant wrapper; persistence, distributed search, and tenant authorization
remain deployment work.

## Evaluate and test

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
python -m src.evaluation.run --min-recall 0.8 --out reports/demo
python -m src.evaluation.answers
python -m src.evaluation.run --dataset scifact --download --repeats 3 --out reports/scifact-lexical
python -m src.evaluation.run --dataset scifact --download --neural --out reports/scifact
```

The public run uses all 300 BEIR/SciFact test queries and all 5,183 documents. It evaluates document
retrieval, not scientific claim truth or generated-answer correctness. The 20-case synthetic answer
fixture is separate and explicitly labeled as development data.

Optional paid answer evaluation:

```bash
python -m src.evaluation.answers --generation openai --judge --out reports/llm/answers.json
```

The judge separately scores correctness, groundedness, and citation support and records rationales.
Human calibration is still needed. No paid generation/judge results are claimed in this repository's
offline benchmark.

CI runs offline tests, an absolute recall smoke gate, and a comparison against the committed fixture
report. Dataset hashes and query identities must match; a recall or nDCG drop over 0.02 fails.
Reports include source fingerprints, dependencies, rankings, and run conditions.

## Results and portfolio evidence

Measured on all **300 BEIR/SciFact test queries** and **5,183 documents**, on CPU.
These are retrieval-only results, not generated-answer accuracy.

| Mode | Recall@5 | MRR@5 | nDCG@5 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| tfidf | 0.6499 | 0.5363 | 0.5569 | 4.34 | 5.69 |
| bm25 | 0.7209 | 0.6257 | 0.6427 | 29.18 | 58.73 |
| dense | 0.7379 | 0.5997 | 0.6293 | 13.97 | 19.29 |
| hybrid | 0.7631 | 0.6434 | 0.6688 | 44.83 | 80.50 |
| hybrid_rerank | 0.7442 | 0.6529 | 0.6669 | 2197.15 | 2354.57 |

Hybrid had the highest Recall@5 and nDCG@5 in this fixed comparison. Reranking increased median retrieval latency to 2.2 seconds without improving nDCG over hybrid, so it remains an ablation rather than a recommended default. The lightweight no-download demo uses BM25; use hybrid to reproduce the strongest measured retrieval configuration.
The run used pinned model revisions, k=5, candidate_k=20, one measured pass after warmup, and no LLM calls.
Model loading/index construction is excluded from query latency. See JSON for confidence intervals and provenance.


See [public benchmark](reports/scifact/retrieval.md), [full measured results](reports/scifact/retrieval.json),
[answer failures](reports/demo/answers.json), and [resume bullets](docs/RESUME.md).
Read the [evaluation protocol](docs/EVALUATION.md) before interpreting or quoting numbers.

## Why these additions matter

Public AI engineering roles emphasize evaluation frameworks, retrieval systems, and end-to-end
implementation. The project demonstrates these through measurable comparisons, tests, failure
inspection, and explicit tradeoffs. The [industry research](docs/INDUSTRY_RESEARCH.md) cites primary
job postings and evaluation documentation; it is not a recruiter survey.

See [repository cleanup](docs/CHANGELOG.md) for what was removed and why.

## Project layout

```text
app/                    Streamlit demo and report viewer
src/ingestion/          Bounded in-memory loaders and stable chunk IDs
src/retrieval/          TF-IDF, BM25, dense, RRF, and reranker
src/generation/         Evidence excerpts and optional structured LLM generation
src/evaluation/         Metrics, public benchmark, answer judge, regression comparison
src/pipeline.py         Shared application orchestration and query traces
tests/                 Unit, integration, quality-gate, and Streamlit interaction tests
data/demo.json         Fictional sample handbook and 20 development cases
reports/               Actual measured runs and environment snapshot
docs/                  Protocol, industry research, cleanup, and resume evidence
```

## Container

```bash
docker build -t ragforge .
docker run --rm -p 8501:8501 ragforge
```

The container includes the offline demo; neural dependencies are not installed in this image.
The Docker recipe is supplied but is not claimed as tested unless noted in validation results.

## Limits

This is a portfolio workbench, not a production-ready hosted service. It has no authentication,
persistent tenant storage, concurrency benchmark, or uptime SLO. Inputs are bounded (10 MiB/file,
30 MiB total; text extraction capped), but untrusted document processing needs stronger resource
isolation in production. Scanned PDFs require external OCR. Prompt instructions are not a complete
defense against prompt injection. Excerpt mode can return related evidence that does not answer the
question; inspect the documented failures.

**Author:** [Md Aban Zeya](https://github.com/randomaban-beep)
