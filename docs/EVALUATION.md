# Evaluation protocol

## Three separate layers

**Retrieval:** run all labeled BEIR/SciFact test queries against the full corpus. A record is its title
plus full abstract; no app-style chunking is applied because relevance labels are document IDs.
No training, query-dependent corpus filtering, or test-set parameter search is performed. BEIR names this split test; it is not the hidden original SciFact leaderboard test split.
All modes share corpus, questions, k=5, and candidate pool setting 20. The lexical modes share tokenization; neural models use their own tokenizers. BM25 uses positive IDF log(1 + (N - df + 0.5)/(df + 0.5)), k1=1.5, and b=0.75. This avoids the negative-IDF edge case on single-document uploads.
The neural model revisions are pinned in code. This tested library returned null for resolved revision metadata; the exact constructor pins are MiniLM 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 and CrossEncoder 233902d25c440f23af6f7d6e94d2946bac0bee0a. The report source fingerprint identifies the code containing those pins. The neural encoders may truncate input according to their model limits. This is not a benchmark
of the document chunker or conversational follow-up handling.

**Answer checks:** a 20-case fictional operations handbook fixture exercises answerable questions,
missing information, out-of-domain queries, and an instruction attack. It is a development fixture,
not held-out data. Offline mode extracts a sentence; it does not reason about whether that sentence
fully answers the question. Failures are retained in reports/demo/answers.json.

**Optional model judgments:** use a structured judge to score correctness, groundedness, and citation
support from 0 to 2 with a written rationale. The model and rubric version are recorded per case.
Judging can be biased, especially when generator and judge share a model. Review a stratified sample
manually and measure judge/human agreement before treating these scores as quality evidence.
A model refusal or missing output fails the run instead of becoming a passing score.

## Metric definitions

- Recall@k: retrieved relevant IDs divided by all judged relevant IDs.
- Precision@k: retrieved relevant IDs divided by k (even when fewer hits are returned).
- MRR@k: reciprocal rank of the first relevant document, or zero if absent.
- nDCG@k: discounted gain with gain 2^relevance-1, normalized by ideal ranking.
- 95% bootstrap intervals: 2,000 query-level resamples with seed 42; comparisons use paired deltas.
- p50/p95 latency: quantiles of observed query times, after one untimed warmup per mode.
  No concurrent traffic; index/model construction is reported separately.
- Reference token F1: multiset token overlap against the reference; not semantic correctness.
- Citation ID validity: fraction of emitted IDs present in retrieved context. No citation means null,
  not a perfect score. This does not test whether the text supports the claim.
- Abstention correctness: agreement with the fixture's expected answer/abstain label.
  Unanswerable abstention and answerable response rates are also reported separately.

SciFact labels identify evidence for scientific claims, including refuting evidence. A high retrieval
score does not mean a claim is true. Unjudged documents are treated as nonrelevant under this protocol.
The benchmark is small and domain-specific; do not generalize its results to arbitrary enterprise data.

## Reproduce

From the repository root, after installing requirements-dev.txt:

```bash
python -m pytest -q
python -m src.evaluation.run --out reports/demo --min-recall 0.8
python -m src.evaluation.answers
python -m src.evaluation.run --dataset scifact --download --repeats 3 --out reports/scifact-lexical
pip install -r requirements-neural.txt
python -m src.evaluation.run --dataset scifact --download --neural --out reports/scifact
```

For the recorded neural run, OMP_NUM_THREADS and MKL_NUM_THREADS were both 4.
Set these in the shell before running to approximate the CPU setup. Windows PowerShell:
`$env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'`.
CPU, operating system, dependencies, data hashes, configuration, rankings, and code fingerprint are
recorded in each report. Latency varies with hardware, background tasks, and thread settings.

The downloader fetches the [BEIR SciFact archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip).
See the [SciFact source project and dataset terms](https://github.com/allenai/scifact) before redistribution.
Downloaded data and model caches are excluded from Git. No dataset text is copied into retrieval reports.

## Paid generation / judgment

Set OPENAI_API_KEY in your local environment or .env, then explicitly opt in:

```bash
python -m src.evaluation.answers --generation openai --judge --out reports/llm/answers.json
```

The optional [Responses structured-output integration](https://developers.openai.com/api/docs/guides/structured-outputs)
uses OPENAI_MODEL and JUDGE_MODEL (default gpt-5-mini), timeouts, bounded retries, and store=False.
Provider calls send the question and retrieved evidence; do not submit private documents without
the right permissions. Input/output tokens are recorded. Dollar cost is deliberately not inferred
from a stale price table, and no paid-call results are claimed without an actual run.

For human review, inspect cases across every category, independently label correctness and citation
support, then compare with judge scores. Record disagreements and update a versioned development set;
keep a separate held-out set for final reporting.

## Regression gates

CI runs deterministic offline tests and a minimum Recall@5 gate of 0.8 on each fixture retrieval mode.
It also compares against the committed fixture report, rejecting an absolute recall or nDCG drop
larger than 0.02. It refuses incompatible dataset hashes, query IDs, or k values.
The tiny fixture is a smoke gate, not evidence of generalization.
Public neural benchmarks are explicit runs so CI does not require large model downloads or API secrets.

```bash
python -m src.evaluation.compare reports/demo/retrieval.json reports/ci/retrieval.json --max-drop 0.02
```

## Known failure modes

Lexical matching misses paraphrases; dense similarity can retrieve plausible but wrong context;
RRF can dilute a strong lexical match; reranking adds CPU cost and cannot recover a document outside
its candidate set. The excerpt generator can copy irrelevant or adversarial text and cannot guarantee
abstention on a related but unanswered question. The LLM prompt is a mitigation, not a security boundary.
For a deployed system, add independent injection testing, access controls, rate limits, resource isolation,
persistent storage, and representative load tests.

Fixture fingerprints hash canonical JSON so Windows/Linux line endings and JSON formatting do not create false regressions. Public BEIR files retain raw-byte hashes.

The public SciFact run's source fingerprint matches GitHub commit c515607ba5ef4656697102129cddc2caabc51e7c. Its report retains the equivalent local benchmark commit ID. The subsequent portability fix changes only the demo fixture fingerprint and its regression test, not retrieval or public-benchmark scores.
