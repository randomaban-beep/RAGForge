"""python -m src.evaluation.run --dataset scifact --download --neural"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import numpy as np
from src.evaluation.datasets import download_scifact, load_beir, load_fixture
from src.evaluation.metrics import bootstrap_interval, retrieval_metrics
from src.retrieval.hybrid import SearchIndex, EMBEDDING_MODEL
from src.retrieval.reranker import Reranker, RERANKER_MODEL


def provenance():
    source = hashlib.sha256()
    for path in sorted(Path("src").rglob("*.py")):
        source.update(path.as_posix().encode())
        source.update(path.read_bytes())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    packages = {}
    for package in ("numpy", "scikit-learn", "rank-bm25", "sentence-transformers", "torch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "source_sha256": source.hexdigest(),
        "python": platform.python_version(),
        "os": platform.system(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "packages": packages,
    }


def evaluate(index, cases, modes, k=5, candidate_k=20, repeats=1, reranker=None):
    if not cases or repeats < 1:
        raise ValueError("Need nonempty cases and positive repeats.")
    results = {}
    for mode in modes:
        print(f"Evaluating {mode}: {len(cases)} queries", flush=True)
        rows, latencies = [], []

        def retrieve(question):
            if mode == "hybrid_rerank":
                return reranker.rerank(
                    question, index.search(question, "hybrid", candidate_k, candidate_k), k
                )
            return index.search(question, mode, k, candidate_k)

        retrieve(cases[0]["question"])  # One untimed warmup per mode.
        for case_index, case in enumerate(cases, 1):
            timings = []
            for _ in range(repeats):
                start = perf_counter()
                hits = retrieve(case["question"])
                timings.append((perf_counter() - start) * 1000)
            ids = [hit.document.id for hit in hits]
            rows.append(
                {
                    "query_id": case["id"],
                    "retrieved_ids": ids,
                    "relevance": case["relevance"],
                    "metrics": retrieval_metrics(ids, case["relevance"], k),
                    "latency_ms": timings,
                }
            )
            latencies.extend(timings)
            if case_index % 50 == 0:
                print(f"  {mode}: {case_index}/{len(cases)}", flush=True)
        summary = {
            name: float(np.mean([row["metrics"][name] for row in rows]))
            for name in rows[0]["metrics"]
        }
        summary["ndcg_ci95"] = bootstrap_interval([r["metrics"][f"ndcg@{k}"] for r in rows])
        summary["latency_p50_ms"], summary["latency_p95_ms"] = [
            float(x) for x in np.quantile(latencies, [0.5, 0.95])
        ]
        results[mode] = {"summary": summary, "queries": rows}
    baseline = modes[0]
    for mode in modes[1:]:
        delta = [
            a["metrics"][f"ndcg@{k}"] - b["metrics"][f"ndcg@{k}"]
            for a, b in zip(results[mode]["queries"], results[baseline]["queries"])
        ]
        results[mode]["vs_baseline"] = {
            "baseline": baseline,
            "ndcg_delta": float(np.mean(delta)),
            "paired_bootstrap_ci95": bootstrap_interval(delta),
        }
    return results


def write_report(report, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    k = report["config"]["k"]
    lines = [
        "# Retrieval benchmark",
        "",
        f"Dataset: {report['dataset']}. "
        f"{report['query_count']} queries; {report['document_count']} documents.",
        "",
        "Retrieval-only, single-process CPU measurements. No LLM calls. "
        "Index construction and model downloads are excluded from query latency.",
        "",
        f"| Mode | Recall@{k} | MRR@{k} | nDCG@{k} | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, result in report["results"].items():
        s = result["summary"]
        lines.append(
            f"| {mode} | {s[f'recall@{k}']:.4f} | {s[f'mrr@{k}']:.4f} | "
            f"{s[f'ndcg@{k}']:.4f} | {s['latency_p50_ms']:.2f} | "
            f"{s['latency_p95_ms']:.2f} |"
        )
    lines += [
        "",
        "See retrieval.json for per-query rankings, paired bootstrap intervals, "
        "dataset hashes, package versions, source hash, and run conditions.",
        "These results do not measure factual correctness, hallucination rate, "
        "concurrent throughput, or deployed service latency.",
    ]
    (out / "retrieval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["demo", "scifact"], default="demo")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--neural", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", default="reports/demo")
    parser.add_argument("--min-recall", type=float, default=0)
    args = parser.parse_args()
    if not 1 <= args.k <= args.candidate_k <= 100 or args.repeats < 1:
        parser.error("Require 1 <= k <= candidate-k <= 100 and repeats >= 1.")
    if not 0 <= args.min_recall <= 1:
        parser.error("min-recall must be between 0 and 1.")
    if args.dataset == "scifact":
        root = (
            download_scifact("data/benchmarks")
            if args.download
            else Path("data/benchmarks/scifact")
        )
        docs, cases, hashes = load_beir(root)
    else:
        docs, all_cases, hashes = load_fixture()
        cases = [c for c in all_cases if c["relevance"]]
    modes = ["tfidf", "bm25"]
    if args.neural:
        modes += ["dense", "hybrid", "hybrid_rerank"]
    started = perf_counter()
    index = SearchIndex(docs, neural=args.neural)
    reranker = Reranker() if args.neural else None
    build_seconds = perf_counter() - started
    results = evaluate(index, cases, modes, args.k, args.candidate_k, args.repeats, reranker)
    report = {
        "schema_version": 1,
        "dataset": args.dataset,
        "dataset_hashes": hashes,
        "document_count": len(docs),
        "query_count": len(cases),
        "config": {
            "k": args.k,
            "candidate_k": args.candidate_k,
            "rrf_constant": 60,
            "repeats": args.repeats,
            "warmups_per_mode": 1,
            "chunking": "none; title + complete document",
            "embedding_model": EMBEDDING_MODEL if args.neural else None,
            "embedding_revision": index.model_revision,
            "reranker_revision": reranker.model_revision if reranker else None,
            "reranker_model": RERANKER_MODEL if args.neural else None,
            "device": "cpu",
            "bm25_idf": "log(1 + (N - df + 0.5) / (df + 0.5))",
            "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
            "mkl_num_threads": os.getenv("MKL_NUM_THREADS"),
            "build_seconds_including_model_load": build_seconds,
        },
        "provenance": provenance(),
        "results": results,
    }
    write_report(report, args.out)
    if any(r["summary"][f"recall@{args.k}"] < args.min_recall for r in results.values()):
        raise SystemExit("Retrieval quality gate failed.")
    print(f"Report written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
