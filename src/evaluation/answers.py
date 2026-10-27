"""Separate synthetic answer suite; optional paid generation and judging."""

import argparse
import json
from pathlib import Path
import numpy as np
from src.evaluation.datasets import load_fixture
from src.evaluation.metrics import answer_metrics
from src.evaluation.run import provenance
from src.pipeline import Config, RAGPipeline


def run_answers(generation="extractive", judge_enabled=False):
    docs, cases, hashes = load_fixture()
    pipeline = RAGPipeline(Config(generation=generation))
    pipeline.index_records(docs, chunk=False)
    judge = None
    if judge_enabled:
        from src.evaluation.judge import AnswerJudge

        judge = AnswerJudge()
    rows = []
    for case in cases:
        result = pipeline.query(case["question"])
        row = {
            "case_id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "reference": case["reference"],
            "expected_abstention": case["expected_abstention"],
            "result": result,
            "metrics": answer_metrics(result, case["reference"], case["expected_abstention"]),
        }
        if judge:
            row["judge"] = judge.evaluate(
                case["question"], result, case["reference"], case["expected_abstention"]
            )
        rows.append(row)
    summary = {}
    for metric in rows[0]["metrics"]:
        values = [r["metrics"][metric] for r in rows if r["metrics"][metric] is not None]
        summary[metric] = float(np.mean(values)) if values else None
        summary[metric + "_n"] = len(values)
    for expected, name in [
        (True, "unanswerable_abstention_rate"),
        (False, "answerable_response_rate"),
    ]:
        subset = [r for r in rows if r["expected_abstention"] == expected]
        summary[name] = float(np.mean([r["result"]["abstained"] == expected for r in subset]))
    latency = [r["result"]["trace"]["total_ms"] for r in rows]
    summary["total_latency_p50_ms"], summary["total_latency_p95_ms"] = [
        float(v) for v in np.quantile(latency, [0.5, 0.95])
    ]
    return {
        "schema_version": 1,
        "dataset": "hand-authored synthetic demo; not held-out",
        "dataset_hashes": hashes,
        "provenance": provenance(),
        "generation": generation,
        "judge_enabled": judge_enabled,
        "summary": summary,
        "cases": rows,
        "limitations": "Token F1 is lexical overlap, not semantic correctness. Citation ID "
        "validity is not citation support. Judge scores require human calibration. "
        "No cost estimate is fabricated; token usage is reported for API calls.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", choices=["extractive", "openai"], default="extractive")
    parser.add_argument("--judge", action="store_true", help="Opt in to paid model judging.")
    parser.add_argument("--out", default="reports/demo/answers.json")
    args = parser.parse_args()
    report = run_answers(args.generation, args.judge)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
