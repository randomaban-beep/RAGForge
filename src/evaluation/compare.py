"""Fail CI if a candidate report regresses on the same judged queries."""

import argparse
import json
import math
from pathlib import Path


def compare(baseline, candidate, max_drop=0.02):
    if not math.isfinite(max_drop) or max_drop < 0:
        raise ValueError("max_drop must be finite and nonnegative.")
    for key in ("dataset", "dataset_hashes", "document_count", "query_count"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Incompatible reports: {key} changed.")
    if baseline["config"]["k"] != candidate["config"]["k"]:
        raise ValueError("Cannot compare different k values.")
    k = baseline["config"]["k"]
    failures = []
    for mode, old in baseline["results"].items():
        if mode not in candidate["results"]:
            raise ValueError(f"Candidate is missing mode {mode}.")
        new = candidate["results"][mode]
        old_rows = {r["query_id"]: r["relevance"] for r in old["queries"]}
        new_rows = {r["query_id"]: r["relevance"] for r in new["queries"]}
        if old_rows != new_rows:
            raise ValueError("Query IDs or relevance judgments changed.")
        for metric in (f"recall@{k}", f"ndcg@{k}"):
            before, after = old["summary"][metric], new["summary"][metric]
            if not math.isfinite(before) or not math.isfinite(after):
                raise ValueError("Metrics must be finite.")
            if before - after > max_drop + 1e-12:
                failures.append(f"{mode} {metric}: {before:.4f} -> {after:.4f}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--max-drop", type=float, default=0.02)
    args = parser.parse_args()
    baseline, candidate = [
        json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.baseline, args.candidate)
    ]
    failures = compare(baseline, candidate, args.max_drop)
    if failures:
        raise SystemExit("Quality regression:\n" + "\n".join(failures))
    print("No retrieval regression beyond the allowed absolute drop.")


if __name__ == "__main__":
    main()
