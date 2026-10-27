from copy import deepcopy
import pytest
from src.evaluation.compare import compare


def reports():
    report = {
        "dataset": "test",
        "dataset_hashes": {"test": "abc"},
        "document_count": 2,
        "query_count": 1,
        "config": {"k": 5},
        "results": {
            "bm25": {
                "summary": {"recall@5": 0.8, "ndcg@5": 0.7},
                "queries": [{"query_id": "1", "relevance": {"a": 1}}],
            }
        },
    }
    return report, deepcopy(report)


def test_regression_gate_detects_drop():
    old, new = reports()
    new["results"]["bm25"]["summary"]["ndcg@5"] = 0.6
    assert len(compare(old, new)) == 1


def test_comparison_refuses_changed_dataset():
    old, new = reports()
    new["dataset_hashes"]["test"] = "changed"
    with pytest.raises(ValueError):
        compare(old, new)


def test_gate_accepts_equal_and_rejects_nan():
    old, new = reports()
    assert compare(old, new) == []
    new["results"]["bm25"]["summary"]["recall@5"] = float("nan")
    with pytest.raises(ValueError):
        compare(old, new)
