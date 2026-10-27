import math
import pytest
from src.evaluation.metrics import answer_metrics, bootstrap_interval, retrieval_metrics, token_f1


def test_hand_computed_ranking():
    metrics = retrieval_metrics(["noise", "a", "b"], {"a": 1, "b": 1}, 3)
    assert metrics["recall@3"] == 1
    assert metrics["precision@3"] == 2 / 3
    assert metrics["mrr@3"] == 0.5
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert metrics["ndcg@3"] == pytest.approx(expected)


def test_duplicate_results_do_not_inflate_recall():
    result = retrieval_metrics(["a", "a", "a"], {"a": 1, "b": 1}, 3)
    assert result["recall@3"] == 0.5
    assert result["ndcg@3"] <= 1


def test_graded_ndcg_and_missing_results():
    assert retrieval_metrics(["a", "b"], {"a": 3, "b": 1}, 2)["ndcg@2"] == 1
    assert retrieval_metrics([], {"a": 1}, 5)["mrr@5"] == 0
    with pytest.raises(ValueError):
        retrieval_metrics(["a"], {}, 5)


def test_f1_counts_token_multiplicity():
    assert token_f1("cat cat dog", "cat dog") == pytest.approx(0.8)
    assert token_f1("cat", "dog") == 0


def test_missing_citations_are_not_perfect():
    result = {"citations": [], "sources": [], "answer": "No evidence.", "abstained": True}
    assert answer_metrics(result, "", True)["citation_id_validity"] is None
    result["citations"] = ["invented"]
    assert answer_metrics(result, "", True)["citation_id_validity"] == 0


def test_bootstrap_is_reproducible():
    assert bootstrap_interval([0, 1, 0.5]) == bootstrap_interval([0, 1, 0.5])
    assert bootstrap_interval([1, 1, 1]) == [1, 1]
