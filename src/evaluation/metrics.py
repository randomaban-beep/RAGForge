import math
import re
from collections import Counter
import numpy as np
from src.retrieval.hybrid import tokenize


def retrieval_metrics(retrieved, relevance, k=5):
    if k < 1:
        raise ValueError("k must be positive.")
    # Ignore duplicates so repeated citations cannot inflate quality.
    ranked = list(dict.fromkeys(retrieved))[:k]
    relevant = {key for key, grade in relevance.items() if grade > 0}
    if not relevant:
        raise ValueError("Retrieval metrics require at least one positive relevance judgment.")
    gains = [relevance.get(key, 0) for key in ranked]
    dcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(gains))
    ideal = sum(
        (2**grade - 1) / math.log2(i + 2)
        for i, grade in enumerate(sorted(relevance.values(), reverse=True)[:k])
    )
    return {
        f"recall@{k}": len(set(ranked) & relevant) / len(relevant),
        f"precision@{k}": len(set(ranked) & relevant) / k,
        f"mrr@{k}": next((1 / (i + 1) for i, key in enumerate(ranked) if key in relevant), 0),
        f"ndcg@{k}": dcg / ideal,
    }


def token_f1(answer, reference):
    a, b = Counter(tokenize(answer)), Counter(tokenize(reference))
    common = sum((a & b).values())
    if not a or not b:
        return float(a == b)
    precision, recall = common / sum(a.values()), common / sum(b.values())
    return 2 * precision * recall / (precision + recall) if common else 0.0


def answer_metrics(result, reference, expected_abstention=False):
    citations = set(result["citations"])
    available = {source["id"] for source in result["sources"]}
    # ID validity is not evidence entailment or citation accuracy.
    validity = len(citations & available) / len(citations) if citations else None
    return {
        "reference_token_f1": None
        if expected_abstention
        else token_f1(result["answer"], reference),
        "citation_id_validity": validity,
        "abstention_correct": int(result["abstained"] == expected_abstention),
    }


def bootstrap_interval(values, seed=42, samples=2000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("Cannot summarize empty values.")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (samples, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()
