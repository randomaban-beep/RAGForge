"""Exact local retrieval. TF-IDF is lexical, never a dense embedding."""

import re
from collections import Counter
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from src.models import Hit

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def reciprocal_rank_fusion(rankings, constant=60):
    if constant < 0:
        raise ValueError("RRF constant must be nonnegative.")
    scores, documents = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            key = hit.document.id
            if key in seen:
                continue
            seen.add(key)
            documents[key] = hit.document
            scores[key] = scores.get(key, 0.0) + 1 / (constant + rank)
    return [
        Hit(documents[key], score)
        for key, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


class SearchIndex:
    def __init__(self, documents, neural=False, embedding_model=EMBEDDING_MODEL):
        self.documents = list(documents)
        if not self.documents or not any(tokenize(d.text) for d in self.documents):
            raise ValueError("Index requires nonempty text documents.")
        if len({d.id for d in self.documents}) != len(self.documents):
            raise ValueError("Document IDs must be unique.")
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
        self.matrix = self.vectorizer.fit_transform(d.text for d in self.documents)
        self.bm25 = BM25Okapi([tokenize(d.text) for d in self.documents])
        # Positive BM25 IDF (Lucene-style) avoids zero/negative scores on small uploads.
        frequencies = Counter(term for freq in self.bm25.doc_freqs for term in freq)
        n = len(self.documents)
        self.bm25.idf = {
            term: float(np.log1p((n - count + 0.5) / (count + 0.5)))
            for term, count in frequencies.items()
        }
        self.encoder = None
        self.model_revision = None
        if neural:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(
                embedding_model,
                device="cpu",
                revision=EMBEDDING_REVISION if embedding_model == EMBEDDING_MODEL else None,
            )
            self.model_revision = getattr(self.encoder[0].auto_model.config, "_commit_hash", None)
            self.vectors = self.encoder.encode(
                [d.text for d in self.documents],
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=64,
            )

    def _rank(self, scores, k, positive_only=False):
        order = np.argsort(-np.asarray(scores), kind="stable")[:k]
        return [
            Hit(self.documents[i], float(scores[i]))
            for i in order
            if not positive_only or scores[i] > 0
        ]

    def search(self, query, mode="bm25", k=5, candidate_k=20):
        if k < 1 or candidate_k < k:
            raise ValueError("Require 1 <= k <= candidate_k.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if mode == "tfidf":
            scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
            return self._rank(scores, k, positive_only=True)
        if mode == "bm25":
            return self._rank(self.bm25.get_scores(tokenize(query)), k, positive_only=True)
        if mode == "dense":
            if self.encoder is None:
                raise ValueError("Dense retrieval requires neural=True and neural dependencies.")
            vector = self.encoder.encode(query, normalize_embeddings=True)
            return self._rank(self.vectors @ vector, k)
        if mode == "hybrid":
            return reciprocal_rank_fusion(
                [
                    self.search(query, "bm25", candidate_k, candidate_k),
                    self.search(query, "dense", candidate_k, candidate_k),
                ]
            )[:k]
        raise ValueError(f"Unknown retrieval mode: {mode}")
