from src.models import Hit

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class Reranker:
    def __init__(self, model_name=RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model_name,
            device="cpu",
            revision=RERANKER_REVISION if model_name == RERANKER_MODEL else None,
        )
        self.model_revision = getattr(self.model.model.config, "_commit_hash", None)

    def rerank(self, query, hits, top_k=5):
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        if not hits:
            return []
        scores = self.model.predict([(query, hit.document.text) for hit in hits])
        return sorted(
            [Hit(hit.document, float(score)) for hit, score in zip(hits, scores)],
            key=lambda hit: (-hit.score, hit.document.id),
        )[:top_k]
