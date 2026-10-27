from dataclasses import asdict, dataclass
from time import perf_counter
from uuid import uuid4
from src.generation.generator import ExtractiveGenerator, OpenAIGenerator, PROMPT_VERSION
from src.ingestion.chunker import chunk_documents
from src.ingestion.loaders import load_documents
from src.retrieval.hybrid import SearchIndex
from src.retrieval.reranker import Reranker


@dataclass(frozen=True)
class Config:
    retrieval: str = "bm25"
    generation: str = "extractive"
    top_k: int = 5
    candidate_k: int = 20

    def __post_init__(self):
        if self.retrieval not in {"tfidf", "bm25", "dense", "hybrid", "hybrid_rerank"}:
            raise ValueError("Unknown retrieval mode.")
        if self.generation not in {"extractive", "openai"}:
            raise ValueError("Unknown generation mode.")
        if not 1 <= self.top_k <= self.candidate_k <= 100:
            raise ValueError("Require 1 <= top_k <= candidate_k <= 100.")


class RAGPipeline:
    def __init__(self, config=None):
        self.config = config or Config()
        self.index = None
        self.reranker = None
        self.generator = (
            ExtractiveGenerator() if self.config.generation == "extractive" else OpenAIGenerator()
        )

    def index_records(self, documents, chunk=True):
        records = chunk_documents(documents) if chunk else list(documents)
        records = list({d.id: d for d in records}.values())
        neural = self.config.retrieval in {"dense", "hybrid", "hybrid_rerank"}
        # Build before assignment: a failed upload leaves the previous index usable.
        new_index = SearchIndex(records, neural=neural)
        new_reranker = Reranker() if self.config.retrieval == "hybrid_rerank" else None
        self.index, self.reranker = new_index, new_reranker
        return len(records)

    def index_documents(self, file_paths):
        return self.index_records(load_documents(file_paths))

    def retrieve(self, query):
        if self.index is None:
            return []
        cfg = self.config
        mode = "hybrid" if self.reranker else cfg.retrieval
        hits = self.index.search(
            query, mode, cfg.candidate_k if self.reranker else cfg.top_k, cfg.candidate_k
        )
        return self.reranker.rerank(query, hits, cfg.top_k) if self.reranker else hits

    def query(self, user_query):
        if not user_query.strip() or len(user_query) > 4000:
            raise ValueError("Question must contain between 1 and 4,000 characters.")
        start = perf_counter()
        hits = self.retrieve(user_query)
        retrieved = perf_counter()
        generated = self.generator.generate(user_query, hits)
        finished = perf_counter()
        return {
            **asdict(generated),
            "sources": [
                {
                    "id": h.document.id,
                    "content": h.document.text,
                    "score": h.score,
                    "metadata": h.document.metadata,
                }
                for h in hits
            ],
            "trace": {
                "request_id": str(uuid4()),
                "config": asdict(self.config),
                "prompt_version": PROMPT_VERSION,
                "retrieval_ms": (retrieved - start) * 1000,
                "generation_ms": (finished - retrieved) * 1000,
                "total_ms": (finished - start) * 1000,
                "retrieved_ids": [h.document.id for h in hits],
                "model": getattr(self.generator, "model", None),
            },
        }

    ask = query
