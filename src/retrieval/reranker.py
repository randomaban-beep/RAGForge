from sentence_transformers import CrossEncoder
import math


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print("Loading reranker...")

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query,
        documents,
        top_k=3
    ):

        if not documents:
            return []

        # ---------------------------------------------
        # Create query-document pairs
        # ---------------------------------------------

        pairs = [
            (
                query,
                document.page_content
            )
            for document in documents
        ]

        # ---------------------------------------------
        # Raw CrossEncoder scores
        # ---------------------------------------------

        raw_scores = self.model.predict(
            pairs
        )

        # ---------------------------------------------
        # Rank using RAW scores
        #
        # This preserves the original ranking.
        # ---------------------------------------------

        ranked = sorted(
            zip(documents, raw_scores),
            key=lambda x: x[1],
            reverse=True
        )

        # ---------------------------------------------
        # Convert raw logits to 0-1 score
        #
        # sigmoid(x) = 1 / (1 + e^-x)
        #
        # This is ONLY for display.
        # ---------------------------------------------

        results = []

        for document, raw_score in ranked[:top_k]:

            raw_score = float(raw_score)

            try:

                normalized_score = (
                    1.0 /
                    (
                        1.0 +
                        math.exp(-raw_score)
                    )
                )

            except OverflowError:

                normalized_score = (
                    0.0
                    if raw_score < 0
                    else 1.0
                )

            results.append(
                (
                    document,
                    normalized_score
                )
            )

        return results