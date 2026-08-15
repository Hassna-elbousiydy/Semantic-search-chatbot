from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL
    ):
        print(
            f"Loading reranker: {model_name}"
        )

        self.model = CrossEncoder(
            model_name,
            device="cpu"
        )

    def rerank(
        self,
        question: str,
        retrieved: list,
        top_k: int = 3
    ):

        if not retrieved:
            return []

        pairs = [
            (question, item["text"])
            for item in retrieved
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for item, score in zip(
            retrieved,
            scores
        ):
            result = item.copy()

            # Preserve original FAISS similarity
            result["retrieval_score"] = result["score"]

            # Add Cross-Encoder score
            result["rerank_score"] = float(score)

            reranked.append(result)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]