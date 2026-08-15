from src.retrieval import SemanticRetriever
from src.generator import AnswerGenerator
from src.reranker import CrossEncoderReranker


class RAGSystem:

    def __init__(
        self,
        records,
        relevance_threshold: float = 0.30,
        use_reranker: bool = True,
        reranker_confidence_threshold: float = 0.0,
        candidate_pool: int = 10,
    ):
        print("Building semantic retriever...")
        self.retriever = SemanticRetriever(records)

        self.use_reranker = use_reranker
        self.relevance_threshold = relevance_threshold
        self.reranker_confidence_threshold = (
            reranker_confidence_threshold
        )
        self.candidate_pool = candidate_pool

        if self.use_reranker:
            print("Loading reranker...")
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None

        print("Loading answer generator...")
        self.generator = AnswerGenerator()

    def answer(self, question: str, k: int = 3):

        candidates = self.retriever.search(
            question,
            k=max(k, self.candidate_pool),
        )

        if not candidates:
            return {
                "question": question,
                "answer": (
                    "I don't know based on "
                    "the provided context."
                ),
                "selection_method": "no_candidates",
                "sources": [],
            }

        faiss_top = candidates[0]
        faiss_top_score = faiss_top["score"]

        # First-stage out-of-domain gate.
        if faiss_top_score < self.relevance_threshold:
            return {
                "question": question,
                "answer": (
                    "I don't know based on "
                    "the provided context."
                ),
                "selection_method": "refusal",
                "sources": self._format_sources(
                    candidates[:k]
                ),
            }

        selected = faiss_top
        source_candidates = candidates
        selection_method = "faiss"

        # Second-stage reranking.
        if self.use_reranker:

            reranked = self.reranker.rerank(
                question,
                candidates,
                top_k=len(candidates),
            )

            if reranked:

                reranker_top = reranked[0]

                if (
                    reranker_top["rerank_score"]
                    > self.reranker_confidence_threshold
                ):
                    selected = reranker_top
                    source_candidates = reranked
                    selection_method = "cross_encoder"

                else:
                    selection_method = "faiss_fallback"

        contexts = [
            selected["text"]
        ]

        answer = self.generator.generate(
            question,
            contexts,
        )

        ordered_sources = self._prioritize_selected(
            selected,
            source_candidates,
            k=k,
        )

        return {
            "question": question,
            "answer": answer,
            "selection_method": selection_method,
            "sources": self._format_sources(
                ordered_sources
            ),
        }

    @staticmethod
    def _item_key(item):

        return (
            item.get("source_file"),
            item.get("page"),
            item.get("chunk_id"),
            item.get("doc_uuid"),
        )

    @classmethod
    def _prioritize_selected(
        cls,
        selected,
        candidates,
        k,
    ):

        selected_key = cls._item_key(selected)

        ordered = [selected]

        for item in candidates:

            if cls._item_key(item) == selected_key:
                continue

            ordered.append(item)

            if len(ordered) >= k:
                break

        return ordered[:k]

    @staticmethod
    def _format_sources(retrieved):

        sources = []

        for rank, item in enumerate(
            retrieved,
            start=1,
        ):

            sources.append({
                "rank": rank,

                # Original FAISS similarity.
                "score": item.get("score"),

                "retrieval_score": item.get(
                    "retrieval_score",
                    item.get("score"),
                ),

                # Available only after reranking.
                "rerank_score": item.get(
                    "rerank_score"
                ),

                # SQuAD compatibility.
                "doc_uuid": item.get("doc_uuid"),

                # Scientific metadata.
                "source_file": item.get(
                    "source_file"
                ),
                "page": item.get("page"),
                "chunk_id": item.get(
                    "chunk_id"
                ),

                "text": item["text"],
            })

        return sources
