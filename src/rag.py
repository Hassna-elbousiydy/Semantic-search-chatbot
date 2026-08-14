from src.retrieval import SemanticRetriever
from src.generator import AnswerGenerator


class RAGSystem:

    def __init__(self, records, relevance_threshold: float = 0.45):
        print("Building semantic retriever...")
        self.retriever = SemanticRetriever(records)

        print("Loading answer generator...")
        self.generator = AnswerGenerator()

        self.relevance_threshold = relevance_threshold

    def answer(self, question: str, k: int = 3):

        retrieved = self.retriever.search(
            question,
            k=k
        )

        if not retrieved:
            return {
                "question": question,
                "answer": "I don't know based on the provided context.",
                "sources": []
            }

        top_score = retrieved[0]["score"]

        # Reject questions that are not sufficiently related
        # to the indexed knowledge base.
        if top_score < self.relevance_threshold:
            return {
                "question": question,
                "answer": "I don't know based on the provided context.",
                "sources": self._format_sources(retrieved)
            }

        # Lightweight generator: use only the best retrieved passage.
        contexts = [
            retrieved[0]["text"]
        ]

        answer = self.generator.generate(
            question,
            contexts
        )

        return {
            "question": question,
            "answer": answer,
            "sources": self._format_sources(retrieved)
        }

    @staticmethod
    def _format_sources(retrieved):

        sources = []

        for rank, item in enumerate(
            retrieved,
            start=1
        ):
            sources.append({
                "rank": rank,
                "score": item["score"],
                "doc_uuid": item.get("doc_uuid"),
                "chunk_id": item.get("chunk_id"),
                "text": item["text"]
            })

        return sources