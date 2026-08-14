from src.retrieval import SemanticRetriever
from src.generator import AnswerGenerator


class RAGSystem:

    def __init__(self, records):
        print("Building semantic retriever...")
        self.retriever = SemanticRetriever(records)

        print("Loading answer generator...")
        self.generator = AnswerGenerator()

    def answer(self, question: str, k: int = 3):

        retrieved = self.retriever.search(
            question,
            k=k
        )

        contexts = [
            item["text"]
            for item in retrieved
        ]

        answer = self.generator.generate(
            question,
            contexts
        )

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

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }