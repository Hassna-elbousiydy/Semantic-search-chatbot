from src.embeddings import EmbeddingModel
from src.vector_store import FaissVectorStore


class SemanticRetriever:

    def __init__(self, records):
        self.records = records

        self.embedding_model = EmbeddingModel()

        texts = [
            record["text"]
            for record in records
        ]

        embeddings = self.embedding_model.encode(texts)

        dimension = embeddings.shape[1]

        self.vector_store = FaissVectorStore(
            dimension=dimension
        )

        self.vector_store.add(embeddings)

    def search(self, query: str, k: int = 5):

        query_embedding = self.embedding_model.encode(
            [query]
        )

        scores, indices = self.vector_store.search(
            query_embedding,
            k=k
        )

        results = []

        for score, index in zip(scores, indices):

            if index < 0:
                continue

            result = dict(self.records[index])
            result["score"] = float(score)

            results.append(result)

        return results