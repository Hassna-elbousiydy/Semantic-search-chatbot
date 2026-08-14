import faiss
import numpy as np


class FaissVectorStore:

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings):
        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        self.index.add(embeddings)

    def search(self, query_embedding, k=5):
        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        return scores[0], indices[0]