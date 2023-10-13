from .embedder import Embedder

class Searcher:
    def __init__(self, index):
        self.index = index
        self.embedder = Embedder()

    def search(self, query):
        query_embedding = self.embedder.run(query).numpy()
        results = self.index.get_nearest_examples("embeddings", query_embedding, k=5)
        return results
