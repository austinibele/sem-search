from .dataset_indexer import DatasetIndexer
from .embedding_model import EmbeddingModel
from .query_engine import QueryEngine

class Searcher:
    def __init__(self, dir="summaries"):
        embedding_model = EmbeddingModel()
        self.indexer = DatasetIndexer(data_path=dir, data_format='json')
        self.indexer.add_embeddings(embedding_model)
        self.indexer.create_faiss_index()
        self.engine = QueryEngine()

    def search(self, query):
        results = self.engine.query(self.indexer, query)
        return results
                
if __name__ == "__main__":
    searcher = Searcher()
    searcher.search('ds-160')

