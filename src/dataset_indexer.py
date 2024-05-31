from datasets import load_dataset
import torch

torch.set_grad_enabled(False)

class DatasetIndexer:
    def __init__(self, data_path, split_range='train', data_format='json', embedding_column='embeddings'):
        if data_format == 'json':
            self.dataset = load_dataset('json', data_files={'train': f'{data_path}/*.json'}, split=split_range)
        else:
            self.dataset = load_dataset(data_path, split=split_range)
        self.embedding_column = embedding_column

    def add_embeddings(self, embedding_model):
        with torch.no_grad(): 
            self.dataset = self.dataset.map(lambda x: {'embeddings': embedding_model.encode(x["article_summary"])})

    def create_faiss_index(self):
        self.dataset.add_faiss_index(column=self.embedding_column)

    def save_faiss_index(self, filename):
        self.dataset.save_faiss_index(self.embedding_column, filename)

    def load_faiss_index(self, filename):
        self.dataset.load_faiss_index(self.embedding_column, filename)

    def get_index(self):
        return self.dataset.get_index(self.embedding_column).faiss_index

