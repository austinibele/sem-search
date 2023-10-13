from transformers import (DPRContextEncoder, DPRContextEncoderTokenizer,
                          DPRQuestionEncoder, DPRQuestionEncoderTokenizer)
from datasets import load_dataset
import torch

torch.set_grad_enabled(False)


class EmbeddingModel:
    def __init__(self, model_type="facebook/dpr-ctx_encoder-single-nq-base"):
        self.ctx_encoder = DPRContextEncoder.from_pretrained(model_type)
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_type)

    def encode(self, text):
        return self.ctx_encoder(**self.ctx_tokenizer(text, return_tensors="pt"))[0][0].numpy()


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


class QueryEngine:
    def __init__(self, model_type="facebook/dpr-question_encoder-single-nq-base"):
        self.q_encoder = DPRQuestionEncoder.from_pretrained(model_type)
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_type)

    def query(self, dataset_indexer, question, k=5):
        question_embedding = self.q_encoder(**self.q_tokenizer(question, return_tensors="pt"))[0][0].numpy()
        scores, retrieved_examples = dataset_indexer.dataset.get_nearest_examples('embeddings', question_embedding, k=k)
        return retrieved_examples["article_summary"]

    def range_search(self, dataset_indexer, question, thresh=0.95):
        question_embedding = self.q_encoder(**self.q_tokenizer(question, return_tensors="pt"))[0][0].numpy()
        faiss_index = dataset_indexer.get_index()
        limits, distances, indices = faiss_index.range_search(x=question_embedding.reshape(1, -1), thresh=thresh)
        return limits, distances, indices


# Usage:
embedding_model = EmbeddingModel()
indexer = DatasetIndexer(data_path='summaries', data_format='json')
indexer.add_embeddings(embedding_model)
indexer.create_faiss_index()

engine = QueryEngine()
print(engine.query(indexer, "Can I get a green card if I'm married to another green card holder?"))
