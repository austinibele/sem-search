from transformers import (DPRQuestionEncoder, DPRQuestionEncoderTokenizer)
import torch

torch.set_grad_enabled(False)


class QueryEngine:
    def __init__(self, model_type="facebook/dpr-question_encoder-single-nq-base"):
        self.q_encoder = DPRQuestionEncoder.from_pretrained(model_type)
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_type)

    def query(self, dataset_indexer, question, k=5):
        question_embedding = self.q_encoder(**self.q_tokenizer(question, return_tensors="pt"))[0][0].numpy()
        scores, retrieved_examples = dataset_indexer.dataset.get_nearest_examples('embeddings', question_embedding, k=k)
        result_dict = {}
        for i, score in enumerate(scores):
            result_dict[i] = {'score': str(score), 'segment': retrieved_examples['segment'][i], 'summary': retrieved_examples['article_summary'][i]}
        return result_dict

    def range_search(self, dataset_indexer, question, thresh=0.95):
        question_embedding = self.q_encoder(**self.q_tokenizer(question, return_tensors="pt"))[0][0].numpy()
        faiss_index = dataset_indexer.get_index()
        limits, distances, indices = faiss_index.range_search(x=question_embedding.reshape(1, -1), thresh=thresh)
        return limits, distances, indices
