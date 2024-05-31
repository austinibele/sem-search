
from transformers import (DPRContextEncoder, DPRContextEncoderTokenizer)

class EmbeddingModel:
    def __init__(self, model_type="facebook/dpr-ctx_encoder-single-nq-base"):
        self.ctx_encoder = DPRContextEncoder.from_pretrained(model_type)
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_type)

    def encode(self, text):
        return self.ctx_encoder(**self.ctx_tokenizer(text[:min(len(text), 512*3)], return_tensors="pt"))[0][0].numpy() # 512*3 since 512 is max token length, and 3 char/token is conservative
