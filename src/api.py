from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .embedding_model import EmbeddingModel
from .dataset_indexer import DatasetIndexer
from .query_engine import QueryEngine

app = FastAPI()

embedding_model = EmbeddingModel()
indexer = DatasetIndexer(data_path='summaries', data_format='json')
indexer.add_embeddings(embedding_model)
indexer.create_faiss_index()

engine = QueryEngine()

class Query(BaseModel):
    text: str

@app.get("/query")
async def read_query(text: str):
    try:
        response = engine.query(indexer, text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)