from typing import List

from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer

class EmbeddingModel:

    def __init__(self, device: str, model_name: str = "all-MiniLM-L6-v2") -> None:

        self.model = SentenceTransformer(model_name, device=device, quantize=True)
    
    def embed(self, batch: List[str]) -> List[List[float]]:

        embeddings = self.model.encode(batch, convert_to_numpy=False, convert_to_tensor=True, normalize_embeddings=True)

        return embeddings.tolist()
