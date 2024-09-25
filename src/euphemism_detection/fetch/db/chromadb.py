from typing import List

import chromadb

import numpy as np


class ChromaDB:

    def __init__(self, persistent_store_path: str):
        self.chroma_client = chromadb.Client()

        self.collection = self.chroma_client.get_or_create_collection(
            name="social_media_posts",
            metadata={"hnsw:space": "cosine"},
            persist_directory=persistent_store_path,
        )

    def add_to_collection(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        dogwhistles: List[str],
        ids: List[str],
    ):
        metadatas = [{"dogwhistle": x} for x in dogwhistles]

        self.collection.add(
            embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids
        )

    def get_top_k_documents(self, centroid: List[float], top_k: int) -> List[str]:
        results = self.collection.query(query_embeddings=centroid, n_results=top_k)

        return results.documents

    def calculate_seed_word_centroid(self, seed_word: str) -> np.ndarray:
        results = self.collection.get(
            where={"dogwhstile": seed_word}, include=["embeddings"]
        )

        return list(np.array(results.embeddings).mean(axis=1))
