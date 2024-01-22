import os

from typing import List, Tuple

import milvus

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

import numpy as np

class MilvusDB:

    def __init__(self, collection_name: str, dim: int) -> None:
        milvus.start()
        connections.connect("default", host="localhost", port="19530")

        tweet_id = FieldSchema(
            name="tweet_id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id = True,
        )

        post = FieldSchema(
            name="post",
            dtype=DataType.VARCHAR,
            max_length=400,
            default_value="Unknown"
        )

        dogwhistle = FieldSchema(
            name="dogwhistle",
            dtype=DataType.VARCHAR,
            max_length=70,
        )

        embedding = FieldSchema(
            name="embeddings", 
            dtype=DataType.FLOAT_VECTOR, 
            dim=dim
        )

        schema = CollectionSchema(
            fields=[tweet_id, post, dogwhistle, embedding],
            description="Semantic Tweet Lookup",
        )

        self.post_lookup = Collection(collection_name, schema, using='default')

        self.collection_name = collection_name
        
        self.dim = dim

        self.files = ["documents.npy", "dogwhistles.npy", "book_intro.npy", "embeddings.npy"]

        self.index_params = {
            "metric_type":"IP",
            "index_type":"DISKANN",
        }
    
    def load_data(self, input_folder: str):

        batches = [os.path.join(input_folder, x) for x in os.path.listdir(input_folder)]

        task_ids = []

        for batch in batches:
            task_id = utility.do_bulk_insert(
                collection_name=self.collection_name,
                files=[os.path.join(batch, x) for x in self.files]
            )

            task_ids.append(task_id)
        
        utility.wait_for_index_building_complete(self.collection_name)
    
    def create_index(self):
        self.post_lookup.create_index(
            field_name="embeddings", 
            index_params=self.index_params
        )

        utility.index_building_progress(self.collection_name)
        

    def get_top_k_documents(self, documents_not_to_include: List[int], centroid: List[float], top_k: int) -> List[str]:
        
        search_param = {
            "data": [[0.1, 0.2]],
            "anns_field": "embeddings",
            "param": {"metric_type": "IP"},
            "limit": top_k,
            "expr": f"not (tweet_id in {documents_not_to_include})",
        }

        res = self.post_lookup.search(**search_param)

        return res
    
    def calculate_seed_word_centroid(self, seed_word: str) -> Tuple[np.ndarray, np.ndarray]:
        res = self.post_lookup.query(
            expr = f"dogwhistle == {seed_word}",
            output_fields = ["tweet_id", "embeddings"],
        )

        return res["tweet_id"], np.array(res["embeddings"]).mean(axis=1)
    
    