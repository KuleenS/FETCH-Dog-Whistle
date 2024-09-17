import os

import random

from typing import List, Tuple

from more_itertools import chunked

from milvus import default_server

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

import numpy as np


from tqdm import tqdm

def group_list(documents, embeddings, dogwhistles, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(documents), group_size):
        yield documents[i:i+group_size], embeddings[i:i+group_size], dogwhistles[i:i+group_size]


class MilvusDB:

    def __init__(self, collection_name: str, dim: int) -> None:

        default_server.start()

        connections.connect(host='127.0.0.1', port=default_server.listen_port)

        print(utility.get_server_version())

        tweet_id = FieldSchema(
            name="tweet_id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id = True,
        )

        if collection_name == "reddit":
            post = FieldSchema(
                name="post",
                dtype=DataType.VARCHAR,
                max_length=19000,
                default_value="Unknown"
            )
        else:
            post = FieldSchema(
                name="post",
                dtype=DataType.VARCHAR,
                max_length=4000,
                default_value="Unknown"
            )
        
        dogwhistle = FieldSchema(
            name="dogwhistle",
            dtype=DataType.VARCHAR,
            max_length=300,
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

        utility.drop_collection(collection_name)

        self.post_lookup = Collection(collection_name, schema, using='default')

        self.collection_name = collection_name
        
        self.dim = dim

        self.files = ["documents", "dogwhistles", "embeddings"]

        self.index_params = {
            "metric_type":"IP",
            "index_type":"FLAT",
        }
    
    def load_data(self, input_folder: str):

        files_to_load = []

        for path, subdirs, files in os.walk(input_folder):
            for name in files:
                file_name = os.path.join(path, name)
                if "data.npz" in file_name:
                    files_to_load.append(file_name)
        
        for file_to_load in files_to_load:
            data = np.load(file_to_load, allow_pickle=True)
            
            documents = data["documents"]
            
            embeddings = data["embeddings"]
            
            dogwhistles = data["dogwhistles"]
            
            dogwhistles = np.char.replace(dogwhistles, '\\', '')
            
            for document_batch, embedding_batch, dogwhistle_batch in tqdm(group_list(documents, embeddings, dogwhistles, 1024)):
                mr = self.post_lookup.insert([document_batch, dogwhistle_batch, embedding_batch])
    
    def create_index(self):
        self.post_lookup.create_index(
            field_name="embeddings", 
            index_params=self.index_params
        )

        self.post_lookup = Collection(self.collection_name)      
        self.post_lookup.load()


    def get_top_k_documents(self, documents_not_to_include: List[int], centroid: List[float], top_k: int) -> List[str]:
        
        search_param = {
            "data": [centroid],
            "anns_field": "embeddings",
            "param": {"metric_type": "IP"},
            "limit": top_k,
            "expr": f"not (tweet_id in {documents_not_to_include})",
            "output_fields": ["post", "dogwhistle"]
        }

        res = self.post_lookup.search(**search_param)

        return res

    def sample_negative_posts(self, documents_not_to_include: List[int], number_of_samples: int) -> List[str]:
        res = self.post_lookup.query(
            expr = f"not (tweet_id in {documents_not_to_include})",
            output_fields = ["post"],
        )

        posts = [x["post"] for x in res]

        return random.sample(posts, number_of_samples)

    
    def calculate_seed_word_centroid(self, seed_word: str) -> Tuple[List[int], List[List[float]], List[str]]:
        res = self.post_lookup.query(
            expr = f'dogwhistle like "{seed_word}%"',
            output_fields = ["tweet_id", "embeddings", "post"],
        )
        
        tweet_ids = [x["tweet_id"] for x in res]
        
        embeddings = [x["embeddings"] for x in res]

        posts = [x["post"] for x in res]

        return tweet_ids, embeddings, posts
    