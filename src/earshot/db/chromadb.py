import os

import random

from typing import List, Tuple

import chromadb

import numpy as np


from tqdm import tqdm


def group_list(documents, embeddings, dogwhistles, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(documents), group_size):
        yield documents[i : i + group_size], embeddings[
            i : i + group_size
        ], dogwhistles[i : i + group_size]

def group_embeddings(embeddings, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(embeddings), group_size):
        yield embeddings[i : i + group_size]


class ChromaDB:

    def __init__(self, chroma_path: str, collection_name: str, input_folder: str) -> None:
        self.client = chromadb.PersistentClient(path=chroma_path)

        self.collection = self.client.get_or_create_collection(name=collection_name)

        self.collection_name = collection_name

        if self.collection.count() == 0:
            self.load_data(input_folder)

    def load_data(self, input_folder: str):

        files_to_load = []

        for path, subdirs, files in os.walk(input_folder):
            for name in files:
                file_name = os.path.join(path, name)
                if "data.npz" in file_name:
                    files_to_load.append(file_name)

        i = 0

        for file_to_load in files_to_load:
            data = np.load(file_to_load, allow_pickle=True)

            documents = data["documents"]

            embeddings = data["embeddings"]

            dogwhistles = data["dogwhistles"]

            dogwhistles = np.char.replace(dogwhistles, "\\", "")

            for document_batch, embedding_batch, dogwhistle_batch in tqdm(
                group_list(documents, embeddings, dogwhistles, 41665)
            ):
                ids = [f"id{x}" for x in range(i, i+len(document_batch))]

                metadata = [{"uuid": x} for x in ids]

                self.collection.add(
                    documents = [dogwhistle+"____||____"+document for document,dogwhistle in zip(document_batch, dogwhistle_batch)],
                    embeddings=embedding_batch,
                    ids = ids,
                    metadatas=metadata
                )

                i += len(document_batch)

    def get_top_k_documents(
        self, documents_not_to_include: List[str], centroid: List[float], top_k: int
    ) -> List[str]:
        
        total_results = []
        
        for centroid_group in tqdm(group_embeddings(centroid, 41665)):
            result = self.collection.query(
                query_embeddings=centroid_group,
                n_results= top_k,
                where={"uuid": {"$nin": documents_not_to_include}},
            )

            total_results.extend([{"dogwhistle": document[0].split("____||____")[0], "post" : document[0].split("____||____")[1]} for document in result["documents"]])
    
        return total_results 

    def sample_negative_posts(
        self, documents_not_to_include: List[int], number_of_samples: int
    ) -> List[str]:
        
        not_include_set = set(int(x[2:]) for x in documents_not_to_include)

        total_numbers = self.collection.count()

        sampled_set = set()

        while len(sampled_set) < number_of_samples:
            
            rand_index = random.randint(0,total_numbers-1)

            if rand_index not in not_include_set and rand_index not in sampled_set:
                sampled_set.add(f"id{rand_index}")

        sampled_ids = list(sampled_set)

        result = self.collection.get(
            ids=sampled_ids
        )

        posts = [x.split("____||____")[1] for x in result["documents"]]

        return posts

    def calculate_seed_word_centroid(
        self, seed_word: str
    ) -> Tuple[List[int], List[List[float]], List[str]]:
        result = self.collection.get(
            where_document={"$contains": seed_word},
            include=["embeddings", "documents"]
        )

        tweet_ids = [x for x in result["ids"]]

        embeddings = [x for x in result["embeddings"]]
        
        posts = [document.split("____||____")[1] for document in result["documents"]]

        return tweet_ids, embeddings, posts
