from typing import List, Tuple

from sklearn.cluster import HDBSCAN

import math


class DiversityReranker:

    def __init__(self):
        pass

    def rerank(self, documents: List[str], embeddings: List[List[float]], top_k: int):
        num_clusters = min(len(embeddings), self.clusters)

        kmeans = HDBSCAN().fit(embeddings)

        proportions = dict(zip(range(num_clusters), [0] * num_clusters))

        for label in kmeans.labels_:
            proportions[label] += 1

        cluster_counts = {}
        for k, v in proportions.items():
            cluster_counts[k] = math.ceil((v / len(embeddings)) * top_k)

        cluster_docs = []
        for i, doc in enumerate(documents):
            label = kmeans.labels_[i]
            if cluster_counts[label] > 0:
                cluster_counts[label] -= 1
                cluster_docs.append(doc)
            else:
                continue

        return cluster_docs
