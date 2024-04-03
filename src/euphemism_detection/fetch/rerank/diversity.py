from typing import List

from sklearn.cluster import KMeans

import math

class DiversityReranker:

    def __init__(self, clusters: int):

        self.clusters = clusters

    
    def rerank(self, returned_embeddings: List[List[float]], top_k: int):
        num_clusters = min(len(returned_embeddings),self.clusters)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(returned_embeddings)

        proportions = dict(zip(range(len(num_clusters)), [0]*len(num_clusters)))

        for label in kmeans.labels_:
            proportions[label] += 1
        
        cluster_counts = {}
        for k,v in proportions.items():
            cluster_counts[k] = math.ceil((v/len(returned_embeddings)) * top_k)
        
        cluster_docs = []
        for i,doc in enumerate(returned_embeddings):
            label = kmeans.labels_[i]
            if cluster_counts[label] > 0:
                cluster_counts[label] -= 1
                cluster_docs.append(doc)
            else:
                continue
        
        return cluster_docs

