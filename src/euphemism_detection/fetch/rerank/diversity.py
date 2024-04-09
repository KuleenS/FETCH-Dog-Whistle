from typing import List, Tuple

from sklearn.cluster import KMeans

import math

class DiversityReranker:

    def __init__(self, clusters: int):

        self.clusters = clusters

    
    def rerank(self, docu_embeds: List[Tuple[str, List[float]]], top_k: int):
        
        embeddings = [x[1] for x in docu_embeds]
        
        documents = [x[0] for x in docu_embeds]
        
        num_clusters = min(len(embeddings),self.clusters)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

        proportions = dict(zip(range(num_clusters), [0]*num_clusters))

        for label in kmeans.labels_:
            proportions[label] += 1
        
        cluster_counts = {}
        for k,v in proportions.items():
            cluster_counts[k] = math.ceil((v/len(embeddings)) * top_k)
        
        cluster_docs = []
        for i,doc in enumerate(documents):
            label = kmeans.labels_[i]
            if cluster_counts[label] > 0:
                cluster_counts[label] -= 1
                cluster_docs.append(doc)
            else:
                continue
        
        return cluster_docs

