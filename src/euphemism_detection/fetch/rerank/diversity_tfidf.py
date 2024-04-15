from typing import List

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer

import math

class DiversityTFIDFReranker:

    def __init__(self, clusters: int):

        self.clusters = clusters

    
    def rerank(self, documents: List[str], dogwhistles: List[str], top_k: int):
        num_clusters = min(len(documents),self.clusters)

        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(documents)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

        proportions = dict(zip(range(num_clusters), [0]*num_clusters))

        for label in kmeans.labels_:
            proportions[label] += 1
        
        cluster_counts = {}
        for k,v in proportions.items():
            cluster_counts[k] = math.ceil((v/len(embeddings)) * top_k)
        
        cluster_docs = []
        for i,doc in enumerate(dogwhistles):
            label = kmeans.labels_[i]
            if cluster_counts[label] > 0:
                cluster_counts[label] -= 1
                cluster_docs.append(doc)
            else:
                continue
        
        return cluster_docs

