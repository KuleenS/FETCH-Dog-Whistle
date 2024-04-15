from collections import Counter

from typing import List

import numpy as np

from rank_bm25 import BM25Okapi

class BM25Reranker:

    def __init__(self, corpus: List[List[str]], min_ranker: bool = False):

        self.bm25 = BM25Okapi(corpus)

        self.min_ranker = min_ranker
    
    def rerank(self, queries: List[str], dogwhistles: List[str], top_k_okapi: int, top_k_count: int):
        tokenized_queries = [x.split(" ") for x in queries]

        top_reranked = Counter()

        for query in tokenized_queries:
            scores = self.bm25.get_scores(query)

            scores =  np.argsort(scores)

            if self.min_ranker:
                top_n =scores[::-1][:top_k_okapi]
            else:
                top_n = scores[:top_k_okapi]

            for index in top_n:
                top_reranked[index] += 1
        
        most_common_indexes = [x[0] for x in top_reranked.most_common(top_k_count)]

        return [dogwhistles[x] for x in most_common_indexes]
            




