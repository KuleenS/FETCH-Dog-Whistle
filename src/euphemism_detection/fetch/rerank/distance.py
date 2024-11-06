from typing import Tuple, List


class DistanceReranker:

    def __init__(self, min_ranker: bool = False):
        self.min_ranker = min_ranker

    def rerank(self, docs: List[Tuple[str, float]], top_k: int):

        docs = sorted(docs, key=lambda x: x[1], reverse=(not self.min_ranker))

        return [x[0] for x in docs[:top_k]]
