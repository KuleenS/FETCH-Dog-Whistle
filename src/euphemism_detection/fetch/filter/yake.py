from abc import ABC

from typing import List

import yake

class YAKEFilter(ABC):
    
    def __init__(self, ngram_range: tuple[int, int]):

        self.kw_extractor = yake.KeywordExtractor(n=ngram_range[1])

    def get_most_important_ngrams(self, corpus: List[str], top_k: int):

        keywords = self.kw_extractor.extract_keywords(corpus)

        return keywords[:top_k]