from typing import List

import yake

from src.euphemism_detection.fetch.extraction.base_filter import BaseFilter


class YAKEFilter(BaseFilter):

    def __init__(self, ngram_range: tuple[int, int]):
        super().__init__()

        self.kw_extractor = yake.KeywordExtractor(n=ngram_range[1])

    def get_most_important_ngrams(self, corpus: List[str], top_k: int):

        keywords = self.kw_extractor.extract_keywords("\n".join(corpus))

        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)

        return [x[0] for x in keywords]
