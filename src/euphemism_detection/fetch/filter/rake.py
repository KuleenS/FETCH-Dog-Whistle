from abc import ABC

from typing import List

from rake_nltk import Rake

class RAKEFilter(ABC):
    
    def __init__(self, ngram_range: tuple[int, int]):

        self.rake = Rake(min_length=ngram_range[0], max_length=ngram_range[1])

    def get_most_important_ngrams(self, corpus: List[str], top_k: int):

        self.rake.extract_keywords_from_text(corpus)

        phrases = self.rake.get_ranked_phrases()

        return phrases[:top_k]