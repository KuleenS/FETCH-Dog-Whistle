from abc import ABC

from typing import List

from keybert import KeyBERT



#https://arxiv.org/pdf/2312.00909.pdf
class KeyBERTFilter(ABC):
    
    def __init__(self, model: str, ngram_range: tuple[int, int]):

        self.kw_model = KeyBERT()

        self.ngram_range = ngram_range
    
    def get_most_important_ngrams(self, corpus: List[str], top_k: int, stop_words: str, max_sum: bool, mmr: bool):

        extracted_words = self.kw_model.extract_keywords(corpus, keyphrase_ngram_range=self.ngram_range, stop_words=stop_words, use_maxsum=max_sum, use_mmr = mmr, top_n = top_k)

        return [x[0] for x in extracted_words]