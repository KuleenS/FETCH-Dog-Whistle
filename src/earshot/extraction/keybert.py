from abc import ABC

from typing import List

from keybert import KeyBERT

from src.earshot.extraction.base_filter import BaseFilter


class KeyBERTFilter(BaseFilter):

    def __init__(self, ngram_range: tuple[int, int]):
        super().__init__()

        self.kw_model = KeyBERT()

        self.ngram_range = ngram_range

    def get_most_important_ngrams(
        self,
        corpus: List[str],
        top_k: int,
        stop_words: List[str] = None,
        max_sum: bool = False,
        mmr: bool = False,
    ):

        extracted_words = self.kw_model.extract_keywords(
            corpus,
            keyphrase_ngram_range=self.ngram_range,
            stop_words=stop_words,
            use_maxsum=max_sum,
            use_mmr=mmr,
            top_n=top_k,
        )

        extracted_words = sorted(
            sum(extracted_words, []), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [x[0] for x in extracted_words]
