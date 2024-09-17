from abc import ABC

from typing import List

import spacy

import pytextrank

from src.euphemism_detection.fetch.extraction.base_filter import BaseFilter

class TextRankFilter(BaseFilter):
    
    def __init__(self, model: str = "en_core_web_md"):
        super().__init__()

        self.nlp = spacy.load(model)

        self.nlp.add_pipe("textrank")

    def get_most_important_ngrams(self, corpus: List[str], top_k: int):

        doc = self.nlp("\n".join(corpus))

        phrases = [x.text for x in list(doc._.phrases)][:top_k]

        return phrases