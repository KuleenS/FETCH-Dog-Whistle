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
    
    def chunk_string(self, string, chunk_size):
        return [string[i:i + chunk_size] for i in range(0, len(string), chunk_size)]

    def get_most_important_ngrams(self, corpus: List[str], top_k: int):

        corpus_total_string = "\n".join(corpus)

        chunked_string = self.chunk_string(corpus_total_string, 95_000)

        phrases = []

        for chunk in chunked_string:
            doc = self.nlp(chunk)

            phrases.extend([(x.text, x.rank) for x in list(doc._.phrases)])
        
        phrases = sorted(phrases, key=lambda x: x[1], reverse=True)[:top_k]

        return [x[0] for x in phrases]
