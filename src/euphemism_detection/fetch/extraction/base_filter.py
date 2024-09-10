from abc import ABC

from typing import List

class BaseFilter(ABC):
    
    def __init__(self):
        pass
    
    def get_most_important_ngrams(self, corpus: List[str], top_k: int):
        pass