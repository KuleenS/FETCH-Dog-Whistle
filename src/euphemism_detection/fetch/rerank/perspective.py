import os

from typing import List, Dict

from perspective import PerspectiveAPI

class ToxicityPerspective:
    def __init__(self):
        self.api = PerspectiveAPI(os.environ["PERSPECTIVE_API_KEY"])
    
    def get_scores(self, sentences: List[str]) -> List[Dict[str, float]]:
        toxicity = []

        for sentence in sentences:
            toxicity.append(self.api.score(sentence))
        
        return toxicity
