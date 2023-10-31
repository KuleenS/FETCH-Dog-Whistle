from argparse import Namespace

import pickle

from typing import Dict, List

from src.euphemism_detection.Euphemism.detection import euphemism_detection, evaluate_detection
from src.euphemism_detection.Euphemism.identification import euphemism_identification

class NeuralEuphemismDetector: 

    def __init__(self, given_keywords: Dict[str, List[str]], target_keywords: List[str], data: str, c1: int = 2, c2: int = 0, coarse: bool = 1):
        self.given_keywords = given_keywords
        self.target_keywords = target_keywords
        self.data = data
        self.c1 = c1
        self.c2 = c2
        self.coarse = coarse
    
    def run(self):
        input_keywords = sorted(list(set([y for x in self.given_keywords.values() for y in x])))
        
        target_name = {}
        count = 0

        for keyword in self.target_keywords:
            target_name[keyword.strip()] = count
            count += 1

        with open(self.data, "r") as f:
            all_text = f.readlines()
        
        args = Namespace(c1 = self.c1, c2=self.c2, coarse = self.coarse)

        input_keywords = [x.lower().strip() for x in input_keywords]

        top_words = euphemism_detection(input_keywords, all_text, ms_limit=2000, filter_uninformative=1)

        euphemism_identification(top_words, all_text, self.given_keywords, input_keywords, target_name, args)