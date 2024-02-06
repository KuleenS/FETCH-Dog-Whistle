from collections import defaultdict

import pandas as pd

from typing import List

import nltk

from nltk.tokenize import word_tokenize

nltk.download('punkt')

class Metrics:

    # map set 
   
    def __init__(self, dogwhistle_file_path) -> None:
        self.dogwhistle_file_path = dogwhistle_file_path

        dogwhistles_df = pd.read_csv(dogwhistle_file_path, sep="\t")
        dogwhistle_set = dogwhistles_df["Surface Forms"].str.split(";").tolist()
        comparison_set = dogwhistles_df["Dogwhistle"].tolist()
        
        self.dogwhistle_to_surface = defaultdict(list)

        for i in range(len(dogwhistle_set)):
            self.dogwhistle_to_surface[comparison_set[i]] = [x.strip().lower() for x in dogwhistle_set[i]]

        self.ngrams = {x.strip().lower() : min([len(word_tokenize(y)) for y in self.dogwhistle_to_surface[x]]) for x in comparison_set}

        self.dogwhistles = dict()

        for i in range(len(dogwhistle_set)):
            for surface_form in dogwhistle_set[i]:
                self.dogwhistles[surface_form.strip().lower()] = comparison_set[i].strip().lower()
        
    def measure_precision(self, predicted_dogwhistles: List[str], gold_dogwhistles: List[str]) -> float:
        dogwhistle_map = {x: self.dogwhistles[x] for x in gold_dogwhistles}

        dogwhistles_found = []

        for dogwhistle in predicted_dogwhistles:
            if dogwhistle in dogwhistle_map:
                dogwhistles_found.append(dogwhistle_map[dogwhistle])
        
        if len(predicted_dogwhistles) == 0:
            return 0

        return len(set(dogwhistles_found))/len(predicted_dogwhistles)


    def measure_recall(self, predicted_dogwhistles: List[str], gold_dogwhistles: List[str]) -> float:
        dogwhistle_map = {x: self.dogwhistles[x] for x in gold_dogwhistles}

        dogwhistles_found = []

        for dogwhistle in predicted_dogwhistles:
            if dogwhistle in dogwhistle_map:
                dogwhistles_found.append(dogwhistle_map[dogwhistle])

        return len(set(dogwhistles_found))/len(set(dogwhistle_map.values()))

    def measure_possible_recall(self, predicted_dogwhistles: List[str], gold_dogwhistles: List[str], max_possible_ngrams: int) -> float:
        dogwhistle_map = {x: self.dogwhistles[x] for x in gold_dogwhistles}

        possible = [x for x in set(dogwhistle_map.values()) if self.ngrams[x] <= max_possible_ngrams]

        dogwhistles_found = []

        for dogwhistle in predicted_dogwhistles:
            if dogwhistle in dogwhistle_map:
                dogwhistles_found.append(dogwhistle_map[dogwhistle])

        return len(set(dogwhistles_found))/len(possible)