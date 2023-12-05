import pandas as pd

from typing import List

from nltk.tokenize import word_tokenize


class Metrics:
   
    def __init__(self, dogwhistle_file_path) -> None:
        self.dogwhistle_file_path = dogwhistle_file_path

        dogwhistles_df = pd.read_csv(dogwhistle_file_path, sep="\t")
        dogwhistle_set = dogwhistles_df["Surface Forms"].str.split(";").tolist()
        comparison_set = dogwhistles_df["Dogwhistle"].tolist()

        self.dogwhistles = dict()

        for i in range(len(dogwhistle_set)):
            for surface_form in dogwhistle_set[i]:
                self.dogwhistles[surface_form.strip().lower()] = comparison_set[i].strip().lower()
        
    def measure_precision(self, found_dogwhistles: List[str], dogwhistles_to_find: List[str]) -> float:
        dogwhistle_map = {x: self.dogwhistles[x] for x in dogwhistles_to_find}

        dogwhistles_found = []

        for dogwhistle in found_dogwhistles:
            if dogwhistle in dogwhistle_map:
                dogwhistles_found.append(dogwhistle_map[dogwhistle])

        return len(dogwhistles_found)/len(found_dogwhistles)


    def measure_recall(self, found_dogwhistles: List[str], dogwhistles_to_find: List[str]) -> float:
        dogwhistle_map = {x: self.dogwhistles[x] for x in dogwhistles_to_find}

        dogwhistles_found = []

        for dogwhistle in found_dogwhistles:
            if dogwhistle in dogwhistle_map:
                dogwhistles_found.append(dogwhistle_map[dogwhistle])

        return len(dogwhistles_found)/len(set(dogwhistle_map.values()))

    def measure_possible_recall(self, found_dogwhistles: List[str], dogwhistles_to_find: List[str], max_possible_ngrams: int) -> float:
        dogwhistle_map = {x: self.dogwhistles[x] for x in dogwhistles_to_find}

        possible = [x for x in set(dogwhistle_map.values()) if len(word_tokenize(x)) <= max_possible_ngrams]

        dogwhistles_found = []

        for dogwhistle in found_dogwhistles:
            if dogwhistle in dogwhistle_map:
                dogwhistles_found.append(dogwhistle_map[dogwhistle])

        return len(dogwhistles_found)/len(possible)