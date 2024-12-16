from collections import defaultdict

import pandas as pd

from typing import List

import nltk

from nltk.tokenize import word_tokenize

nltk.download("punkt")


class Metrics:
    def __init__(self, file: str):
        if file.endswith(".parquet"):
            self.metrics = MetricsReddit(file)
        else:
            self.metrics = MetricsOther(file)

    def measure_precision(
        self, predicted_dogwhistles: List[str], gold_dogwhistles: List[str]
    ) -> float:
        return self.metrics.measure_precision(predicted_dogwhistles, gold_dogwhistles)

    def measure_recall(
        self, predicted_dogwhistles: List[str], gold_dogwhistles: List[str]
    ) -> float:
        return self.metrics.measure_recall(predicted_dogwhistles, gold_dogwhistles)

    def measure_possible_recall(
        self,
        predicted_dogwhistles: List[str],
        gold_dogwhistles: List[str],
        max_possible_ngrams: int,
    ) -> float:
        return self.metrics.measure_possible_recall(
            predicted_dogwhistles, gold_dogwhistles, max_possible_ngrams
        )


class MetricsReddit:
    def __init__(self, reddit_file: str):

        self.reddit_file = reddit_file

        self.data = pd.read_parquet(reddit_file)

        self.dogwhistle_data = (
            self.data[["dog_whistle", "dog_whistle_root"]]
            .groupby("dog_whistle_root")
            .agg(set)
            .reset_index()
        )

        self.ngrams = {
            z: min([len(word_tokenize(y)) for y in list(x) + [z]])
            for z, x in zip(
                self.dogwhistle_data["dog_whistle_root"],
                self.dogwhistle_data["dog_whistle"],
            )
        }

        self.dogwhistles = dict()

        for _, row in self.dogwhistle_data.iterrows():
            for surface_form in row.dog_whistle:
                self.dogwhistles[surface_form.strip().lower()] = row.dog_whistle_root

    def measure_precision(
        self, predicted_dogwhistles: List[str], gold_dogwhistles: List[str]
    ) -> float:
        dogwhistle_map = {
            x.strip().lower(): self.dogwhistles[x.strip().lower()]
            for x in gold_dogwhistles
        }

        predicted_dogwhistles_cleaned = [
            x.strip().lower() for x in predicted_dogwhistles
        ]

        dogwhistles_found = []

        for dogwhistle in predicted_dogwhistles_cleaned:
            if dogwhistle in dogwhistle_map:
                dogwhistles_found.append(dogwhistle_map[dogwhistle])

        if len(predicted_dogwhistles) == 0:
            return 0

        return len(dogwhistles_found) / len(predicted_dogwhistles_cleaned)

    def measure_recall(
        self, predicted_dogwhistles: List[str], gold_dogwhistles: List[str]
    ) -> float:
        dogwhistle_map = {x: self.dogwhistles[x] for x in gold_dogwhistles}

        predicted_dogwhistles_cleaned = [
            x.strip().lower() for x in predicted_dogwhistles
        ]

        dogwhistles_found = []

        for dogwhistle in predicted_dogwhistles_cleaned:
            if dogwhistle in dogwhistle_map:
                dogwhistles_found.append(dogwhistle_map[dogwhistle])

        return len(set(dogwhistles_found)) / len(set(dogwhistle_map.values()))

    def measure_possible_recall(
        self,
        predicted_dogwhistles: List[str],
        gold_dogwhistles: List[str],
        max_possible_ngrams: int,
    ) -> float:
        dogwhistle_map = {x: self.dogwhistles[x] for x in gold_dogwhistles}

        predicted_dogwhistles_cleaned = [
            x.strip().lower() for x in predicted_dogwhistles
        ]

        possible = [
            x
            for x in set(dogwhistle_map.values())
            if self.ngrams[x] <= max_possible_ngrams
        ]

        dogwhistles_found = []

        for dogwhistle in predicted_dogwhistles_cleaned:
            if dogwhistle in dogwhistle_map:
                dogwhistles_found.append(dogwhistle_map[dogwhistle])

        return len(set(dogwhistles_found)) / len(possible)


class MetricsOther:

    def __init__(self, dogwhistle_file_path: str) -> None:
        self.dogwhistle_file_path = dogwhistle_file_path

        dogwhistles_df = pd.read_csv(dogwhistle_file_path, sep="\t")

        dogwhistle_set = dogwhistles_df["Surface Forms"].str.split(";").tolist()

        comparison_set = [
            x.strip().lower() for x in dogwhistles_df["Dogwhistle"].tolist()
        ]

        self.dogwhistle_to_surface = defaultdict(list)

        for i in range(len(dogwhistle_set)):
            self.dogwhistle_to_surface[comparison_set[i]] = set(
                [x.strip().lower() for x in dogwhistle_set[i]] + [comparison_set[i]]
            )

        self.ngrams = {
            x: min([len(word_tokenize(y)) for y in self.dogwhistle_to_surface[x]])
            for x in comparison_set
        }

        self.dogwhistles = dict()

        for i in range(len(dogwhistle_set)):
            for surface_form in dogwhistle_set[i]:
                self.dogwhistles[surface_form.strip().lower()] = comparison_set[i]

            self.dogwhistles[comparison_set[i]] = comparison_set[i]

    def measure_precision(
        self, predicted_dogwhistles: List[str], gold_dogwhistles: List[str]
    ) -> float:
        dogwhistle_map = {
            x.strip().lower(): self.dogwhistles[x.strip().lower()]
            for x in gold_dogwhistles
        }

        predicted_dogwhistles_cleaned = [
            x.strip().lower() for x in predicted_dogwhistles
        ]

        dogwhistles_found = []

        for dogwhistle in predicted_dogwhistles_cleaned:
            if dogwhistle in dogwhistle_map:
                dogwhistles_found.append(dogwhistle_map[dogwhistle])

        if len(predicted_dogwhistles) == 0:
            return 0

        return len(dogwhistles_found) / len(predicted_dogwhistles_cleaned)

    def measure_recall(
        self, predicted_dogwhistles: List[str], gold_dogwhistles: List[str]
    ) -> float:
        dogwhistle_map = {x: self.dogwhistles[x] for x in gold_dogwhistles}

        predicted_dogwhistles_cleaned = [
            x.strip().lower() for x in predicted_dogwhistles
        ]

        dogwhistles_found = []

        for dogwhistle in predicted_dogwhistles_cleaned:
            if dogwhistle in dogwhistle_map:
                dogwhistles_found.append(dogwhistle_map[dogwhistle])

        return len(set(dogwhistles_found)) / len(set(dogwhistle_map.values()))

    def measure_possible_recall(
        self,
        predicted_dogwhistles: List[str],
        gold_dogwhistles: List[str],
        max_possible_ngrams: int,
    ) -> float:
        dogwhistle_map = {x: self.dogwhistles[x] for x in gold_dogwhistles}

        predicted_dogwhistles_cleaned = [
            x.strip().lower() for x in predicted_dogwhistles
        ]

        possible = [
            x
            for x in set(dogwhistle_map.values())
            if self.ngrams[x] <= max_possible_ngrams
        ]

        dogwhistles_found = []

        for dogwhistle in predicted_dogwhistles_cleaned:
            if dogwhistle in dogwhistle_map:
                dogwhistles_found.append(dogwhistle_map[dogwhistle])

        return len(set(dogwhistles_found)) / len(possible)
