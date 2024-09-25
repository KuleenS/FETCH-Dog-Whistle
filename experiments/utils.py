import argparse

from collections import defaultdict

import os

from nltk.tokenize import word_tokenize

import pickle

import pandas as pd

from sklearn.model_selection import train_test_split

import argparse

from collections import defaultdict, Counter

import os

from nltk.tokenize import word_tokenize

import pickle

import pandas as pd

from sklearn.model_selection import train_test_split


class DogwhistleSplitter:

    def __init__(self, glossary_path: str, seen_dogwhistles: str) -> None:
        df = pd.read_csv(glossary_path, sep="\t")

        dogwhistle_set = df["Surface Forms"].str.split(";").tolist()

        comparison_set = df["Dogwhistle"].tolist()

        self.ngrams = {x: len(word_tokenize(x)) for x in comparison_set}

        self.dogwhistles = defaultdict(list)

        for i in range(len(dogwhistle_set)):
            self.dogwhistles[comparison_set[i]] = [x.strip() for x in dogwhistle_set[i]]

        self.dogwhistle_to_ngrams = dict(zip(comparison_set, self.ngrams))

        self.seen_dogwhistles = pickle.load(open(seen_dogwhistles, "rb"))

        self.seen_dogwhistles = list(
            set([x for x in self.seen_dogwhistles if self.seen_dogwhistles[x] != 0])
        )

    def split(self):
        ngrams = [self.ngrams[x] for x in self.seen_dogwhistles]

        ngrams = [x if (x < 4) else 4 for x in ngrams]

        extrapolating_dogwhistles, given_dogwhistles = train_test_split(
            self.seen_dogwhistles, test_size=0.2, stratify=ngrams
        )

        given_dogwhistles = given_dogwhistles

        given_dogwhistles_surface_forms = []
        extrapolating_dogwhistles_surface_forms = []

        for given_dogwhistle in given_dogwhistles:
            given_dogwhistles_surface_forms.extend(self.dogwhistles[given_dogwhistle])

        for extrapolating_dogwhistle in extrapolating_dogwhistles:
            extrapolating_dogwhistles_surface_forms.extend(
                self.dogwhistles[extrapolating_dogwhistle]
            )

        return given_dogwhistles_surface_forms, extrapolating_dogwhistles_surface_forms


def process_expansions_json(expansion_json_path: str):
    df = pd.read_json(expansion_json_path, orient="records", lines=True)

    df["source"] = df["source"].apply(lambda x: x[0])

    df = pd.concat(
        (df, pd.DataFrame(df["source"].tolist(), columns=["source", "level"])), axis=1
    )

    return df
