from collections import defaultdict

import pickle

from nltk.tokenize import word_tokenize

import pandas as pd

from sklearn.model_selection import train_test_split


class DogwhistleSplitter:
    def __init__(self, file, **kwargs):
        if file.endswith(".parquet"):
            self.splitter = DogwhistleSplitterReddit(file)
        elif "seen_dogwhistles" in kwargs:
            self.splitter = DogwhistleSplitterOther(file, kwargs["seen_dogwhistles"])
        else:
            raise ValueError(
                "Either provide a parquet if you're splitting the reddit data or provide a pickle for the seen dogwhistles"
            )

    def split(self):
        return self.splitter.split()


class DogwhistleSplitterReddit:
    def __init__(self, reddit_file: str) -> None:

        self.reddit_file = reddit_file

        self.data = pd.read_parquet(reddit_file)

        self.dogwhistle_data = (
            self.data[["dog_whistle", "dog_whistle_root"]]
            .groupby("dog_whistle_root")
            .agg(set)
            .reset_index()
        )

        self.dogwhistle_data["ngrams"] = [
            len(word_tokenize(x)) for x in self.dogwhistle_data["dog_whistle_root"]
        ]

        self.dogwhistle_data["effective_ngrams"] = [
            x if (x < 4) else 4 for x in self.dogwhistle_data["ngrams"]
        ]

    def split(self):
        extrapolating_dogwhistles, given_dogwhistles = train_test_split(
            self.dogwhistle_data,
            test_size=0.2,
            stratify=self.dogwhistle_data["effective_ngrams"],
        )

        extrapolating_dogwhistles_surface_forms = sum(
            [list(x) for x in extrapolating_dogwhistles["dog_whistle"]], []
        )

        given_dogwhistles_surface_forms = sum(
            [list(x) for x in given_dogwhistles["dog_whistle"]], []
        )

        return given_dogwhistles_surface_forms, extrapolating_dogwhistles_surface_forms


class DogwhistleSplitterOther:

    def __init__(self, glossary_path: str, seen_dogwhistles: str) -> None:
        df = pd.read_csv(glossary_path, sep="\t")

        dogwhistle_set = df["Surface Forms"].str.split(";").tolist()

        comparison_set = df["Dogwhistle"].tolist()

        self.dogwhistles = defaultdict(list)

        for i in range(len(dogwhistle_set)):
            self.dogwhistles[comparison_set[i]] = [
                x.strip().lower() for x in dogwhistle_set[i]
            ]

        self.ngrams = {
            x: min([len(word_tokenize(y)) for y in self.dogwhistles[x]])
            for x in comparison_set
        }

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
