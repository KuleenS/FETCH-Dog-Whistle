import argparse

import os

import pandas as pd

from src.neural.models.single_neural_euphemism import (
    SingleNeuralEuphemismDetector,
)

from src.metrics import Metrics


def main(args):

    tweet_files = args.tweet_files
    sampled_file = args.sampled_file
    file = args.file

    with open(os.path.join(args.dogwhistle_path, "given.dogwhistles"), "r") as f:
        given_dogwhistles_surface_forms = f.readlines()

    with open(
        os.path.join(args.dogwhistle_path, "extrapolating.dogwhistles"), "r"
    ) as f:
        extrapolating_dogwhistles_surface_forms = f.readlines()

    given_dogwhistles_surface_forms = [
        x.strip().lower() for x in given_dogwhistles_surface_forms
    ]

    extrapolating_dogwhistles_surface_forms = [
        x.strip().lower() for x in extrapolating_dogwhistles_surface_forms
    ]

    if tweet_files is not None:
        euphemism_detector = SingleNeuralEuphemismDetector(
            given_dogwhistles_surface_forms, tweet_files, 25600, args.model_name, "raw"
        )
    elif sampled_file is not None:
        with open(os.path.join(sampled_file), "r") as f:
            tweet_files = f.readlines()
        tweet_files = [
            x.strip().replace('"', "").replace("'", "").strip("][") for x in tweet_files
        ][1:]

        euphemism_detector = SingleNeuralEuphemismDetector(
            given_dogwhistles_surface_forms,
            tweet_files,
            25600,
            args.model_name,
            "sampled",
        )

    elif file is not None:

        if ".parquet" in file:

            df = pd.read_parquet(file)

            data = list(df["content"])
        
        else:
            df = pd.read_csv(file, lineterminator="\n")

            data = list(df["tweet"])

        euphemism_detector = SingleNeuralEuphemismDetector(
            given_dogwhistles_surface_forms, data, 25600, args.model_name, "txt"
        )
    else:
        raise ValueError("Must specify input")

    top_words = euphemism_detector.run()

    thresholds = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

    for threshold in thresholds:
        top_words_at_t = top_words[:threshold]

        metrics = Metrics(args.dogwhistle_file)

        precision = metrics.measure_precision(
            top_words_at_t, extrapolating_dogwhistles_surface_forms
        )

        recall = metrics.measure_recall(
            top_words_at_t, extrapolating_dogwhistles_surface_forms
        )

        possible_recall = metrics.measure_possible_recall(
            top_words_at_t, extrapolating_dogwhistles_surface_forms, 1
        )

        print(threshold, precision, recall, possible_recall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dogwhistle_file")
    parser.add_argument("--dogwhistle_path")
    parser.add_argument("--model_name")

    parser.add_argument("--tweet_files", nargs="+", required=False, default=None)
    parser.add_argument("--file", required=False, default=None)
    parser.add_argument("--sampled_file", required=False, default=None)

    args = parser.parse_args()
    main(args)
