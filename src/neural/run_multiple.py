import argparse

import csv

import os

import pandas as pd

from tqdm import tqdm

from src.neural.models.multiple_neural_euphemism import (
    MultiNeuralEuphemismDetector,
)

from src.metrics import Metrics


def main(args):
    phrase_path = args.phrase_candidate_file

    word2vec_path = args.word2vec_file

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

    with open(phrase_path, "r") as f:
        phrases = f.readlines()

    phrases = [x.split("\t")[1].strip() for x in phrases]

    if tweet_files is not None:
        euphemism_detector = MultiNeuralEuphemismDetector(
            given_dogwhistles_surface_forms,
            tweet_files,
            phrases,
            word2vec_path,
            args.output_path,
            args.model_name,
            25600,
            "raw",
        )

    elif sampled_file is not None:
        with open(sampled_file, "r") as f:
            tweet_files = f.readlines()
        tweet_files = [
            x.strip().replace('"', "").replace("'", "").strip("][") for x in tweet_files
        ][1:]

        euphemism_detector = MultiNeuralEuphemismDetector(
            given_dogwhistles_surface_forms,
            tweet_files,
            phrases,
            word2vec_path,
            args.output_path,
            args.model_name,
            25600,
            "sampled",
        )

    elif file is not None:

        if ".parquet" in file:
            df = pd.read_parquet(file)

            data = list(df["content"])
        else:
            df = pd.read_csv(file, lineterminator="\n")

            data = list(df["tweet"])

        euphemism_detector = MultiNeuralEuphemismDetector(
            given_dogwhistles_surface_forms,
            data,
            phrases,
            word2vec_path,
            args.output_path,
            args.model_name,
            25600,
            "txt",
        )
    else:
        raise ValueError("Must specify input")

    euphemism_detector.run()

    metrics = Metrics(args.dogwhistle_file)

    files = [os.path.join(args.output_path, x) for x in os.listdir(args.output_path)]

    predictions = []

    for file in tqdm(files):
        df = pd.read_csv(file)

        prediction = list(df["word"])[::-1]

        predictions.append(prediction)

    threholds = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

    results = []

    for threshold in tqdm(threholds):
        top_words = sum([x[:threshold] for x in predictions], [])

        precision = metrics.measure_precision(
            top_words, extrapolating_dogwhistles_surface_forms
        )

        recall = metrics.measure_recall(
            top_words, extrapolating_dogwhistles_surface_forms
        )

        possible_recall = metrics.measure_possible_recall(
            top_words, extrapolating_dogwhistles_surface_forms, 3
        )

        results.append((threshold, precision, recall, possible_recall))

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dogwhistle_file")
    parser.add_argument("--dogwhistle_path")

    parser.add_argument("--phrase_candidate_file")
    parser.add_argument("--word2vec_file")
    parser.add_argument("--model_name")

    parser.add_argument("--tweet_files", nargs="+", required=False, default=None)
    parser.add_argument("--file", required=False, default=None)
    parser.add_argument("--sampled_file", required=False, default=None)

    parser.add_argument("--output_path")

    args = parser.parse_args()
    main(args)
