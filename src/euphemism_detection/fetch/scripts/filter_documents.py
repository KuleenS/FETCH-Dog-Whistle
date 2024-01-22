import argparse

import csv

import os

from src.euphemism_detection.metrics import Metrics
from src.euphemism_detection.fetch.filter.tfidf import TFIDF


def main(args):

    corpus = []

    for input_document in args.input_documents:
        with open(input_document, "r") as f:
            csv_reader = csv.reader(f)

            next(csv_reader)

            for row in csv_reader:
                corpus.append(row)

    if args.method == "tfidf":
        filterer = TFIDF((args.ngram_lower_bound, args.ngram_upper_bound))

    else:
        raise ValueError(f"{args.method} does not exist!")
    
    with open(os.path.join(args.dogwhistle_path, "extrapolating_dogwhistles_surface_forms.csv"), "r") as f:
        extrapolating_dogwhistles_surface_forms = f.readlines()
    
    extrapolating_dogwhistles_surface_forms = [x.strip().lower() for x in extrapolating_dogwhistles_surface_forms][1:]

    top_words = filterer.get_most_important_ngrams(top_k=args.topk)

    metrics = Metrics(os.path.join(args.dogwhistle_file_path, "glossary.tsv"))

    precision = metrics.measure_precision(top_words, extrapolating_dogwhistles_surface_forms)

    recall = metrics.measure_recall(top_words, extrapolating_dogwhistles_surface_forms)

    possible_recall = metrics.measure_possible_recall(top_words, extrapolating_dogwhistles_surface_forms, args.ngram_upper_bound)

    print(precision, recall, possible_recall)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_documents", nargs="+")

    parser.add_argument('--dogwhistle_file_path')

    parser.add_argument('--extrapolating_dogwhistles_file_path')

    parser.add_argument('--ngram_lower_bound')

    parser.add_argument('--ngram_upper_bound')

    parser.add_argument('--method')

    parser.add_argument('--top_k')

    args = parser.parse_args()
    main(args)