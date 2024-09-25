import argparse

import pandas as pd

import csv

import os

from littlebird import BERTweetTokenizer

import spacy

from tqdm import tqdm


def main(args):
    input_file = args.file

    results = []

    tokenizer = BERTweetTokenizer()

    nlp = spacy.load("en_core_web_sm")

    nlp.disable_pipes("ner", "tagger", "parser", "tok2vec", "attribute_ruler")

    if ".parquet" in input_file:
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file, lineterminator="\n")

    with open(os.path.join(args.output_folder, f"tweets.txt"), "w") as f:

        writer_csv = csv.writer(f, escapechar="\\")

        for row in tqdm(df.itertuples()):

            if ".parquet" in input_file:
                doc = nlp(row.content)
            else:
                doc = nlp(row.tweet)

            filtered_text = " ".join(
                [token.lemma_ for token in doc if not token.is_stop]
            )

            normalized_text = " ".join(tokenizer.tokenize(filtered_text)).replace(
                "\n", ""
            )

            results.append([normalized_text])

            if len(results) > 500:
                writer_csv.writerows(results)
                results = []

        writer_csv.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--output_folder")

    args = parser.parse_args()

    main(args)
