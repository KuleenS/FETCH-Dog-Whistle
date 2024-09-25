import argparse

import gzip

import os

import json

import zlib

import pandas as pd

import numpy as np

try:
    import re2 as re
except ImportError:
    import re

from tqdm import tqdm

from src.euphemism_detection.fetch.embedding.sentencetransformer import (
    SentenceTransformerEmbedder,
)


def main(args):
    input_files = args.input_files

    sge_id = args.id

    model = SentenceTransformerEmbedder("cuda:0")

    output_folder = os.path.join(args.output_path, sge_id)

    dogwhistle_glossary_df = pd.read_csv(args.dogwhistle_file_path, sep="\t")

    dogwhistle_set = [
        item
        for sublist in dogwhistle_glossary_df["Surface Forms"].str.split(";").tolist()
        for item in sublist
    ]

    dogwhistle_set = list(set([x.lower().strip() for x in dogwhistle_set]))

    pattern = re.compile("|".join([re.escape(x) for x in dogwhistle_set]))

    batch_id = 0

    for i, tweet_file in tqdm(enumerate(input_files), desc="Twitter Files"):

        try:

            tweets = gzip.open(tweet_file, "rt")

        except (zlib.error, gzip.BadGzipFile):
            tweets = []

        batch = []

        documents = []

        embeddings = []

        dogwhistles_found = []

        try:
            while True:
                try:
                    tweet = next(tweets)
                except (IOError, StopIteration, zlib.error):
                    break

                if len(tweet) != 0:
                    if isinstance(tweet, str):
                        try:
                            tweet = json.loads(tweet)
                        except json.decoder.JSONDecodeError:
                            print("Decode failure")
                            continue

                    tweet_text = ""

                    if "text" in tweet and "lang" in tweet and tweet["lang"] == "en":

                        tweet_text = tweet["text"].lower().strip()

                        tweet_text = re.sub(
                            r"https:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})",
                            "",
                            tweet_text,
                        )

                    # dealing with gab data
                    elif "body" in tweet:
                        tweet_text = tweet["body"]

                        if isinstance(tweet_text, str):
                            tweet_text = tweet_text.lower().strip()
                        else:
                            tweet_text = ""

                        tweet_text = re.sub(r"http\S+", "", tweet_text)

                    if len(tweet_text) != 0:

                        batch.append(tweet_text)

                        matches = [
                            str(x)
                            .replace("(?:\\s|$)", "")
                            .replace("(?:^|\\s)", "")
                            .replace("\\", "")
                            for x in re.findall(pattern, tweet_text)
                        ]

                        dogwhistles_found.append(matches)

                    if len(batch) == 32:
                        embeddings_out = model.embed(batch)

                        embeddings.extend(embeddings_out)

                        documents.extend(batch)

                        batch = []

                    if len(documents) % 1_024_000 == 0 and len(documents) != 0:
                        documents = np.array(documents)

                        dogwhistles_found = ["||".join(x) for x in dogwhistles_found]

                        dogwhistles_found = np.array(dogwhistles_found)

                        embeddings = np.array(embeddings)

                        out = os.path.join(output_folder, str(batch_id))

                        if not os.path.exists(out):
                            os.makedirs(out, exist_ok=True)

                        np.savez(
                            os.path.join(output_folder, str(batch_id), f"data.npz"),
                            documents=documents,
                            dogwhistles=dogwhistles_found,
                            embeddings=embeddings,
                        )

                        batch_id += 1

                        documents = []
                        dogwhistles_found = []
                        embeddings = []

                        batch = []

        except EOFError:
            print(f"{tweet_file} was not downloaded properly")

    embeddings_batch = model.embed(batch)

    embeddings.extend(embeddings_batch)

    documents.extend(batch)

    out = os.path.join(output_folder, str(batch_id))

    if not os.path.exists(out):
        os.makedirs(out, exist_ok=True)

    documents = np.array(documents)

    dogwhistles_found = ["||".join(x) for x in dogwhistles_found]

    dogwhistles_found = np.array(dogwhistles_found)

    embeddings = np.array(embeddings)

    np.savez(
        os.path.join(output_folder, str(batch_id), f"data.npz"),
        documents=documents,
        dogwhistles=dogwhistles_found,
        embeddings=embeddings,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path")
    parser.add_argument("--input_files", nargs="+")
    parser.add_argument("--dogwhistle_file_path")
    parser.add_argument("--id")
    args = parser.parse_args()
    main(args)
