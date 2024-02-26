import argparse

import gzip

import os

import json

import re

from typing import List, Tuple

import zlib

import numpy as np

import pandas as pd

import hyperscan

from tqdm import tqdm

from src.euphemism_detection.fetch.embedding.sentencetransformer import SentenceTransformerEmbedder

class Context:
    def __init__(self, id_lookup: Tuple[str, int, int], tweet: str, twitter_file: str, results: List[str]) -> None:
        self.id_lookup = id_lookup
        self.tweet = tweet
        self.twitter_file = twitter_file
        self.results = results

def on_match(id: int, start: int, end: int, flags: int, context: Context) -> None:

    matched_item = context.id_lookup[id][0].decode()

    context.results.append(matched_item)

def main(args): 
    input_files = args.input_files

    sge_id = args.id

    model = SentenceTransformerEmbedder("cuda:0")

    output_folder = os.path.join(args.output_path, sge_id)

    dogwhistle_glossary_df = pd.read_csv(args.dogwhistle_file_path, sep="\t")

    dogwhistle_set = [item for sublist in dogwhistle_glossary_df["Surface Forms"].str.split(";").tolist() for item in sublist]

    dogwhistle_set = list(set([x.lower().strip() for x in dogwhistle_set]))

    dogwhistle_set = [re.escape(x).encode("utf-8") for x in dogwhistle_set]

    db = hyperscan.Database()

    patterns = tuple(zip(dogwhistle_set, range(len(dogwhistle_set)), [hyperscan.HS_FLAG_CASELESS | hyperscan.HS_FLAG_SINGLEMATCH]*len(dogwhistle_set)))

    expressions, ids, flags = zip(*patterns)

    db.compile(
        expressions=expressions, ids=ids, elements=len(patterns), flags=flags
    )

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

                        tweet_text = re.sub(r"https:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})", "", tweet_text)

                    # dealing with gab data
                    elif "body" in tweet:
                        tweet_text = tweet["body"]

                        if isinstance(tweet_text, str):
                            tweet_text= tweet_text.lower().strip()
                        else:
                            tweet_text = ""

                        tweet_text = re.sub(r"http\S+", "", tweet_text)
                    
                    if len(tweet_text) != 0:

                        batch.append(tweet_text)

                        dogwhistles_for_word = []

                        db.scan(tweet_text.encode("utf-8"), match_event_handler=on_match, context = Context(patterns, tweet_text, tweet_file, dogwhistles_for_word))

                        dogwhistles_found.append(dogwhistles_for_word)

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

                        np.savez(os.path.join(output_folder, str(batch_id), f"data.npz"), documents=documents, dogwhistles=dogwhistles_found, embeddings=embeddings)

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

    np.savez(os.path.join(output_folder, str(batch_id), f"data.npz"), documents=documents, dogwhistles=dogwhistles_found, embeddings=embeddings)

                        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path')
    parser.add_argument('--input_files', nargs="+")
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--id')
    args = parser.parse_args()
    main(args)