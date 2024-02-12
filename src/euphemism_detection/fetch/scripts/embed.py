import argparse

import csv

import gzip

import os

import json

import re

from typing import List, Tuple

import zlib

import numpy as np

import pandas as pd

import hyperscan

from more_itertools import chunked

from tqdm import tqdm

from src.euphemism_detection.fetch.embedding.sentencetransformer import SentenceTransformerEmbedder

class Context:
    def __init__(self, id_lookup: Tuple[str, int, int], tweet: str, results: List[Tuple[str, str, str]]) -> None:
        self.id_lookup = id_lookup
        self.tweet = tweet
        self.results = results

def on_match(id: int, start: int, end: int, flags: int, context: Context) -> None:

    matched_item = context.id_lookup[id][0].decode()

    context.results.append(matched_item)

def main(args): 
    input_files = args.input_files

    sge_id = args.id
    
    embeddings = []

    documents = []

    dogwhistle_found = []

    batch = []

    model = SentenceTransformerEmbedder("cuda:0")

    output_folder = os.path.join(args.output_path, sge_id)

    dogwhistle_glossary_df = pd.read_csv(args.dogwhistle_file_path, sep="\t")

    dogwhistle_set = [item for sublist in dogwhistle_glossary_df["Surface Forms"].str.split(";").tolist() for item in sublist]

    dogwhistle_set = list(set([x.lower().strip() for x in dogwhistle_set]))

    dogwhistle_set = [re.escape(x).encode("utf-8") for x in dogwhistle_set]

    db = hyperscan.Database(mode=hyperscan.HS_MODE_STREAM)

    patterns = tuple(zip(dogwhistle_set, range(len(dogwhistle_set)), [hyperscan.HS_FLAG_CASELESS | hyperscan.HS_FLAG_SINGLEMATCH]*len(dogwhistle_set)))

    expressions, ids, flags = zip(*patterns)

    db.compile(
        expressions=expressions, ids=ids, elements=len(patterns), flags=flags
    )

    batch_id = 0

    with db.stream(match_event_handler=on_match) as stream:

        for i, tweet_file in tqdm(enumerate(input_files), desc="Twitter Files"):

                try:

                    tweets = gzip.open(tweet_file, "rt")
                
                except (zlib.error, gzip.BadGzipFile):
                    tweets = []

                batch = []

                documents = []

                embeddings = []

                dogwhistle_found = []

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
                                
                                tweet_text = tweet["text"].lower()
                                
                                tweet_text = re.sub(r"https:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})", "", tweet_text)

                            # dealing with gab data
                            elif "body" in tweet:
                                tweet_text = tweet["body"]
        
                                if isinstance(tweet_text, str):
                                    tweet_text= tweet_text.lower()
                                else:
                                    tweet_text = ""
                                
                                tweet_text = re.sub(r"http\S+", "", tweet_text)

                            batch.append(tweet_text)

                            stream.scan(tweet_text.encode("utf-8"), context = Context(patterns, tweet_text, dogwhistle_found))
                            
                            if len(batch) == 32:
                                embeddings_out = model.embed(batch)

                                embeddings.extend(embeddings_out)

                                documents.extend(batch)

                                batch = []

                            if len(documents) % 102400 == 0 and len(documents) != 0:
                                documents = np.array(documents)
                                dogwhistle_found = np.array(dogwhistle_found)
                                embeddings = np.array(embeddings)

                                out = os.path.join(output_folder, str(batch_id))

                                if not os.path.exists(out):
                                    os.makedirs(out, exist_ok=True)

                                np.save(os.path.join(output_folder, str(batch_id), f"documents.npy"), documents)
                                np.save(os.path.join(output_folder, str(batch_id), f"dogwhistles.npy"), dogwhistle_found)
                                np.save(os.path.join(output_folder, str(batch_id), f"embeddings.npy"), embeddings)

                                batch_id += 1

                                documents = []
                                dogwhistle_found = []
                                embeddings = []

                                batch = []

                except EOFError:
                    print(f"{tweet_file} was not downloaded properly")
                
                embeddings_batch = model.embed(batch)

                embeddings.extend(embeddings_batch)

                out = os.path.join(output_folder, str(batch_id))

                if not os.path.exists(out):
                    os.makedirs(out, exist_ok=True)
                
                documents = np.array(documents)
                dogwhistle_found = np.array(dogwhistle_found)
                embeddings = np.array(embeddings)

                np.save(os.path.join(output_folder, str(batch_id), f"documents.npy"), documents)
                np.save(os.path.join(output_folder, str(batch_id), f"dogwhistles.npy"), dogwhistle_found)
                np.save(os.path.join(output_folder, str(batch_id), f"embeddings.npy"), embeddings)
                        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path')
    parser.add_argument('--input_files', nargs="+")
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--id')
    args = parser.parse_args()
    main(args)