import argparse

import csv

import os

import re

from typing import List, Tuple

import numpy as np

import pandas as pd

import hyperscan

from more_itertools import chunked

from src.euphemism_detection.fetch.embedding import EmbeddingModel


class Context:
    def __init__(self, id_lookup: Tuple[str, int, int], tweet: str, results: List[Tuple[str, str, str]]) -> None:
        self.id_lookup = id_lookup
        self.tweet = tweet
        self.results = results

def on_match(id: int, start: int, end: int, flags: int, context: Context) -> None:

    matched_item = context.id_lookup[id][0].decode()

    context.results.append(matched_item)

def main(args): 
    input_file = args.tweet_file

    sge_id = args.id
    
    embeddings = []

    documents = []

    dogwhistle_found = []

    batch = []

    model = EmbeddingModel("cuda:0")

    output_folder = os.path.join(args.output_folder, sge_id.id)

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

    with open(input_file, mode ='r') as file:
        csvFile = csv.reader(file)

        results = []

        with db.stream(match_event_handler=on_match) as stream:

            for batch in chunked(csvFile, 32):

                embeddings_out = model.embed(batch)

                embeddings.extend(embeddings_out)

                documents.extend(batch)

                for tweet_text in batch:

                    stream.scan(tweet_text.encode("utf-8"), context = Context(patterns, tweet_text, results))
                            
                if len(documents) % 102400 == 0 and len(documents) != 0:
                    documents = np.array(documents)
                    dogwhistle_found = np.array(dogwhistle_found)
                    embeddings = np.array(embeddings)

                    np.save(os.path.join(output_folder), batch_id, f"documents.npy", documents)
                    np.save(os.path.join(output_folder), batch_id, f"dogwhistles.npy", dogwhistle_found)
                    np.save(os.path.join(output_folder), batch_id, f"embeddings.npy", embeddings)

                    batch_id += 1

                    documents = []
                    dogwhistle_found = []
                    embeddings = []

                    results = []

            embeddings_batch = model.embed(batch)

            embeddings.extend(embeddings_batch)
            
            documents = np.array(documents)
            dogwhistle_found = np.array(dogwhistle_found)
            embeddings = np.array(embeddings)

            np.save(os.path.join(output_folder), batch_id, f"documents.npy", documents)
            np.save(os.path.join(output_folder), batch_id, f"dogwhistles.npy", dogwhistle_found)
            np.save(os.path.join(output_folder), batch_id, f"embeddings.npy", embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path')
    parser.add_argument('--input_path')
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--id', type=int)
    args = parser.parse_args()
    main(args)