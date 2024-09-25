import argparse

import gzip

import os

import json

import zlib

import numpy as np

import pandas as pd

try:
    import re2 as re
except ImportError:
    import re

from tqdm import tqdm

from src.euphemism_detection.fetch.embedding.sentencetransformer import (
    SentenceTransformerEmbedder,
)


def main(args):

    input_file = args.input_file

    sge_id = args.id

    if ".parquet" in input_file:
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file, lineterminator="\n")

    model = SentenceTransformerEmbedder("cuda:0")

    output_folder = os.path.join(args.output_path, sge_id)

    batch = []

    dogwhistles_found = []

    documents = []

    embeddings = []

    batch_id = 0

    if args.dogwhistle_file_path is not None:

        dogwhistle_glossary_df = pd.read_csv(args.dogwhistle_file_path, sep="\t")

        dogwhistle_set = [
            item
            for sublist in dogwhistle_glossary_df["Surface Forms"].str.split(";").tolist()
            for item in sublist
        ]

        dogwhistle_set = list(set([x.lower().strip() for x in dogwhistle_set]))

        pattern = re.compile("|".join([re.escape(x) for x in dogwhistle_set]))

    for row in tqdm(df.itertuples()):
        if ".parquet" in input_file:
            text = row.content

            dogwhistle = row.dog_whistle

            text = text.lower().strip()

            batch.append(text)
            
            dogwhistles_found.append(dogwhistle)

        else:
            text = row.tweet

            text = text.lower().strip()
        
            re.sub(r"https:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})", "", text)

            batch.append(text)

            matches = [
                str(x)
                .replace("(?:\\s|$)", "")
                .replace("(?:^|\\s)", "")
                .replace("\\", "")
                for x in re.findall(pattern, text)
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
    parser.add_argument("--input_file")
    parser.add_argument("--id")
    parser.add_argument("--dogwhistle_file_path", required=False)
    args = parser.parse_args()
    main(args)
