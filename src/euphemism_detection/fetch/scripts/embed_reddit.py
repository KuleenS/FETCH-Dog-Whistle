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

from src.euphemism_detection.fetch.embedding.sentencetransformer import SentenceTransformerEmbedder


def main(args):

    input_file = args.input_file

    sge_id = args.id

    df = pd.read_parquet(input_file)

    model = SentenceTransformerEmbedder("cuda:0")

    output_folder = os.path.join(args.output_path, sge_id)

    batch = []

    dogwhistles_found = []

    documents = []

    embeddings = []

    batch_id = 0

    for row in tqdm(df.itertuples()):

        dogwhistle = row.dog_whistle

        text = row.content

        text = text.lower().strip()

        re.sub(r"https:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})", "", text)

        batch.append(text)

        dogwhistles_found.append(dogwhistle)

        if len(batch) == 32:
            embeddings_out = model.embed(batch)

            embeddings.extend(embeddings_out)

            documents.extend(batch)

            batch = []

        if len(documents) % 1_024_000 == 0 and len(documents) != 0:

            documents = np.array(documents)

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

    embeddings_batch = model.embed(batch)

    embeddings.extend(embeddings_batch)

    documents.extend(batch)

    out = os.path.join(output_folder, str(batch_id))

    if not os.path.exists(out):
        os.makedirs(out, exist_ok=True)

    documents = np.array(documents)

    dogwhistles_found = np.array(dogwhistles_found)

    embeddings = np.array(embeddings)

    np.savez(os.path.join(output_folder, str(batch_id), f"data.npz"), documents=documents, dogwhistles=dogwhistles_found, embeddings=embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path')
    parser.add_argument('--input_file')
    parser.add_argument('--id')
    args = parser.parse_args()
    main(args)