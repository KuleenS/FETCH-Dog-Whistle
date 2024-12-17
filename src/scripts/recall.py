import argparse

import os

import pickle

try:
    import re2 as re
except ImportError:
    import re

import pandas as pd

from tqdm import tqdm


def main(args):
    input_file = args.file

    dogwhistles_found = set()

    if ".parquet" in input_file:
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file, lineterminator="\n")
    
    dogwhistle_glossary_df = pd.read_csv(args.dogwhistle_file_path, sep="\t")

    dogwhistle_set = [
        item
        for sublist in dogwhistle_glossary_df["Surface Forms"].str.split(";").tolist()
        for item in sublist
    ]

    dogwhistle_set = list(set([x.lower().strip() for x in dogwhistle_set]))

    pattern = re.compile("|".join([re.escape(x) for x in dogwhistle_set]))

    for row in tqdm(df.itertuples()):

        tweet_text = row.tweet.lower().strip()

        tweet_text = re.sub(
            r"https:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})",
            "",
            tweet_text,
        )

        matches = [
            str(x)
            .replace("(?:\\s|$)", "")
            .replace("(?:^|\\s)", "")
            .replace("\\", "")
            for x in re.findall(pattern, tweet_text)
        ]

        dogwhistles_found.update(matches)

    with open(os.path.join(args.output_folder, "dogwhistles.recall"), "w") as f:
        pickle.dump(list(dogwhistles_found), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--dogwhistle_file_path")
    parser.add_argument("--output_folder")

    args = parser.parse_args()

    main(args)
