import argparse

import pandas as pd 

from src.euphemism_detection.metrics import Metrics

def process_expansions_json(expansion_json_path: str):
    df = pd.read_json(expansion_json_path, orient="records", lines=True)

    df["source"] = df["source"].apply(lambda x: x[0])

    df = pd.concat((df, pd.DataFrame(df['source'].tolist(), columns=["source", "level"])), axis=1)

    return df

def main(args):

    with open(args.extrapolating_dogwhistle_path, "r") as f:
        extrapolating_dogwhistles = f.readlines()

    extrapolating_dogwhistles = [x.strip().lower() for x in extrapolating_dogwhistles]

    metrics = Metrics(args.dogwhistle_file_path)

    df = process_expansions_json(args.expansions_path)
    total_found = []

    for x in df.groupby("level"):

        a,b = x

        found_at_stage = [x.lower() for x in b["term"].tolist()]

        total_found += found_at_stage

        print(metrics.measure_precision(total_found, extrapolating_dogwhistles), metrics.measure_recall(total_found, extrapolating_dogwhistles), metrics.measure_possible_recall(total_found, extrapolating_dogwhistles, args.ngrams_possible))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--extrapolating_dogwhistle_path')
    parser.add_argument('--expansions_path')
    parser.add_argument('--ngrams_possible')

    args = parser.parse_args()
    main(args)