import argparse

import csv

from collections import defaultdict

import pickle

import os

import re

import pandas as pd

from tqdm import tqdm

def main(args):
    input_files = args.filtered_tweet_files

    output_folder = args.output_folder
    
    dogwhistle_glossary_df = pd.read_csv(args.dogwhistle_file_path, sep="\t")

    dogwhistle_set = dogwhistle_glossary_df["Surface Forms"].str.split(";").tolist()

    comparison_set = None

    comparison_set = dogwhistle_glossary_df["Dogwhistle"].tolist()
    
    surface_form_comparison_pairs = dict()

    for i in range(len(dogwhistle_set)):
        surface_forms = dogwhistle_set[i]
        for j in range(len(surface_forms)):
            surface_form_comparison_pairs[re.escape(surface_forms[j].lower().strip()).encode("utf-8")] = comparison_set[i]
    
    d = defaultdict(int)

    for dogwhistle in comparison_set:
        d[dogwhistle] = 0
    
    print(f"Running on {input_files}")

    for input_file in input_files:
        print(f"Processing {input_file}")
        with open(input_file, "r") as csvfile:
            datareader = csv.reader(csvfile)
            next(datareader)

            for row in tqdm(datareader):
                tweet_file, match, tweet, date = row

                dogwhistle_match = surface_form_comparison_pairs[match.encode("utf-8")]

                d[dogwhistle_match] += 1

    with open(os.path.join(output_folder, "recall.pickle"), "wb") as f:
        pickle.dump(d, f)

    print((len(d) - len([x for x in d if d[x]==0]))/len(d))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--filtered_tweet_files', nargs='+')
    parser.add_argument('--output_folder')

    args = parser.parse_args()

    main(args)