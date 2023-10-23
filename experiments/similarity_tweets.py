import argparse

import csv

import os

import re

from typing import List

import evaluate

import pandas as pd

from tqdm import tqdm

def main(args):
    input_files = args.filtered_tweet_files
    
    dogwhistle_glossary_df = pd.read_csv(args.dogwhistle_file_path, sep="\t")

    dogwhistle_set = dogwhistle_glossary_df["Surface Forms"].str.split(";").tolist()

    comparison_set = None

    comparing = None

    if args.group_meanings:
        comparison_set = dogwhistle_glossary_df["Covert (in-group) meaning"].tolist()
        comparing = "group_meanings"
    else:
        comparison_set = dogwhistle_glossary_df["Additional explanation directly from source"].tolist()
        comparing = "explanations"
    
    surface_form_comparison_pairs = dict()

    for i in range(len(dogwhistle_set)):
        surface_forms = dogwhistle_set[i]
        for j in range(len(surface_forms)):
            surface_form_comparison_pairs[re.escape(surface_forms[j].lower().strip()).encode("utf-8")] = comparison_set[i]
    
    with open(os.path.join(args.output_folder, f"comparison_scores_{comparing}_{args.id}.csv"), "w") as f:
        csvwriter = csv.writer(f)

        csvwriter.writerow(["tweet_file", "match", "tweet"]) 

        rouge = evaluate.load('rouge')

        for input_file in input_files:
            with open(input_file, "r") as csvfile:
                datareader = csv.reader(csvfile)
                next(datareader)

                for row in tqdm(datareader):

                    tweet_file, match, tweet = row

                    results = rouge.compute(predictions=[tweet],
                    references=[surface_form_comparison_pairs[match.encode("utf-8")]])

                    output = [tweet_file, tweet, match, results['rouge1'], results['rouge2'], results['rougeL'], results['rougeLsum']]

                    csvwriter.writerow(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--filtered_tweet_files', nargs='+')
    parser.add_argument('--group_meanings', action='store_true')
    parser.add_argument('--output_folder')
    parser.add_argument('--id')
    args = parser.parse_args()

    main(args)