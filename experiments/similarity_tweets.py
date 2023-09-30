import argparse

import csv

import os

from typing import List

import evaluate

import pandas as pd

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
            surface_form_comparison_pairs[surface_forms[j].lower().strip()] = comparison_set[i]
    
    with open(os.path.join(args.output_folder, f"comparison_scores_{comparing}_{args.id}.csv"), "w") as f:
        csvwriter = csv.writer(f)

        csvwriter.writerow(["tweet_file", "match", "tweet"]) 

        rouge = evaluate.load('rouge')

        for input_file in input_files:
            with open(input_file, "r") as csvfile:
                datareader = csv.reader(csvfile)
                next(datareader)

                tweets = []
                tweet_files = []
                comparisons = []

                for row in datareader:

                    tweet_file, match, tweet = row.rstrip().split(",")

                    tweets.append(tweet)
                    tweet_files.append(tweet_file)
                    comparisons.append(surface_form_comparison_pairs[match])

                    if len(tweets) > 500:
                        results = rouge.compute(predictions=tweets,
                        references=comparisons, use_aggregator=False)

                        output = zip(tweet_files, tweets, comparisons, results['rouge1'], results['rouge2'], results['rougeL'], results['rougeLsum'])

                        csvwriter.writerows(output)

                        tweets = []
                        tweet_files = []
                        comparisons = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--filtered_tweet_files', nargs='+')
    parser.add_argument('--group_meanings', action='store_true')
    parser.add_argument('--output_folder')
    parser.add_argument('--id')
    args = parser.parse_args()

    main(args)