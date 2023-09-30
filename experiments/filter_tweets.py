import argparse

import csv

import os

from typing import Tuple, List

import pandas as pd

import hyperscan

class Context:
    def __init__(self, id_lookup: Tuple[str, int, int], tweet: str, twitter_file: str, results: List[Tuple[str, str, str]]) -> None:
        self.id_lookup = id_lookup
        self.tweet = tweet
        self.twitter_file = twitter_file
        self.results = results

def process_twitter_file(twitter_file_path: str) -> List[str]:
    pass


def on_match(id: int, start: int, end: int, flags: int, context: Context) -> None:

    matched_item = context.id_lookup[id][0].decode()

    context.results.append((context.twitter_file, matched_item, context.tweet))

    
def main(args):
    input_files = args.tweet_files
    
    dogwhistle_glossary_df = pd.read_csv(args.dogwhistle_file_path, sep="\t")

    dogwhistle_set = dogwhistle_glossary_df["Surface Forms"].str.split(";").tolist()

    dogwhistle_set = list(set([x.lower().strip() for x in dogwhistle_set]))

    dogwhistle_set = [bytes(x) for x in dogwhistle_set]

    db = hyperscan.Database(mode=hyperscan.HS_MODE_STREAM)

    patterns = tuple(zip(dogwhistle_set, range(len(dogwhistle_set)), [hyperscan.HS_FLAG_CASELESS | hyperscan.HS_FLAG_SINGLEMATCH]*len(dogwhistle_set)))

    expressions, ids, flags = zip(*patterns)

    db.compile(
        expressions=expressions, ids=ids, elements=len(patterns), flags=flags
    )

    results = []

    with open(os.path.join(args.output_folder, f"filtered_tweets_{args.id}.csv"), "w") as f:

        csvwriter = csv.writer(f)

        csvwriter.writerow(["tweet_file", "match", "tweet"]) 

        for i, tweet_file in enumerate(input_files):

            tweets = process_twitter_file(tweet_file)

            with db.stream(match_event_handler=on_match) as stream:

                for tweet in tweets:

                    stream.scan(tweet, context = Context(patterns, tweet, tweet_file, results))
            
                    if len(results) > 500:
                        csvwriter.writerows(results)
                        results = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--tweet_files', nargs='+')
    parser.add_argument('--output_folder')
    parser.add_argument('--id')
    args = parser.parse_args()

    main(args.name)