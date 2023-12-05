import argparse

import csv

import gzip

import json

import os

import re

from typing import Tuple, List

import zlib

import hyperscan

import pandas as pd

from tqdm import tqdm

class Context:
    def __init__(self, id_lookup: Tuple[str, int, int], tweet: str, twitter_file: str, tweet_date: str, results: List[Tuple[str, str, str]]) -> None:
        self.id_lookup = id_lookup
        self.tweet = tweet
        self.twitter_file = twitter_file
        self.tweet_date = tweet_date
        self.results = results

def on_match(id: int, start: int, end: int, flags: int, context: Context) -> None:

    matched_item = context.id_lookup[id][0].decode()

    context.results.append((context.twitter_file, matched_item, context.tweet, context.tweet_date))

    
def main(args):
    input_files = args.tweet_files
    
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

    results = []

    with open(os.path.join(args.output_folder, f"filtered_tweets_dates_{args.id}.csv"), "w") as f:

        csvwriter = csv.writer(f)

        csvwriter.writerow(["tweet_file", "match", "tweet", "date"]) 

        for i, tweet_file in tqdm(enumerate(input_files), desc="Twitter Files"):

            try:

                tweets = gzip.open(tweet_file, "rt")
            
            except (zlib.error, gzip.BadGzipFile):
                tweets = []

            try:

                with db.stream(match_event_handler=on_match) as stream:

                    while True:
                        try:
                            tweet = next(tweets)
                        except (IOError, StopIteration, zlib.error):
                            break

                        tweet = tweet.strip()

                        if len(tweet) != 0:

                            if isinstance(tweet, str):
                                try: 
                                    tweet = json.loads(tweet)
                                except json.decoder.JSONDecodeError:
                                    print("Decode failure")
                                    continue

                            if "text" in tweet and "lang" in tweet and tweet["lang"] == "en":
                                
                                tweet_text = tweet["text"].lower()

                                tweet_date = tweet["created_at"]
                                
                                tweet_text = re.sub(r"https:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})", "", tweet_text)

                                stream.scan(tweet_text.encode("utf-8"), context = Context(patterns, tweet["text"], tweet_file, tweet_date, results))
                        
                            # dealing with gab data
                            elif "body" in tweet:
                                tweet_text = tweet["body"].lower()

                                tweet_date = tweet["created_at"]
                                
                                tweet_text = re.sub(r"http\S+", "", tweet_text)

                                stream.scan(tweet_text.encode("utf-8"), context = Context(patterns, tweet["body"], tweet_file, tweet_date, results))
                        
                            if len(results) > 500:
                                csvwriter.writerows(results)
                                results = []


            except EOFError:
                print(f"{tweet_file} was not downloaded properly")
            
            csvwriter.writerows(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--tweet_files', nargs='+')
    parser.add_argument('--output_folder')
    parser.add_argument('--id')
    args = parser.parse_args()

    main(args)