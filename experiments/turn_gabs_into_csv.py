import argparse

import csv

import gzip

import json

import os

import re

import zlib

from tqdm import tqdm

def main(args):
    input_files = args.tweet_files

    sge_id = args.id
    
    results = []

    with open(os.path.join(args.output_folder, f"tweets_{sge_id}.txt"), "w") as f:

        writer_csv = csv.writer(f, escapechar='\\')

        writer_csv.writerow(["tweet"])

        for i, tweet_file in tqdm(enumerate(input_files), desc="Twitter Files"):

            try:

                tweets = gzip.open(tweet_file, "rt")
            
            except (zlib.error, gzip.BadGzipFile):
                tweets = []

            try:
                while True:
                    try:
                        tweet = next(tweets)
                    except (IOError, StopIteration, zlib.error):
                        break

                    try:

                        if len(tweet) != 0:

                            if isinstance(tweet, str):
                                try: 
                                    tweet = json.loads(tweet)
                                except json.decoder.JSONDecodeError:
                                    print("Decode failure")
                                    continue

                            if "text" in tweet and "lang" in tweet and tweet["lang"] == "en":
                                
                                tweet_text = tweet["text"].lower()
                                
                                tweet_text = re.sub(r"https:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})", "", tweet_text)

                                results.append([tweet_text.strip()])

                            # dealing with gab data
                            elif "body" in tweet:
                                tweet_text = tweet["body"]
        
                                if isinstance(tweet_text, str):
                                    tweet_text= tweet_text.lower()
                                else:
                                    tweet_text = ""
                                
                                tweet_text = re.sub(r"http\S+", "", tweet_text)

                                results.append([tweet_text.strip()])

                            if len(results) > 500:
                                writer_csv.writerows(results)
                                results = []
                    except:
                        print(f"tweet was bad")

            except EOFError:
                print(f"{tweet_file} was not downloaded properly")
            
            writer_csv.writerows(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tweet_files', nargs='+')
    parser.add_argument('--output_folder')
    parser.add_argument('--id')

    args = parser.parse_args()

    main(args)