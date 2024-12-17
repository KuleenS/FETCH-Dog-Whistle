import argparse

import csv

import gzip 

import json

import os

import random

from tqdm import tqdm

import argparse

import zlib

import gzip

import json

try:
    import re2 as re
except ImportError:
    import re
    
import pandas as pd

from tqdm import tqdm

def main(args):
    input_files = args.tweet_files

    output_folder = args.output_folder

    slurm_id = args.id

    K = args.k

    look_for_dogwhistles = args.dogwhistle_search
    
    dogwhistle_glossary_df = pd.read_csv(args.dogwhistle_file_path, sep="\t")

    dogwhistle_set = [item for sublist in dogwhistle_glossary_df["Surface Forms"].str.split(";").tolist() for item in sublist]

    dogwhistle_set = list(set([x.lower().strip() for x in dogwhistle_set]))

    pattern = re.compile("|".join([re.escape(x) for x in dogwhistle_set]))

    result = []

    N = 0

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

                if len(tweet) != 0:
                    if isinstance(tweet, str):
                        try: 
                            tweet = json.loads(tweet)
                        except json.decoder.JSONDecodeError:
                            continue
                    
                    tweet_text = None
                
                    if isinstance(tweet, dict) and "text" in tweet and "lang" in tweet and tweet["lang"] == "en":
                        
                        tweet_text = tweet["text"].lower()
                        
                    elif isinstance(tweet, dict) and "body" in tweet:
                        try:
                            tweet_text = tweet["body"].lower()
                        except:
                            tweet_text = None
                
                    if isinstance(tweet_text, str) and tweet_text is not None:  
                        if look_for_dogwhistles:
                            if re.match(pattern, tweet_text) is not None:
                                N += 1

                                if len(result) < K:
                                    result += [tweet_text]
                                else:
                                    s = int(random.random() * N)
                                    if s < K:
                                        result[s] = [tweet_text]
                        else:
                            N += 1

                            if len(result) < K:
                                result += [tweet_text]
                            else:
                                s = int(random.random() * N)
                                if s < K:
                                    result[s] = [tweet_text]
        except EOFError:
            print(f"{tweet_file} was not downloaded properly")
    
    filename = f"sampled_tweets_{slurm_id}.csv"

    if look_for_dogwhistles:
        filename = "dogwhistle_" + filename
            
    with open(os.path.join(output_folder, filename), "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["tweet"])

        for tweet in result:
            csvwriter.writerow([tweet])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--tweet_files', nargs='+')
    parser.add_argument('--output_folder')
    parser.add_argument("--id")
    parser.add_argument("--k", type=int)
    parser.add_argument("--dogwhistle_search", action="store_true", default=False)

    args = parser.parse_args()

    main(args)