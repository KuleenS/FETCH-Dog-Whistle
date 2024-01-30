import argparse

import csv

import gzip

import json

import os

import zlib

from littlebird import BERTweetTokenizer

import spacy

from tqdm import tqdm

def main(args):
    input_files = args.tweet_files

    sge_id = args.id
    
    results = []

    tokenizer = BERTweetTokenizer()

    nlp = spacy.load("en_core_web_md")

    with open(os.path.join(args.output_folder, f"tweets_{sge_id}.txt"), "w") as f:

        writer_csv = csv.writer(f, quoting=csv.QUOTE_NONE, quotechar='', escapechar='\\')

        for i, tweet_file in tqdm(enumerate(input_files), desc="Twitter Files"):

            try:

                tweets = gzip.open(tweet_file, "rt")
            
            except (zlib.error, gzip.BadGzipFile):
                tweets = []
            
            tweet_ids = set()

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
                                if tweet["id_str"] in tweet_ids:
                                    tweet_text = tweet["text"]
                                    tweet_ids.add(tweet["id_str"])
                                else:
                                    tweet_text = ""
                
                            elif "body" in tweet:
                                if tweet["id"] in tweet_ids:
                                    tweet_text = tweet["body"]
                                    tweet_ids.add(tweet["id"])
                                
                                else:
                                    tweet_text = ""

                                if not isinstance(tweet_text, str):
                                    tweet_text = ""

                            if len(tweet_text) != 0:

                                doc = nlp(tweet_text)

                                filtered_text = " ".join([token.lemma_ for token in doc if not token.is_stop])

                                normalized_text = " ".join(tokenizer.tokenize(filtered_text)).replace("\n", "")

                                results.append(normalized_text)

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