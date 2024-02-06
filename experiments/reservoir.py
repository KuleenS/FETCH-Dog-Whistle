import argparse

import csv

import gzip 

import json

import os

import random

import nltk

from tqdm import tqdm

from utils import DogwhistleSplitter

def main(args):

    files = args.tweet_files

    output_folder = args.output_folder

    possible_dogwhistles = args.possible_dogwhistles

    dogwhistle_path = args.dogwhistle_file_path

    splitter = DogwhistleSplitter(dogwhistle_path, possible_dogwhistles)

    given_dogwhistles_surface_forms, extrapolating_dogwhistles_surface_forms = splitter.split()

    result = []

    N = 0

    K = 2000

    MASK = ' [MASK] '

    for i, tweet_file in tqdm(enumerate(files), desc="Twitter Files"):

        tweets = gzip.open(tweet_file, "rt")

        try:

            for tweet in tqdm(tweets, desc=f"Processing {tweet_file}"):

                tweet = tweet.strip()

                if len(tweet) != 0:
                    if isinstance(tweet, str):
                        try: 
                            tweet = json.loads(tweet)
                        except json.decoder.JSONDecodeError:
                            print("Decode failure")
                            continue
                        
                    tweet_text = ""

                    if "text" in tweet and "lang" in tweet and tweet["lang"] == "en":
                        
                        tweet_text = tweet["text"].lower()
                        
                    elif "body" in tweet:
                        try:
                            tweet_text = tweet["body"].lower()
                        except:
                            print(tweet, " failed")
                            tweet_text = ""
                
                if isinstance(tweet_text, str) and tweet_text is not None:  
                    temp = nltk.word_tokenize(tweet_text)
                    for target in given_dogwhistles_surface_forms:
                        if target not in temp:
                            continue
                        temp_index = temp.index(target)

                        N += 1

                        if len(result) < K:
                            result += [' '.join(temp[: temp_index]) + MASK + ' '.join(temp[temp_index + 1:])]
                        else:
                            s = int(random.random() * N)
                            if s < K:
                                result[s] = [' '.join(temp[: temp_index]) + MASK + ' '.join(temp[temp_index + 1:])]
            
        except EOFError:
            print(f"{tweet_file} was not downloaded properly")
    
    with open(os.path.join(output_folder, "sampled_tweets.csv"), "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["tweet"])

        for tweet in result:
            csvwriter.writerow([tweet])
    
    with open(os.path.join(output_folder, "given_dogwhistles_surface_forms.csv"), "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["dogwhistle"])

        for tweet in given_dogwhistles_surface_forms:
            csvwriter.writerow([tweet])
    
    with open(os.path.join(output_folder, "extrapolating_dogwhistles_surface_forms.csv"), "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["dogwhistle"])

        for tweet in extrapolating_dogwhistles_surface_forms:
            csvwriter.writerow([tweet])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--possible_dogwhistles')
    parser.add_argument('--tweet_files', nargs='+')
    parser.add_argument('--output_folder')
    args = parser.parse_args()

    main(args)