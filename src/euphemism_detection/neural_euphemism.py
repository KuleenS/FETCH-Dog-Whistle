from argparse import Namespace

from collections import defaultdict

import gzip

import json

import random

from typing import Dict, List

import nltk

import torch

from tqdm import tqdm

from Euphemism.detection import MLM

class NeuralEuphemismDetector: 

    def __init__(self, given_keywords: List[str], data: List[str]):
        self.given_keywords = given_keywords
        self.data = data
        
    
    def euphemism_detection(self, input_keywords, files, ms_limit, filter_uninformative):
        print('\n' + '*' * 40 + ' [Euphemism Detection] ' + '*' * 40)
        print('[util.py] Input Keyword: ', end='')
        print(input_keywords)
        print('[util.py] Extracting masked sentences for input keywords...')
        masked_sentence = []

        files = [files[0]]

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
                            
                    temp = nltk.word_tokenize(tweet_text)
                    for input_keyword_i in input_keywords:
                        if input_keyword_i not in temp:
                            continue
                        temp_index = temp.index(input_keyword_i)
                        masked_sentence += [' '.join(temp[: temp_index]) + ' <mask> ' + ' '.join(temp[temp_index + 1:])]
            
            except EOFError:
                print(f"{tweet_file} was not downloaded properly")
        
        random.shuffle(masked_sentence)
        masked_sentence = masked_sentence[:ms_limit]
        print('[util.py] Generating top candidates...')
        top_words, _, _ = MLM(masked_sentence, input_keywords, thres=5, filter_uninformative=filter_uninformative)
        return top_words
    
    def run(self):
        input_keywords = [x.lower().strip() for x in self.given_keywords]

        top_words = self.euphemism_detection(input_keywords, self.data, ms_limit=2000, filter_uninformative=1)

        print(top_words)
