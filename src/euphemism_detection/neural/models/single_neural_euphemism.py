from collections import defaultdict

import gzip

import json

import random

import string

from typing import List

import nltk

import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

from tqdm import tqdm

from nltk.corpus import stopwords

nltk.download('stopwords')

class SingleNeuralEuphemismDetector: 

    def __init__(self, given_keywords: List[str], data: List[str], thres : int, model_name: str, data_is_tweets: str):
        self.given_keywords = given_keywords
        self.data = data
        self.data_is_tweets = data_is_tweets
        self.thres = thres

        self.model_name = model_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)

        self.PAD = self.bert_tokenizer.pad_token
        self.MASK = self.bert_tokenizer.mask_token
        self.CLS = self.bert_tokenizer.cls_token
        self.SEP = self.bert_tokenizer.sep_token

    def single_MLM(self, message, threshold):

        max_length = self.bert_model.config.max_position_embeddings - 2

        tokens = self.bert_tokenizer(message, truncation=True, max_length=max_length, return_tensors="pt")

        tokenized = self.bert_tokenizer.tokenize(message, truncation=True)[:max_length]

        tokens.to(self.device)

        with torch.no_grad():
            output = self.bert_model(**tokens)
        
        logits = output.logits

        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)

        out = []

        for idx, token in enumerate(tokenized):
            if token.strip() == self.MASK:
                topk_prob, topk_indices = torch.topk(probs[idx, :], threshold)
                topk_tokens = self.bert_tokenizer.convert_ids_to_tokens(topk_indices.cpu().numpy())
                out = [[topk_tokens[i], float(topk_prob[i])] for i in range(threshold)]

        return out


    def MLM(self, sgs, input_keywords, thres=1, filter_uninformative=1):
        MLM_score = defaultdict(float)

        sgs = [x.replace("[MASK]", self.MASK) for x in sgs]

        temp = sgs if len(sgs) < 10 else tqdm(sgs)
        skip_ms_num = 0
        good_sgs = []

        for sgs_i in temp:
            top_words = self.single_MLM(sgs_i, thres)
            seen_input = 0
            for input_i in input_keywords:
                if input_i in [x[0] for x in top_words[:thres]]:
                    seen_input += 1
            if filter_uninformative == 1 and seen_input < 2:
                skip_ms_num += 1
                continue
            good_sgs.append(sgs_i)
            for j in top_words:
                if j[0] in string.punctuation:
                    continue
                if j[0] in stopwords.words('english'):
                    continue
                if j[0] in input_keywords:
                    continue
                if j[0] in ['drug', 'drugs']:  # exclude these two for the drug dataset.
                    continue
                if j[0][:2] == '##':  # the '##' by BERT indicates that is not a word.
                    continue
                MLM_score[j[0]] += j[1]
            # print(sgs_i)
            # print([x[0] for x in top_words[:20]])
        out = sorted(MLM_score, key=lambda x: MLM_score[x], reverse=True)
        out_tuple = [[x, MLM_score[x]] for x in out]
        if len(sgs) >= 10:
            print('The percentage of uninformative masked sentences is {:d}/{:d} = {:.2f}%'.format(skip_ms_num, len(sgs), float(skip_ms_num)/len(sgs)*100))
        return out, out_tuple, good_sgs
        
    
    def euphemism_detection(self, input_keywords, files, ms_limit, filter_uninformative):
        print('\n' + '*' * 40 + ' [Euphemism Detection] ' + '*' * 40)
        print('[util.py] Input Keyword: ', end='')
        print(input_keywords)
        print('[util.py] Extracting masked sentences for input keywords...')
        masked_sentence = []

        N = 0
        K = ms_limit

        MASK = ' [MASK] '

        if self.data_is_tweets == "raw":

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
                            print(tweet_text)
                            print(type(tweet_text))
                            temp = nltk.word_tokenize(tweet_text)
                            for target in input_keywords:
                                if target not in temp:
                                    continue
                                temp_index = temp.index(target)

                                N += 1

                                if len(masked_sentence) < K:
                                    masked_sentence += [' '.join(temp[: temp_index]) + MASK + ' '.join(temp[temp_index + 1:])]
                                else:
                                    s = int(random.random() * N)
                                    if s < K:
                                        masked_sentence[s] = [' '.join(temp[: temp_index]) + MASK + ' '.join(temp[temp_index + 1:])]
                
                except EOFError:
                    print(f"{tweet_file} was not downloaded properly")
            
            print('[util.py] Generating top candidates...')
        
        elif self.data_is_tweets == "txt":
            masked_sentence = []

            for tweet_text in self.data:
                temp = nltk.word_tokenize(tweet_text)
                for target in input_keywords:
                    if target not in temp:
                        continue
                    
                    temp_index = temp.index(target)

                    masked_sentence += [' '.join(temp[: temp_index]) + MASK + ' '.join(temp[temp_index + 1:])]
        else:
            masked_sentence = self.data
        
        top_words, _, _ = self.MLM(masked_sentence, input_keywords, self.thres, filter_uninformative=filter_uninformative)

        return top_words
    
    def run(self):
        input_keywords = [x.lower().strip() for x in self.given_keywords]

        top_words = self.euphemism_detection(input_keywords, self.data, ms_limit=2000, filter_uninformative=0)

        return top_words
