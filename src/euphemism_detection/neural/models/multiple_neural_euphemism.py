import csv

from collections import defaultdict

import gzip

import json

import os

import random

import string

from typing import List

import nltk

import numpy as np

from gensim.models.word2vec import Word2Vec as W2V

import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

from tqdm import tqdm

from nltk.corpus import stopwords

from src.euphemism_detection.neural.fitbert import FitBert

nltk.download("stopwords")


class MultiNeuralEuphemismDetector:

    def __init__(
        self,
        given_keywords: List[str],
        data: List[str],
        phrase_candidates: List[str],
        word_2_vec: str,
        output_dir: str,
        model_name: str,
        thres: int,
        data_is_tweets: str,
    ):
        self.given_keywords = given_keywords
        self.data = data
        self.phrase_candidates = phrase_candidates
        self.output_dir = output_dir
        self.word_2_vec = word_2_vec
        self.data_is_tweets = data_is_tweets
        self.thres = thres

        self.model_name = model_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(
            self.device
        )

        self.PAD = self.bert_tokenizer.pad_token
        self.MASK = self.bert_tokenizer.mask_token
        self.CLS = self.bert_tokenizer.cls_token
        self.SEP = self.bert_tokenizer.sep_token

    def filter_phrase(self, phrase_cand, top_words):
        out = []
        top_words = set(top_words)
        block_words = set([y.lower() for y in self.given_keywords])
        # block_words = set([y.lower() for y in drug_formal + ['prescription', 'vendor', 'pain', 'medical', 'synthetic', 'quality']])
        for phrase_i in phrase_cand:
            temp = [x.lower() for x in phrase_i.split()]
            if not any(
                x in top_words for x in temp
            ):  # Euphemisms must contain top 1-gram.
                continue
            if any(
                x in block_words for x in temp
            ):  # Euphemisms should not contain drug formal names and other block names.
                continue
            out.append(phrase_i.lower())
        return out

    def single_MLM(self, message, threshold):

        max_length = self.bert_model.config.max_position_embeddings - 2

        tokens = self.bert_tokenizer(
            message, truncation=True, max_length=max_length, return_tensors="pt"
        )

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
                topk_tokens = self.bert_tokenizer.convert_ids_to_tokens(
                    topk_indices.cpu().numpy()
                )
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
                if j[0] in stopwords.words("english"):
                    continue
                if j[0] in input_keywords:
                    continue
                if j[0] in ["drug", "drugs"]:  # exclude these two for the drug dataset.
                    continue
                if j[0][:2] == "##":  # the '##' by BERT indicates that is not a word.
                    continue
                MLM_score[j[0]] += j[1]
            # print(sgs_i)
            # print([x[0] for x in top_words[:20]])
        out = sorted(MLM_score, key=lambda x: MLM_score[x], reverse=True)
        out_tuple = [[x, MLM_score[x]] for x in out]
        if len(sgs) >= 10:
            print(
                "The percentage of uninformative masked sentences is {:d}/{:d} = {:.2f}%".format(
                    skip_ms_num, len(sgs), float(skip_ms_num) / len(sgs) * 100
                )
            )
        return out, out_tuple, good_sgs

    # def train_word2vec_embed(self, sentences, new_text_file, embed_fn, ft=10, vec_dim=50, window=8):
    #     with open(new_text_file, 'w') as fout:
    #         for i in sentences:
    #             fout.write(i + '\n')
    #     sentences = LineSentence(new_text_file)
    #     sent_cnt = 0
    #     for sentence in sentences:
    #         sent_cnt += 1
    #     print("# of sents: {}".format(sent_cnt))
    #     start = time.time()
    #     model = Word2Vec(sentences, min_count=ft, size=vec_dim, window=window, iter=10, workers=30)
    #     print("embed train time: {}s".format(time.time() - start))
    #     model.wv.save_word2vec_format(embed_fn, binary=False)
    #     return model

    def rank_by_word2vec(self, given_keywords, phrase_cand):
        new_text = []
        for line in self.data:
            for j in phrase_cand:
                line = line.replace(j, "_".join(j.split()))
            new_text.append(line.strip())

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        word2vec_model = W2V.load(self.word_2_vec)

        emb_dict = word2vec_model.wv

        target_vector = []
        seq = []
        for i, seed in enumerate(given_keywords):
            if seed in emb_dict:
                target_vector.append(emb_dict[seed])
                seq.append(i)
        target_vector = np.array(target_vector)
        target_vector_ave = np.sum(target_vector, 0) / len(target_vector)
        out = [
            " ".join(x[0].split("_"))
            for x in word2vec_model.wv.similar_by_vector(target_vector_ave, topn=1000)
            if not any(y in given_keywords for y in x[0].split("_"))
        ]
        return out, []

    def rank_by_spanbert(self, phrase_cand, sgs, drug_formal, thres):
        fb = FitBert(
            model=self.bert_model, tokenizer=self.bert_tokenizer, mask_token=self.MASK
        )
        MLM_score = defaultdict(float)
        temp = sgs if len(sgs) < 10 else tqdm(sgs)
        for i, sgs_i in enumerate(temp):
            if not any(x in sgs_i for x in drug_formal):
                continue
            words, scores = fb.rank_multi(sgs_i, phrase_cand)

            top_words = [[words[i], scores[i]] for i in range(min(len(words), thres))]

            with open(os.path.join(self.output_dir, f"output_words_{i}.csv"), "w") as f:

                csvwrite = csv.writer(f)

                csvwrite.writerow(["word", "score"])

                csvwrite.writerows(top_words)

            # for j in top_words:
            #     if j[0] in string.punctuation:
            #         continue
            #     if j[0] in nltk.corpus.stopwords.words('english'):
            #         continue
            #     if j[0] in drug_formal:
            #         continue
            #     if j[0] in ['drug', 'drugs']:
            #         continue
            #     if j[0][:2] == '##':  # the '##' by BERT indicates that is not a word.
            #         continue
            #     MLM_score[j[0]] += j[1]

        # out = sorted(MLM_score, key=lambda x: MLM_score[x], reverse=True)

        # out_tuple = [[x, MLM_score[x]] for x in out]

        # return out, out_tuple

    def multi_MLM(
        self,
        sentences: List[str],
        given_keywords: List[str],
        top_words: List[str],
        thres: int,
    ):
        print("input cand", len(self.phrase_candidates))

        phrase_cand = self.filter_phrase(self.phrase_candidates, top_words)

        print("output cand", len(phrase_cand))

        phrase_cand, _ = self.rank_by_word2vec(phrase_cand, given_keywords)

        print("output cand", len(phrase_cand))

        self.rank_by_spanbert(phrase_cand, sentences, given_keywords, thres)

        # print("output cand", len(phrase_cand))

        # return phrase_cand, []

    def euphemism_detection(self, input_keywords: List[str], files: List[str]):
        MASK = " [MASK] "
        masked_sentence = []

        N = 0
        K = 2000

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

                            if (
                                "text" in tweet
                                and "lang" in tweet
                                and tweet["lang"] == "en"
                            ):

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
                                    masked_sentence += [
                                        " ".join(temp[:temp_index])
                                        + MASK
                                        + " ".join(temp[temp_index + 1 :])
                                    ]
                                else:
                                    s = int(random.random() * N)
                                    if s < K:
                                        masked_sentence[s] = [
                                            " ".join(temp[:temp_index])
                                            + MASK
                                            + " ".join(temp[temp_index + 1 :])
                                        ]

                except EOFError:
                    print(f"{tweet_file} was not downloaded properly")

            print("[util.py] Generating top candidates...")

        elif self.data_is_tweets == "txt":
            masked_sentence = []

            for tweet_text in self.data:
                temp = nltk.word_tokenize(tweet_text)
                for target in input_keywords:
                    if target not in temp:
                        continue

                    temp_index = temp.index(target)

                    masked_sentence += [
                        " ".join(temp[:temp_index])
                        + MASK
                        + " ".join(temp[temp_index + 1 :])
                    ]
        else:
            masked_sentence = self.data

        ini_top_words, _, good_masked_sentences = self.MLM(
            masked_sentence, input_keywords, thres=self.thres, filter_uninformative=0
        )
        self.multi_MLM(
            good_masked_sentences, input_keywords, ini_top_words, thres=self.thres
        )

    def run(self):
        input_keywords = [x.lower().strip() for x in self.given_keywords]

        self.euphemism_detection(input_keywords, self.data)
