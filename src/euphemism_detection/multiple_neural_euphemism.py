from collections import defaultdict

import gzip

import json

import os

import time

import random

import string

from typing import Dict, List

from fitbert import FitBert

from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors, Word2Vec

import nltk

import numpy as np

import torch

from transformers import BertForMaskedLM, BertTokenizer

from tqdm import tqdm

from EPD.detection import MLM

class MultiNeuralEuphemismDetector: 

    def __init__(self, given_keywords: List[str], data: List[str], phrase_candidates: List[str], dataset_name: str, output_dir: str, bert_model: str):
        self.given_keywords = given_keywords
        self.data = data
        self.phrase_candidates = phrase_candidates
        self.output_dir = output_dir
        self.bert_model = bert_model
        self.dataset_name = dataset_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def filter_phrase(self, phrase_cand, top_words):
        out = []
        top_words = set(top_words)
        block_words = set([y.lower() for y in self.given_keywords])
        # block_words = set([y.lower() for y in drug_formal + ['prescription', 'vendor', 'pain', 'medical', 'synthetic', 'quality']])
        for phrase_i in phrase_cand:
            temp = [x.lower() for x in phrase_i.split()]
            if not any(x in top_words for x in temp):  # Euphemisms must contain top 1-gram.
                continue
            if any(x in block_words for x in temp):  # Euphemisms should not contain drug formal names and other block names.
                continue
            out.append(phrase_i.lower())
        return out

    def train_word2vec_embed(self, sentences, new_text_file, embed_fn, ft=10, vec_dim=50, window=8):
        with open(new_text_file, 'w') as fout:
            for i in sentences:
                fout.write(i + '\n')
        sentences = LineSentence(new_text_file)
        sent_cnt = 0
        for sentence in sentences:
            sent_cnt += 1
        print("# of sents: {}".format(sent_cnt))
        start = time.time()
        model = Word2Vec(sentences, min_count=ft, size=vec_dim, window=window, iter=10, workers=30)
        print("embed train time: {}s".format(time.time() - start))
        model.wv.save_word2vec_format(embed_fn, binary=False)
        return model

    def rank_by_word2vec(self, given_keywords, phrase_cand):
        new_text = []
        for line in self.data:
            for j in phrase_cand:
                line = line.replace(j, '_'.join(j.split()))
            new_text.append(line.strip())

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir) 

        embed_file = os.path.join(self.output_dir, "embeddings_" + self.dataset_name+".txt")
        new_text_file = os.path.join(self.output_dir, "new_" + self.dataset_name+".txt")

        word2vec_model = self.train_word2vec_embed(new_text, new_text_file, embed_file)
        emb_dict = KeyedVectors.load_word2vec_format(embed_file, binary=False, limit=20000)
        target_vector = []
        seq = []
        for i, seed in enumerate(given_keywords):
            if seed in emb_dict:
                target_vector.append(emb_dict[seed])
                seq.append(i)
        target_vector = np.array(target_vector)
        target_vector_ave = np.sum(target_vector, 0) / len(target_vector)
        out = [' '.join(x[0].split('_')) for x in word2vec_model.wv.similar_by_vector(target_vector_ave, topn=len(emb_dict.vocab)) if '_' in x[0] and not any(y in given_keywords for y in x[0].split('_'))]
        return out, []
    
    def rank_by_spanbert(self, phrase_cand, sgs, drug_formal):
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        bert_model = BertForMaskedLM.from_pretrained(self.bert_model).to(self.device)
        fb = FitBert(model=bert_model, tokenizer=bert_tokenizer, mask_token='[MASK]')
        MLM_score = defaultdict(float)
        temp = sgs if len(sgs) < 10 else tqdm(sgs)
        for sgs_i in temp:
            if not any(x in sgs_i for x in drug_formal):
                continue
            temp = fb.rank_multi(sgs_i, phrase_cand)
            scores = [x / max(temp[1]) for x in temp[1]]
            scores = fb.softmax(torch.tensor(scores).unsqueeze(0)).tolist()[0]
            top_words = [[temp[0][i], scores[i]] for i in range(min(len(temp[0]), 50))]
            for j in top_words:
                if j[0] in string.punctuation:
                    continue
                if j[0] in nltk.corpus.stopwords.words('english'):
                    continue
                if j[0] in drug_formal:
                    continue
                if j[0] in ['drug', 'drugs']:
                    continue
                if j[0][:2] == '##':  # the '##' by BERT indicates that is not a word.
                    continue
                MLM_score[j[0]] += j[1]
            print(sgs_i)
            print([x[0] for x in top_words[:20]])
        out = sorted(MLM_score, key=lambda x: MLM_score[x], reverse=True)
        out_tuple = [[x, MLM_score[x]] for x in out]
        return out, out_tuple

    def multi_MLM(self, sentence: str, given_keywords: List[str], top_words: List[str]):
        phrase_cand = self.filter_phrase(self.phrase_candidates, top_words)

        phrase_cand, _ = self.rank_by_word2vec(phrase_cand, given_keywords)

        phrase_cand, _ = self.rank_by_spanbert(phrase_cand, sentence, given_keywords)
        
        return phrase_cand, []

    def euphemism_detection(self, given_keywords: List[str], all_text: List[str], skip: bool, multi: bool):
        MASK = ' [MASK] '
        masked_sentence = []
        for target in given_keywords:
            for i in tqdm(all_text):
                temp = nltk.word_tokenize(i)
                if target not in temp:
                    continue
                temp_index = temp.index(target)
                masked_sentence += [' '.join(temp[: temp_index]) + MASK + ' '.join(temp[temp_index + 1:])]
        
        random.shuffle(masked_sentence)
        masked_sentence = masked_sentence[:2000]

        if multi:
            top_words, top_words_tuple, _ = MLM(masked_sentence, given_keywords, thres=5, skip_flag=skip)
        else:
            ini_top_words, _, good_masked_sentence = MLM(masked_sentence, given_keywords, thres=5, skip_flag=skip)
            top_words, top_words_tuple = self.multi_MLM(good_masked_sentence, given_keywords, ini_top_words[:100])

        return top_words
    
    def run(self):
        input_keywords = [x.lower().strip() for x in self.given_keywords]

        top_words = self.euphemism_detection(input_keywords, self.data, ms_limit=2000, filter_uninformative=1)

        print(top_words)