import argparse

import os

import logging

from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
import multiprocessing

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)

def main(args):
    model = Word2Vec(sentences=PathLineSentences(args.input_dir),
                         min_count=5,
                         max_vocab_size=500000,
                         vector_size=100,
                         window=5,
                         sample=0.001,
                         workers=multiprocessing.cpu_count() * 2,
                         sg=0,
                         hs=0,
                         epochs=10,
                         compute_loss=True,
                         negative=5,
                         seed=42)

    model.save(os.path.join(args.output_dir, "word2vec.model"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')

    args = parser.parse_args()
    main(args)

