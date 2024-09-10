import argparse

import pandas as pd

import csv

import os

from littlebird import BERTweetTokenizer

import spacy

from tqdm import tqdm

def main(args):
    input_file = args.reddit_file

    results = []

    tokenizer = BERTweetTokenizer()

    nlp = spacy.load("en_core_web_sm")

    nlp.disable_pipes('ner', 'tagger', 'parser', 'tok2vec', 'attribute_ruler')
    
    df = pd.read_parquet(input_file)

    with open(os.path.join(args.output_folder, f"tweets.txt"), "w") as f:

        writer_csv = csv.writer(f, escapechar='\\')

        for row in tqdm(df.itertuples()):
            
            doc = nlp(row.content)

            filtered_text = " ".join([token.lemma_ for token in doc if not token.is_stop])

            normalized_text = " ".join(tokenizer.tokenize(filtered_text)).replace("\n", "")

            results.append([normalized_text])

            if len(results) > 500:
                writer_csv.writerows(results)
                results = []
        
        writer_csv.writerows(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reddit_file')
    parser.add_argument('--output_folder')

    args = parser.parse_args()

    main(args)