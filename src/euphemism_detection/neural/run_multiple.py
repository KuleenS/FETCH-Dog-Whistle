import argparse

import csv

import os

import pandas as pd

from tqdm import tqdm

from multiple_neural_euphemism import MultiNeuralEuphemismDetector

from utils import DogwhistleSplitter

from metrics import Metrics


def main(args):
    # tweet_files = args.tweet_files

    # possible_dogwhistles = args.possible_dogwhistles

    # dogwhistle_path = args.dogwhistle_file_path

    phrase_path = args.phrase_candidate_folder

    word2vec_path = args.word2vec_file

    # splitter = DogwhistleSplitter(dogwhistle_path, possible_dogwhistles)

    # given_dogwhistles_surface_forms, extrapolating_dogwhistles_surface_forms = splitter.split()

    with open(os.path.join(args.dogwhistle_path, "given_dogwhistles.csv"), "r") as f:
        given_dogwhistles_surface_forms = f.readlines()
    
    with open(os.path.join(args.dogwhistle_path, "extrapolating_dogwhistles_surface_forms.csv"), "r") as f:
        extrapolating_dogwhistles_surface_forms = f.readlines()
    
    extrapolating_dogwhistles_surface_forms = [x.strip().lower() for x in extrapolating_dogwhistles_surface_forms]

    given_dogwhistles_surface_forms = [x.strip().lower() for x in given_dogwhistles_surface_forms]
    
    with open(os.path.join(args.data_path, "sampled_tweets.csv"), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        
        tweet_files = [x[0] for x in reader]

    tweet_files = [x.strip().replace('"', "").replace("'", "").strip("][") for x in tweet_files][1:]

    with open(os.path.join(phrase_path, "AutoPhrase.txt"), "r") as f:
        phrases = f.readlines()

    phrases = [x.split("\t")[1].strip() for x in phrases]

    euphemism_detector = MultiNeuralEuphemismDetector(given_dogwhistles_surface_forms, tweet_files, phrases, word2vec_path, args.data_name, args.output_path, args.model_name, args.threshold, True)

    euphemism_detector.run()

    metrics = Metrics(os.path.join(args.dogwhistle_file_path, "glossary.tsv"))

    files = [os.path.join(args.output_path, x) for x in os.listdir(args.output_path)]
    
    predictions = []

    for file in tqdm(files):
        df = pd.read_csv(file)
        
        prediction = list(df["word"])[::-1]
        
        predictions.append(prediction)
    
    threholds = [50,
    100,
    200,
    400,
    800,
    1600,
    3200,
    6400,
    12800,
    25600]

    results = []

    for threshold in tqdm(threholds):
        top_words = sum([x[:threshold] for x in predictions], [])
        
        precision = metrics.measure_precision(top_words, extrapolating_dogwhistles_surface_forms)

        recall = metrics.measure_recall(top_words, extrapolating_dogwhistles_surface_forms)

        possible_recall = metrics.measure_possible_recall(top_words, extrapolating_dogwhistles_surface_forms, 3)

        results.append((threshold, precision, recall, possible_recall))
    
    print(results)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dogwhistle_file_path')
    # parser.add_argument('--possible_dogwhistles')
    parser.add_argument('--dogwhistle_path')
    parser.add_argument('--data_path')
    parser.add_argument('--phrase_candidate_folder')
    parser.add_argument('--word2vec_file')
    # parser.add_argument('--tweet_files', nargs='+')
    parser.add_argument('--output_path')
    parser.add_argument('--model_name')
    parser.add_argument('--data_name')
    parser.add_argument('--threshold', type=int)


    args = parser.parse_args()
    main(args)