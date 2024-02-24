import argparse

import os

from single_neural_euphemism import SingleNeuralEuphemismDetector

from utils import DogwhistleSplitter

from metrics import Metrics

def main(args):
    # possible_dogwhistles = args.possible_dogwhistles

    # dogwhistle_path = args.dogwhistle_file_path

    # splitter = DogwhistleSplitter(dogwhistle_path, possible_dogwhistles)

    # given_dogwhistles_surface_forms, extrapolating_dogwhistles_surface_forms = splitter.split()

    with open(os.path.join(args.dogwhistle_path, "given_dogwhistles.csv"), "r") as f:
        given_dogwhistles_surface_forms = f.readlines()
    
    with open(os.path.join(args.dogwhistle_path, "extrapolating_dogwhistles_surface_forms.csv"), "r") as f:
        extrapolating_dogwhistles_surface_forms = f.readlines()
    
    with open(os.path.join(args.data_path, "sampled_tweets.csv"), "r") as f:
        tweet_files = f.readlines()
    
    given_dogwhistles_surface_forms = [x.strip().lower() for x in given_dogwhistles_surface_forms]

    extrapolating_dogwhistles_surface_forms = [x.strip().lower() for x in extrapolating_dogwhistles_surface_forms]

    tweet_files = [x.strip().replace('"', "").replace("'", "").strip("][") for x in tweet_files][1:]

    euphemism_detector = SingleNeuralEuphemismDetector(given_dogwhistles_surface_forms, tweet_files, args.threshold, args.model_name, True)

    top_words = euphemism_detector.run()

    metrics = Metrics(os.path.join(args.dogwhistle_file_path, "glossary.tsv"))

    precision = metrics.measure_precision(top_words, extrapolating_dogwhistles_surface_forms)

    recall = metrics.measure_recall(top_words, extrapolating_dogwhistles_surface_forms)

    possible_recall = metrics.measure_possible_recall(top_words, extrapolating_dogwhistles_surface_forms, 1)

    print(precision, recall, possible_recall)

    # with open(os.path.join(args.output_path, "given_dogwhistles"), "w") as f:
    #     f.write("\n".join(given_dogwhistles_surface_forms))
    
    # with open(os.path.join(args.output_path, "extrapolating_dogwhistles"), "w") as f:
    #     f.write("\n".join(extrapolating_dogwhistles_surface_forms))
    
    with open(os.path.join(args.output_path, "top_words"), "w") as f:
        f.write("\n".join(top_words))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dogwhistle_file_path')
    # parser.add_argument('--possible_dogwhistles')
    parser.add_argument('--dogwhistle_path')
    parser.add_argument('--data_path')
    parser.add_argument('--model_name')
    # parser.add_argument('--tweet_files', nargs='+')
    parser.add_argument('--output_path')
    parser.add_argument('--threshold', type=int)

    args = parser.parse_args()
    main(args)