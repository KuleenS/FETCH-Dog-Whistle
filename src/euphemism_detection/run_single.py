import argparse

import os

from single_neural_euphemism import SingleNeuralEuphemismDetector

from utils import DogwhistleSplitter


def main(args):
    tweet_files = args.tweet_files

    possible_dogwhistles = args.possible_dogwhistles

    dogwhistle_path = args.dogwhistle_file_path

    splitter = DogwhistleSplitter(dogwhistle_path, possible_dogwhistles)

    given_dogwhistles_surface_forms, extrapolating_dogwhistles_surface_forms = splitter.split()

    euphemism_detector = SingleNeuralEuphemismDetector(given_dogwhistles_surface_forms, tweet_files)

    top_words = euphemism_detector.run()

    with open(os.path.join(args.output_path, "given_dogwhistles"), "w") as f:
        f.write("\n".join(given_dogwhistles_surface_forms))
    
    with open(os.path.join(args.output_path, "extrapolating_dogwhistles"), "w") as f:
        f.write("\n".join(extrapolating_dogwhistles_surface_forms))
    
    with open(os.path.join(args.output_path, "top_words"), "w") as f:
        f.write("\n".join(top_words))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--possible_dogwhistles')
    parser.add_argument('--tweet_files', nargs='+')
    parser.add_argument('--output_path')

    args = parser.parse_args()
    main(args)