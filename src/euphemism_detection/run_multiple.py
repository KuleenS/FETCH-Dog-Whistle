import argparse

import os

from multiple_neural_euphemism import MultiNeuralEuphemismDetector

from utils import DogwhistleSplitter


def main(args):
    tweet_files = args.tweet_files

    possible_dogwhistles = args.possible_dogwhistles

    dogwhistle_path = args.dogwhistle_file_path

    phrase_path = args.phrase_candidate_folder

    word2vec_path = args.word2vec_file

    splitter = DogwhistleSplitter(dogwhistle_path, possible_dogwhistles)

    given_dogwhistles_surface_forms, extrapolating_dogwhistles_surface_forms = splitter.split()


    with open(os.path.join(phrase_path, "final_quality_salient.txt"), "r") as f:
        salient_phrases = f.readlines()

    with open(os.path.join(phrase_path, "final_quality_unigrams.txt"), "r") as f:
        salient_unigrams = f.readlines()

    with open(os.path.join(phrase_path, "token_mapping.txt"), "r") as f:
        token_mapping = f.readlines()

    salient_phrases = [x.strip().split("\t")[1].split(" ") for x in salient_phrases]
    salient_unigrams = [x.strip().split("\t")[1] for x in salient_unigrams]

    token_mapping = [x.split(" ") for x in token_mapping]

    token_mapping = {x[0] : x[1] for x in token_mapping}

    for i in range(len(salient_phrases)):
        salient_phrase = salient_phrases[i]

        for j in range(len(salient_phrase)):
            salient_phrases[i][j] = token_mapping[salient_phrases]
    
    for i in range(len(salient_unigrams)):
        salient_unigrams[i] = token_mapping[salient_unigrams[i]]
    
    phrases = []

    for i in range(len(salient_phrases)):
        phrases.append(" ".join(salient_phrases[i]))
    
    phrases += salient_unigrams

    euphemism_detector = MultiNeuralEuphemismDetector(given_dogwhistles_surface_forms, tweet_files, phrases, word2vec_path, "gab", args.output_path, "SpanBERT/spanbert-base-cased")

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
    parser.add_argument('--phrase_candidate_folder')
    parser.add_argument('--word2vec_file')
    parser.add_argument('--tweet_files', nargs='+')
    parser.add_argument('--output_path')

    args = parser.parse_args()
    main(args)