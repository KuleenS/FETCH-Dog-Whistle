import argparse

from collections import defaultdict

import os

import pickle

import pandas as pd

from sklearn.model_selection import train_test_split

def main(args):
    df = pd.read_csv(args.dogwhistle_file_path, sep="\t")

    dogwhistle_set = df["Surface Forms"].str.split(";").tolist()

    comparison_set = df["Dogwhistle"].tolist()

    dogwhistles = defaultdict(list)

    for i in range(len(dogwhistle_set)):
        dogwhistles[comparison_set[i]] = dogwhistle_set[i]

    data = pickle.load(open(args.recall_file, "rb"))

    dogwhistles_seen = set([x for x in data if data[x] != 0])

    extrapolating_dogwhistles, given_dogwhistles = train_test_split(dogwhistles_seen, test_size=0.2)

    given_dogwhistles_surface_forms = []
    extrapolating_dogwhistles_surface_forms = []

    for given_dogwhistle in given_dogwhistles:
        given_dogwhistles_surface_forms.extend(dogwhistles[given_dogwhistle])

    for extrapolating_dogwhistle in extrapolating_dogwhistles:
        extrapolating_dogwhistles_surface_forms.extend(dogwhistles[extrapolating_dogwhistle])
    
    with open(os.path.join(args.output_folder, "given.dogwhistles"), "w") as f:
        f.write("\n".join(given_dogwhistles_surface_forms))
    with open(os.path.join(args.output_folder, "extrapolating.dogwhistles"), "w") as f:
        f.write("\n".join(extrapolating_dogwhistles_surface_forms))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--recall_file')

    parser.add_argument('--output_folder')
    args = parser.parse_args()

    main(args)