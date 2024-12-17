import argparse

import os

from src.utils import DogwhistleSplitter

def main(args):
    splitter = DogwhistleSplitter(args.dogwhistle_file_path, seen_dogwhistles=args.recall_file)

    given_dogwhistles_surface_forms, extrapolating_dogwhistles_surface_forms = splitter.split()
    
    with open(os.path.join(args.output_folder, "given.dogwhistles"), "w") as f:
        f.write("\n".join(given_dogwhistles_surface_forms))
    with open(os.path.join(args.output_folder, "extrapolating.dogwhistles"), "w") as f:
        f.write("\n".join(extrapolating_dogwhistles_surface_forms))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--recall_file', required=False, default=None)
    parser.add_argument('--output_folder')
    args = parser.parse_args()

    main(args)