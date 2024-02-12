import argparse

import csv

import os

from src.euphemism_detection.fetch.db.milvusdb import MilvusDB


def main(args):

    collection_name = os.path.basename(args.embeddings_store)

    db = MilvusDB(collection_name, 768)

    db.load_data(args.embeddings_store)

    db.create_index()

    centroids = {}

    doc_ids = {}

    with open(args.input_dogwhistles, "r") as f:

        seed_words = [x.strip() for x in f.readlines()]

    for seed_word in seed_words:

        document_ids, centroid = db.calculate_seed_word_centroid(seed_word)

        centroids[seed_word] = centroid 

        doc_ids[seed_word] = document_ids
    

    for seed_word in seed_words:
        with open(os.path.join(args.output_path, f"{seed_word}_documents_top_k_{args.top_k_documents}.csv"), "w") as f:
            csvwriter = csv.writer(f)

            csvwriter.writerow(["documents"])

            documents = db.get_top_k_documents(doc_ids[seed_word], centroids[seed_word], args.top_k_documents)
            
            csvwriter.writerows(documents)
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--embeddings_store")

    parser.add_argument("--top_k_documents", type=int)

    parser.add_argument("--output_path")

    parser.add_argument("--input_dogwhistles")

    args = parser.parse_args()
    main(args)
