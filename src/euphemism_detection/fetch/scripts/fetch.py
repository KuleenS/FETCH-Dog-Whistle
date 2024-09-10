import argparse

import os

import numpy as np

from tqdm import tqdm

from src.euphemism_detection.bert.predict import PredictBERT
from src.euphemism_detection.bert.train import TrainBERT

from src.euphemism_detection.fetch.db.milvusdb import MilvusDB

from src.euphemism_detection.fetch.extraction.tfidf import TFIDF
from src.euphemism_detection.fetch.extraction.keybert import KeyBERTFilter
from src.euphemism_detection.fetch.extraction.rake import RAKEFilter
from src.euphemism_detection.fetch.extraction.textrank import TextRankFilter
from src.euphemism_detection.fetch.extraction.yake import YAKEFilter

from src.euphemism_detection.llm.chatgpt import ChatGPTLLM
from src.euphemism_detection.llm.gemini import GeminiLLM
from src.euphemism_detection.llm.offline import OfflineLLM

from src.euphemism_detection.metrics import Metrics

def main(args):

    with open(os.path.join(args.dogwhistle_path, "given.dogwhistles"), "r") as f:
        given_dogwhistles_surface_forms = f.readlines()
    
    with open(os.path.join(args.dogwhistle_path, "extrapolating.dogwhistles"), "r") as f:
        extrapolating_dogwhistles_surface_forms = f.readlines()

    db = MilvusDB(args.collection_name, 384)

    db.load_data(args.embedding_folder)

    db.create_index()

    centroid = []

    total_tweet_ids = []

    total_seed_posts = []

    for seed_word in tqdm(given_dogwhistles_surface_forms):
        tweet_ids, embeddings, seed_posts = db.calculate_seed_word_centroid(seed_word)
        
        total_tweet_ids.extend(tweet_ids)
        
        centroid.extend(embeddings)

        total_seed_posts.extend(seed_posts)

    posts = []

    dogwhistles = []

    for j in tqdm(range(len(centroid))):
        if not np.isnan(centroid[j]).all():

            returned =  list(db.get_top_k_documents(total_tweet_ids, centroid[j], 1)[0])

            dogwhistles.extend([x.dogwhistle for x in returned])
            posts.extend([x.post for x in returned])
    
    if args.filtering_mode == "gemini" or args.filtering_mode == "offline" or args.filtering_mode == "chatgpt":
        prompt = ("You are a helpful assistant participating in a content moderation task of dog whistle detection.\n"
        "A dog whistle is the use of coded or suggestive language in political messaging to garner support from a particular group without provoking opposition. The concept is named after ultrasonic dog whistles, which are audible to dogs but not humans. Dog whistles use language that appears normal to the majority but communicates specific things to intended audiences. They are generally used to convey messages on issues likely to provoke controversy without attracting negative attention. Is a dog whistle present in the following sentence?\n"
        "{sentence}\n"
        "Please only reply yes or no")

        if args.filtering_mode == "chatgpt":
            model = ChatGPTLLM(args.model, args.temperature)
        elif args.filtering_mode == "offline":
            model = OfflineLLM(args.model, args.temperature)
        elif args.filtering_mode == "gemini":
            model = GeminiLLM(args.model, args.temperature)

        predictions = model.generate_from_prompts([prompt.format(sentence = x) for x in posts])

        filtered_posts = [post for prediction, post in zip(predictions, posts) if prediction in ["Yes", "yes"]]

    elif args.filtering_mode == "bert-train":

        model = TrainBERT(args.model, args.lr, args.weight_decay, args.batch_size, args.epoch, args.output_folder)

        sampled_posts = db.sample_negative_posts(total_tweet_ids, len(total_seed_posts))

        X = total_seed_posts + sampled_posts

        y = ["dogwhistle"]*len(total_seed_posts) + ["no_dogwhistle"]*len(sampled_posts)

        model.train(X,y)

        model = PredictBERT(args.output_folder)

        predictions = model.prediction(posts)

        filtered_posts = [post for prediction, post in zip(predictions, posts) if prediction in ["dogwhistle"]]

    elif args.filtering_mode == "bert-predict":
        model = PredictBERT(args.model)

        predictions = model.prediction(posts)

        filtered_posts = [post for prediction, post in zip(predictions, posts) if prediction == 1]

    elif args.filtering_mode == "none":
        filtered_posts = posts

    metrics = Metrics(args.dogwhistle_file)

    thresholds = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

    filterers = [TFIDF((1,args.extraction_n_grams)), KeyBERTFilter((1,args.extraction_n_grams)), RAKEFilter((1,args.extraction_n_grams)), YAKEFilter((1,args.extraction_n_grams)), TextRankFilter()]

    filter_names = ["tfidf", "keybert", "rake", "yake", "textrank"]

    results = []

    for k in range(len(filterers)):
        for j in range(len(thresholds)):
            if k == 1:
                top_words = list(set(filterers[k].get_most_important_ngrams(filtered_posts, thresholds[j], None, False, False)))
            else:
                top_words = list(set(filterers[k].get_most_important_ngrams(filtered_posts, thresholds[j])))
            
            precision = metrics.measure_precision(top_words, extrapolating_dogwhistles_surface_forms)

            recall = metrics.measure_recall(top_words, extrapolating_dogwhistles_surface_forms)

            possible_recall = metrics.measure_possible_recall(top_words, extrapolating_dogwhistles_surface_forms, args.extraction_n_grams)

            results.append((filter_names[k], thresholds[j], precision, recall, possible_recall))
    
    print(results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_folder")

    parser.add_argument('--dogwhistle_file')

    parser.add_argument('--dogwhistle_path')

    parser.add_argument('--collection_name')

    parser.add_argument('--filtering_method', type=str, options=["gemini", "bert-train", "bert-predict", "chatgpt", "offline", "none"], default="none")

    parser.add_argument('--model', type=str, required=False)

    parser.add_argument('--temperature', type=float, required=False)
    parser.add_argument('--lr', type=float, required=False)
    parser.add_argument('--weight_decay', type=float, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--epochs', type=int, required=False)
    parser.add_argument('--output_folder', type=str, required=False)

    parser.add_argument('--extraction_n_grams', type=int)

    parser.add_argument("--output_path")

    args = parser.parse_args()
    main(args)
