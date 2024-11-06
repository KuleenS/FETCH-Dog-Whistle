import argparse

import os

import pickle

import json

import ast

import nltk

import numpy as np

from tqdm import tqdm

from src.euphemism_detection.bert.predict import PredictBERT
from src.euphemism_detection.bert.train import TrainBERT

from src.euphemism_detection.fetch.db.chromadb import ChromaDB

from src.euphemism_detection.fetch.extraction.tfidf import TFIDF
from src.euphemism_detection.fetch.extraction.keybert import KeyBERTFilter
from src.euphemism_detection.fetch.extraction.rake import RAKEFilter
from src.euphemism_detection.fetch.extraction.textrank import TextRankFilter
from src.euphemism_detection.fetch.extraction.yake import YAKEFilter

from src.euphemism_detection.llm.chatgpt import ChatGPTLLM
from src.euphemism_detection.llm.gemini import GeminiLLM
from src.euphemism_detection.llm.offline import OfflineLLM

from src.euphemism_detection.metrics import Metrics

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def measure_max_effectiveness(
    args, dogwhistles_at_step, extrapolating_dogwhistles_surface_forms
):
    metrics = Metrics(args.dogwhistle_file)

    precision = metrics.measure_precision(
        dogwhistles_at_step, extrapolating_dogwhistles_surface_forms
    )

    recall = metrics.measure_recall(
        dogwhistles_at_step, extrapolating_dogwhistles_surface_forms
    )

    possible_recall_1 = metrics.measure_possible_recall(
        dogwhistles_at_step, extrapolating_dogwhistles_surface_forms, 1
    )
    possible_recall_2 = metrics.measure_possible_recall(
        dogwhistles_at_step, extrapolating_dogwhistles_surface_forms, 2
    )
    possible_recall_3 = metrics.measure_possible_recall(
        dogwhistles_at_step, extrapolating_dogwhistles_surface_forms, 3
    )

    print(
        f"Best Dogwhistle We Can Do: {precision}, {recall}, {possible_recall_1}, {possible_recall_2}, {possible_recall_3}"
    )


def run_extraction(args, filtered_posts, extrapolating_dogwhistles_surface_forms):
    metrics = Metrics(args.dogwhistle_file)

    thresholds = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

    results = []

    for ngram in args.extraction_n_grams:

        filterers = [
            TFIDF((1, ngram)),
            KeyBERTFilter((1, ngram)),
            RAKEFilter((1, ngram)),
            YAKEFilter((1, ngram)),
            TextRankFilter(),
        ]

        filter_names = ["tfidf", "keybert", "rake", "yake", "textrank"]

        for k in range(len(filterers)):
            for j in range(len(thresholds)):
                if k == 1:
                    top_words = list(
                        set(
                            filterers[k].get_most_important_ngrams(
                                filtered_posts, thresholds[j], None, False, False
                            )
                        )
                    )
                else:
                    top_words = list(
                        set(
                            filterers[k].get_most_important_ngrams(
                                filtered_posts, thresholds[j]
                            )
                        )
                    )

                precision = metrics.measure_precision(
                    top_words, extrapolating_dogwhistles_surface_forms
                )

                recall = metrics.measure_recall(
                    top_words, extrapolating_dogwhistles_surface_forms
                )

                possible_recall = metrics.measure_possible_recall(
                    top_words, extrapolating_dogwhistles_surface_forms, ngram
                )

                results.append(
                    (
                        ngram,
                        filter_names[k],
                        thresholds[j],
                        precision,
                        recall,
                        possible_recall,
                    )
                )

    print(results)


def main(args):

    with open(os.path.join(args.dogwhistle_path, "given.dogwhistles"), "r") as f:
        given_dogwhistles_surface_forms = [x.lower().strip() for x in f.readlines()]

    with open(
        os.path.join(args.dogwhistle_path, "extrapolating.dogwhistles"), "r"
    ) as f:
        extrapolating_dogwhistles_surface_forms = [
            x.lower().strip() for x in f.readlines()
        ]

    db = ChromaDB(args.db_path, args.collection_name, args.embedding_folder)

    if not os.path.exists(
        os.path.join(args.output_folder, f"lookup_saved_{args.collection_name}.pickle")
    ):
        centroid = []

        total_tweet_ids = []

        total_seed_posts = []

        print(given_dogwhistles_surface_forms[:10])

        for seed_word in tqdm(given_dogwhistles_surface_forms):
            tweet_ids, embeddings, seed_posts = db.calculate_seed_word_centroid(
                seed_word
            )

            total_tweet_ids.extend(tweet_ids)

            centroid.extend(embeddings)

            total_seed_posts.extend(seed_posts)

        returned = db.get_top_k_documents(total_tweet_ids, centroid, 1)

        posts = [x["post"] for x in returned]

        dogwhistles = [x["dogwhistle"] for x in returned]
        
        with open(
            os.path.join(
                args.output_folder, f"lookup_saved_{args.collection_name}.pickle"
            ),
            "wb",
        ) as f:

            lookup_output = (
                total_tweet_ids,
                total_seed_posts,
                centroid,
                posts,
                dogwhistles,
            )

            pickle.dump(lookup_output, f)
    else:
        with open(
            os.path.join(
                args.output_folder, f"lookup_saved_{args.collection_name}.pickle"
            ),
            "rb",
        ) as f:
            total_tweet_ids, total_seed_posts, centroid, posts, dogwhistles = (
                pickle.load(f)
            )

    if (
        args.filtering_method == "gemini-twostep"
        or args.filtering_method == "offline-twostep"
        or args.filtering_method == "chatgpt-twostep"
    ):
        prompt = (
            "You are a helpful assistant participating in a content moderation task of dog whistle detection.\n"
            "A dog whistle is the use of coded or suggestive language in political messaging to garner support from a particular group without provoking opposition. The concept is named after ultrasonic dog whistles, which are audible to dogs but not humans. Dog whistles use language that appears normal to the majority but communicates specific things to intended audiences. They are generally used to convey messages on issues likely to provoke controversy without attracting negative attention. Are dog whistles present in the following sentences?\n"
            "{sentence}\n"
            "Please respond only Yes or No in for each post.\n"
            'Please respond with a JSON for a function call with its proper arguments that best answers the given prompt. Respond in the format {{"predictions": list of Yes or No}}.\n'
        )

        for model_name in args.models:

            if args.filtering_method == "chatgpt-twostep":
                model = ChatGPTLLM(model_name,max_tokens=3*args.llm_batch_size)
            elif args.filtering_method == "offline-twostep":
                model = OfflineLLM(model_name, max_tokens=3*args.llm_batch_size)
            elif args.filtering_method == "gemini-twostep":
                model = GeminiLLM(model_name, max_tokens=3*args.llm_batch_size)

            batched_posts = list(batch(posts, args.llm_batch_size))

            batched_dogwhistles = list(batch(dogwhistles, args.llm_batch_size))

            prompts = []

            for batched_post in batched_posts:
                prompts.append(prompt.format(sentence="\n".join(batched_post)))

            predictions = model.generate_from_prompts(prompts, "predictions")

            predictions = [x["predictions"] for x in predictions]

            with open(os.path.join(args.output_folder, f"{model_name.replace('/', '-')}_predictions-twostep.txt"), "w") as f:
                f.write("\n".join([",".join(x) for x in predictions]))

            cut_down_dogwhistles = sum([batched_dogwhistle[:len(prediction)] for batched_dogwhistle,prediction in zip(batched_dogwhistles, predictions)], [])

            cut_down_posts = sum([batched_post[:len(prediction)] for batched_post,prediction in zip(batched_posts, predictions)], [])

            predictions = sum(predictions, [])

            filtered_posts = [
                post
                for prediction, post in zip(predictions, cut_down_posts)
                if any(x in prediction for x in ["yes", "y"])
            ]

            dogwhistles_at_step = [
                dogwhistle
                for dogwhistle, prediction in zip(cut_down_dogwhistles, predictions)
                if any(x in prediction for x in ["yes", "y"])
            ]

            if len(filtered_posts) != 0:
                measure_max_effectiveness(
                    args, dogwhistles_at_step, extrapolating_dogwhistles_surface_forms
                )
                run_extraction(
                    args, filtered_posts, extrapolating_dogwhistles_surface_forms
                )

            else:
                print("FAILURE NO POSTS")
    elif (
        args.filtering_method == "gemini-direct"
        or args.filtering_method == "offline-direct"
        or args.filtering_method == "chatgpt-direct"
    ):
        prompt = (
            "You are a helpful assistant participating in a content moderation task of dog whistle detection.\n"
            "A dog whistle is the use of coded or suggestive language in political messaging to garner support from a particular group without provoking opposition. The concept is named after ultrasonic dog whistles, which are audible to dogs but not humans. Dog whistles use language that appears normal to the majority but communicates specific things to intended audiences. They are generally used to convey messages on issues likely to provoke controversy without attracting negative attention. Do the following posts contain dog whistles? If they do, please identify the dog whistles in each one.\n"
            "{sentence}\n"
            'Please respond with a JSON for a function call with its proper arguments that best answers the given prompt. Respond in the format {{"dogwhistles": list of strings}}.\n'
        )

        for model_name in args.models:

            if args.filtering_method == "chatgpt-direct":
                model = ChatGPTLLM(model_name, args.temperature, max_tokens = 600)
            elif args.filtering_method == "offline-direct":
                model = OfflineLLM(model_name, args.temperature, max_tokens = 600)
            elif args.filtering_method == "gemini-direct":
                model = GeminiLLM(model_name, args.temperature, max_tokens = 600)

            print("Filtered with:", model_name)

            batched_posts = batch(posts, args.llm_batch_size)

            prompts = []

            for batched_post in batched_posts:
                prompts.append(prompt.format(sentence="\n".join(batched_post)))
            
            predictions = model.generate_from_prompts(prompts, "dogwhistle")

            predictions = [x["dogwhistles"] for x in predictions]

            with open(os.path.join(args.output_folder, f"{model_name.replace('/', '-')}_predictions-direct.txt"), "w") as f:
                f.write("\n".join([",".join(x) for x in predictions]))

            dogwhistles_found = sum(predictions, [])
                
            ngrams = max([len(nltk.tokenize.word_tokenize(x)) for x in dogwhistles_found if isinstance(x, str)])
            
            metrics = Metrics(args.dogwhistle_file)

            precision = metrics.measure_precision(
                dogwhistles_found, extrapolating_dogwhistles_surface_forms
            )

            recall = metrics.measure_recall(
                dogwhistles_found, extrapolating_dogwhistles_surface_forms
            )

            possible_recall = metrics.measure_possible_recall(
                dogwhistles_found, extrapolating_dogwhistles_surface_forms, ngrams
            )

            print("Filtered with a trained version of:", model_name)

            print(precision, recall, possible_recall)

    elif args.filtering_method == "bert-train":

        sampled_posts = db.sample_negative_posts(total_tweet_ids, len(total_seed_posts))

        X = total_seed_posts + sampled_posts

        y = ["dogwhistle"] * len(total_seed_posts) + ["no_dogwhistle"] * len(
            sampled_posts
        )

        for model_name in args.models:
            model = TrainBERT(
                model_name,
                args.lr,
                args.weight_decay,
                args.batch_size,
                args.epochs,
                args.output_folder,
            )

            model.train(X, y)

            model = PredictBERT(
                os.path.join(args.output_folder, model_name.replace("/", "-")),
                model_name,
            )

            predictions = model.prediction(posts)

            filtered_posts = [
                post for prediction, post in zip(predictions, posts) if prediction == 1
            ]

            print("Filtered with a trained version of :", model_name)

            dogwhistles_at_step = [
                dogwhistle
                for dogwhistle, prediction in zip(dogwhistles, predictions)
                if prediction == 1
            ]

            if len(filtered_posts) != 0:
                measure_max_effectiveness(
                    args, dogwhistles_at_step, extrapolating_dogwhistles_surface_forms
                )
                run_extraction(
                    args, filtered_posts, extrapolating_dogwhistles_surface_forms
                )

            else:
                print("FAILURE NO POSTS")

    elif args.filtering_method == "bert-predict":
        for model_name in args.models:
            model = PredictBERT(model_folder=model_name, model_name=model_name)

            predictions = model.prediction(posts)

            print(predictions[:10])

            filtered_posts = [
                post for prediction, post in zip(predictions, posts) if prediction == 1
            ]

            print("Filtered with:", model_name)

            dogwhistles_at_step = [
                dogwhistle for dogwhistle, prediction in zip(dogwhistles, predictions) if prediction == 1
            ]

            if len(filtered_posts) != 0:
                measure_max_effectiveness(
                    args, dogwhistles_at_step, extrapolating_dogwhistles_surface_forms
                )
                run_extraction(
                    args, filtered_posts, extrapolating_dogwhistles_surface_forms
                )

            else:
                print("FAILURE NO POSTS")

    elif args.filtering_method == "none":
        filtered_posts = posts
        measure_max_effectiveness(
            args, dogwhistles, extrapolating_dogwhistles_surface_forms
        )
        run_extraction(args, filtered_posts, extrapolating_dogwhistles_surface_forms)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_folder")

    parser.add_argument("--dogwhistle_file")

    parser.add_argument("--dogwhistle_path")

    parser.add_argument("--collection_name")

    parser.add_argument("--output_folder")

    parser.add_argument("--db_path")

    parser.add_argument(
        "--filtering_method",
        type=str,
        choices=[
            "bert-train",
            "bert-predict",
            "gemini-twostep",
            "chatgpt-twostep",
            "offline-twostep",
            "offline-direct",
            "chatgpt-direct",
            "gemini-direct",
            "none",
        ],
        default="none",
    )

    parser.add_argument("--llm_batch_size", type=int, required=False, default=1)

    parser.add_argument("--models", type=str, nargs="+", required=False)

    parser.add_argument("--temperature", type=float, required=False)
    parser.add_argument("--lr", type=float, required=False)
    parser.add_argument("--weight_decay", type=float, required=False)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--epochs", type=int, required=False)
    parser.add_argument("--model_output_folder", type=str)

    parser.add_argument("--extraction_n_grams", type=int, nargs="+")

    args = parser.parse_args()
    main(args)
