import argparse

import os

import pickle

import json

import re

import nltk

from tqdm import tqdm

from src.earshot.bert.predict import PredictBERT
from src.earshot.bert.train import TrainBERT

from src.earshot.db.chromadb import ChromaDB

from src.earshot.extraction.tfidf import TFIDF
from src.earshot.extraction.keybert import KeyBERTFilter
from src.earshot.extraction.rake import RAKEFilter
from src.earshot.extraction.textrank import TextRankFilter
from src.earshot.extraction.yake import YAKEFilter

from src.earshot.llm.chatgpt import ChatGPTLLM
from src.earshot.llm.gemini import GeminiLLM
from src.earshot.llm.offline import OfflineLLM

from src.metrics import Metrics

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
        two_step_prompt = (
            "You are a helpful assistant participating in a content moderation task of dog whistle detection.\n"
            "A dog whistle is the use of coded or suggestive language in political messaging to garner support from a particular group without provoking opposition. The concept is named after ultrasonic dog whistles, which are audible to dogs but not humans. Dog whistles use language that appears normal to the majority but communicates specific things to intended audiences. They are generally used to convey messages on issues likely to provoke controversy without attracting negative attention. Are dog whistles present in the following sentence?\n"
            "{sentence}\n"
            "Please respond ONLY Yes or No. Do not explain your answer.\n"
        )

        for model_name in args.models:

            if args.filtering_method == "chatgpt-twostep":
                model = ChatGPTLLM(model_name,max_tokens=5)
            elif args.filtering_method == "offline-twostep":
                model = OfflineLLM(model_name, args.llm_batch_size, max_tokens=5)
            elif args.filtering_method == "gemini-twostep":
                model = GeminiLLM(model_name, max_tokens=5)

            predictions = model.generate_from_prompts([two_step_prompt.format(sentence=x) for x in posts])

            with open(os.path.join(args.output_folder, f"{model_name.replace('/', '-')}_predictions-twostep.txt"), "w") as f:
                f.write("\n".join([",".join(x) for x in predictions]))

            filtered_posts = [
                post
                for prediction, post in zip(predictions, posts)
                if any(x in prediction for x in ["yes", "y"])
            ]

            if len(filtered_posts) != 0:
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
        direct_prompt = (
            "You are a helpful assistant participating in a content moderation task of dog whistle detection.\n"
            "A dog whistle is the use of coded or suggestive language in political messaging to garner support from a particular group without provoking opposition. The concept is named after ultrasonic dog whistles, which are audible to dogs but not humans. Dog whistles use language that appears normal to the majority but communicates specific things to intended audiences. They are generally used to convey messages on issues likely to provoke controversy without attracting negative attention. Does the following post contain dog whistles? If they do, please identify the dog whistles.\n"
            "{sentence}\n"
            'Please respond with a JSON for a function call with its proper arguments that best answers the given prompt. Respond in the format {{"dogwhistles": list of strings}}.\n'
        )

        for model_name in args.models:

            if args.filtering_method == "chatgpt-direct":
                model = ChatGPTLLM(model_name, max_tokens = 300)
            elif args.filtering_method == "offline-direct":
                model = OfflineLLM(model_name, args.llm_batch_size, max_tokens=300)
            elif args.filtering_method == "gemini-direct":
                model = GeminiLLM(model_name, max_tokens = 300)

            print("Filtered with:", model_name)

            predictions = model.generate_from_prompts([direct_prompt.format(sentence=x) for x in posts])

            with open(os.path.join(args.output_folder, f"{model_name.replace('/', '-')}_predictions-direct.txt"), "w") as f:
                f.write("\n".join([",".join(x) for x in predictions]))

            dogwhistles_found = []

            valid_json_objects = []

            for prediction in predictions:
                json_pattern = r'\{.*?\}'

                json_strings = re.findall(json_pattern, prediction)
                
                for js in json_strings:
                    try:
                        valid_json_objects.append(json.loads(js)) 
                    except json.JSONDecodeError:
                        pass  
            
            for valid_json_object in valid_json_objects:
                if "dogwhistle" in valid_json_object:
                    dogwhistles_found.extend(valid_json_object["dogwhistle"])
                else:
                    for item in valid_json_object:
                        if isinstance(valid_json_object[item], list):
                            dogwhistles_found.extend([str(x) for x in valid_json_object[item]])

            print("Filtered with a trained version of:", args.input_files[0].split("_")[0])
        
            if len(dogwhistles_found) > 0:

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

                print(precision, recall, possible_recall)
            
            else: 
                print("NO DOGWHISTLES FOUND")

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

            if len(filtered_posts) != 0:
                run_extraction(
                    args, filtered_posts, extrapolating_dogwhistles_surface_forms
                )

            else:
                print("FAILURE NO POSTS")

    elif args.filtering_method == "bert-predict":
        for model_name in args.models:
            model = PredictBERT(model_folder=model_name, model_name=model_name)

            predictions = model.prediction(posts)

            filtered_posts = [
                post for prediction, post in zip(predictions, posts) if prediction == 1
            ]

            print("Filtered with:", model_name)

            if len(filtered_posts) != 0:
                run_extraction(
                    args, filtered_posts, extrapolating_dogwhistles_surface_forms
                )

            else:
                print("FAILURE NO POSTS")

    elif args.filtering_method == "none":

        filtered_posts = posts

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
