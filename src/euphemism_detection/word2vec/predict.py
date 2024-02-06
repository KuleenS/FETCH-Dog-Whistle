## Standard Library
import os

import json
import argparse

## External
import pandas as pd

from gensim.models import Word2Vec

from tracker import Tracker

def initialize_output_directory(args):
    """
    
    """
    if args.output_dir is None:
        raise ValueError("Please specify an --output_dir")
    _ = os.makedirs(args.output_dir, exist_ok = True)
    return args.output_dir

def get_ordered_vocabulary(model):
    return model.wv.index_to_key
    

def initialize_keywords(args):
    """
    
    """
    if not os.path.exists(args.keywords):
        raise FileNotFoundError(f"Could not find keyword file: '{args.keywords}'")
    keywords = []
    with open(args.keywords,"r") as the_file:
        for line in the_file:
            keywords.append(line.strip())
    keywords = list(filter(lambda l: len(l) > 0, keywords))
    return keywords

def initialize_word2vec(args):
    model = Word2Vec.load(args.model)
    return model

def find_vocabulary_terms(keywords,
                          model):
    """
    
    """
    ## Initialize Tracker
    tracker = Tracker(include_mentions=False)
    tracker = tracker.add_terms(keywords, include_hashtags=False)
    ## Find Relevant Terms
    keyword_terms = list(zip(get_ordered_vocabulary(model), list(map(tracker.search, get_ordered_vocabulary(model)))))
    keyword_terms = list(filter(lambda t: len(t[1])>0, keyword_terms))
    keyword_terms = [(x, y[0]) for x, y in keyword_terms]
    ## Re-Map
    term2keyword = {}
    for term, (keyword, keyword_matched, span) in keyword_terms:
        term2keyword[term] = keyword
    assert len(term2keyword) == len(keyword_terms)
    ## Return
    return term2keyword

def get_similar(model,
                word,
                top_k=10):
        """
        
        """
        return model.wv.similar_by_word(word, top_k)

def main(args):
    print("[Parsing Command Line]")
    print(f"[Initializing Output Directory: '{args.output_dir}']")

    output_dir = initialize_output_directory(args)

    print("[Initialzing Keyword Seed Set]")
    keywords = initialize_keywords(args)
    ## Load the Model
    print("[Loading Trained Word2Vec model]")
    model = initialize_word2vec(args)
    ## Find Relevant Vocabulary Terms
    print("[Finding Seed Vocabulary Terms]")
    term2keywords = find_vocabulary_terms(keywords=keywords,
                                          model=model)
    ## Initialize Expansion Cache
    print("[Initializing Expansion Cache]")
    expansion = {x:[(y, 0)] for x, y in term2keywords.items()}
    ## Run Expansion
    print("[Beginning Keyword Expansion]")
    queries = list(expansion.keys())
    for round in range(args.n_rounds):
        print(f">> Round {round+1}/{args.n_rounds}")
        ## Run Search
        round_expansion = {q:[i[0] for i in get_similar(model, q, top_k=args.top_k)] for q in queries}
        ## Track New Terms
        round_new = []
        ## Parse Search Result
        for query, query_result in round_expansion.items():
            for qr in query_result:
                if qr not in expansion:
                    round_new.append(qr)
                    expansion[qr] = []
                expansion[qr].append((query, round+1))
        ## Prepare Next Query Round
        queries = round_new
        ## Early Stopping
        if len(queries) == 0 and round != args.n_rounds - 1:
            print("~~~~ No new terms were found! Stopping early. ~~~~")
            break
    ## Format
    expansion = pd.Series(expansion).to_frame("source").reset_index().rename(columns={"index":"term"})
    expansion = expansion.to_dict(orient="records")
    ## Cache
    with open(f"{args.output_dir}/expansion.json","w") as the_file:
        for i, e in enumerate(expansion):
            the_file.write(json.dumps(e) + "\n" * int(i != len(expansion)-1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="Path to trained word2vec model")

    parser.add_argument("--keywords", type=str, help="Path to .keywords file containing words to expand.")

    parser.add_argument("--output_dir", type=str, default=None, help="Where to output results.")

    parser.add_argument("--n_rounds", type=int, default=1)

    parser.add_argument("--top_k", type=int, help="How many terms to expand within each round.")

    args = parser.parse_args()

    main(args)