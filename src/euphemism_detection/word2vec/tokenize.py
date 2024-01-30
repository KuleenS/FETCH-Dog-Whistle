import argparse

import os

import json

import csv

def original_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    r"""Bigram scoring function, based on the original `Mikolov, et. al: "Distributed Representations
    of Words and Phrases and their Compositionality" <https://arxiv.org/abs/1310.4546>`_.

    Parameters
    ----------
    worda_count : int
        Number of occurrences for first word.
    wordb_count : int
        Number of occurrences for second word.
    bigram_count : int
        Number of co-occurrences for phrase "worda_wordb".
    len_vocab : int
        Size of vocabulary.
    min_count: int
        Minimum collocation count threshold.
    corpus_word_count : int
        Not used in this particular scoring technique.

    Returns
    -------
    float
        Score for given phrase. Can be negative.

    Notes
    -----
    Formula: :math:`\frac{(bigram\_count - min\_count) * len\_vocab }{ (worda\_count * wordb\_count)}`.

    """
    denom = worda_count * wordb_count
    if denom == 0:
        return float('-inf')
    return (bigram_count - min_count) / float(denom) * len_vocab

def score_candidate(vocab, word_a, word_b, in_between, delimiter = "_"):
    # Micro optimization: check for quick early-out conditions, before the actual scoring.

    min_count=5
    
    threshold=10.0

    corpus_word_count = len(vocab)

    word_a_cnt = vocab.get(word_a, 0)
    if word_a_cnt <= 0:
        return None, None

    word_b_cnt = vocab.get(word_b, 0)
    if word_b_cnt <= 0:
        return None, None

    phrase = delimiter.join([word_a] + in_between + [word_b])
    # XXX: Why do we care about *all* phrase tokens? Why not just score the start+end bigram?
    phrase_cnt = vocab.get(phrase, 0)
    if phrase_cnt <= 0:
        return None, None

    score = original_scorer(
        worda_count=word_a_cnt, wordb_count=word_b_cnt, bigram_count=phrase_cnt,
        len_vocab=len(vocab), min_count=min_count, corpus_word_count=corpus_word_count,
    )
    if score <= threshold:
        return None, None

    return phrase, score

def analyze_sentence(sentence, connector_words, vocab):
    """Analyze a sentence, concatenating any detected phrases into a single token.

    Parameters
    ----------
    sentence : iterable of str
        Token sequence representing the sentence to be analyzed.

    Yields
    ------
    (str, {float, None})
        Iterate through the input sentence tokens and yield 2-tuples of:
        - ``(concatenated_phrase_tokens, score)`` for token sequences that form a phrase.
        - ``(word, None)`` if the token is not a part of a phrase.

    """
    start_token, in_between = None, []
    for word in sentence:
        if word not in connector_words:
            # The current word is a normal token, not a connector word, which means it's a potential
            # beginning (or end) of a phrase.
            if start_token:
                # We're inside a potential phrase, of which this word is the end.
                phrase, score = score_candidate(vocab, start_token, word, in_between)
                if score is not None:
                    # Phrase detected!
                    yield phrase, score
                    start_token, in_between = None, []
                else:
                    # Not a phrase after all. Dissolve the candidate's constituent tokens as individual words.
                    yield start_token, None
                    for w in in_between:
                        yield w, None
                    start_token, in_between = word, []  # new potential phrase starts here
            else:
                # Not inside a phrase yet; start a new phrase candidate here.
                start_token, in_between = word, []
        else:  # We're a connector word.
            if start_token:
                # We're inside a potential phrase: add the connector word and keep growing the phrase.
                in_between.append(word)
            else:
                # Not inside a phrase: emit the connector word and move on.
                yield word, None
    # Emit any non-phrase tokens at the end.
    if start_token:
        yield start_token, None
        for w in in_between:
            yield w, None

def main(args):

    with open(input_file) as f:
        vocab = json.load(f)

    connector_words = frozenset(
        " a an the "  # articles; we never care about these in MWEs
        " for of with without at from to in on by "  # prepositions; incomplete on purpose, to minimize FNs
        " and or "  # conjunctions; incomplete on purpose, to minimize FNs
        .split()    
    )
    
    for input_file in args.input_files:

        out_file = os.path.basename(input_file)

        results = []
        
        with open(os.path.join(args.output_folder, out_file), "w") as out:
            writer_csv = csv.writer(out)

            with open(input_file) as in_file:
                for line in in_file:
                    sentence = line.strip().split()

                    results.append(" ".join([token for token, _ in analyze_sentence(sentence, connector_words, vocab)]))

                    if len(results) > 500:
                        writer_csv.writerows(results)
                        results = []
                    
                    writer_csv.writerows(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', nargs="+")
    parser.add_argument('--output_folder')
    parser.add_argument('--vocab')

    args = parser.parse_args()
    main(args)