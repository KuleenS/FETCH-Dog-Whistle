import json

import argparse

import itertools

RULE_DEFAULT = 0
RULE_DISCARD = 1
RULE_KEEP = 2

def keep_vocab_item(word, count, min_count, trim_rule=None):
    """Should we keep `word` in the vocab or remove it?

    Parameters
    ----------
    word : str
        Input word.
    count : int
        Number of times that word appeared in a corpus.
    min_count : int
        Discard words with frequency smaller than this.
    trim_rule : function, optional
        Custom function to decide whether to keep or discard this word.
        If a custom `trim_rule` is not specified, the default behaviour is simply `count >= min_count`.

    Returns
    -------
    bool
        True if `word` should stay, False otherwise.

    """
    default_res = count >= min_count

    if trim_rule is None:
        return default_res
    else:
        rule_res = trim_rule(word, count, min_count)
        if rule_res == RULE_KEEP:
            return True
        elif rule_res == RULE_DISCARD:
            return False
        else:
            return default_res

def prune_vocab(vocab, min_reduce, trim_rule=None):
    """Remove all entries from the `vocab` dictionary with count smaller than `min_reduce`.

    Modifies `vocab` in place, returns the sum of all counts that were pruned.

    Parameters
    ----------
    vocab : dict
        Input dictionary.
    min_reduce : int
        Frequency threshold for tokens in `vocab`.
    trim_rule : function, optional
        Function for trimming entities from vocab, default behaviour is `vocab[w] <= min_reduce`.

    Returns
    -------
    result : int
        Sum of all counts that were pruned.

    """
    result = 0
    old_len = len(vocab)
    for w in list(vocab):  # make a copy of dict's keys
        if not keep_vocab_item(w, vocab[w], min_reduce, trim_rule):  # vocab[w] <= min_reduce:
            result += vocab[w]
            del vocab[w]
    print(
        "pruned out %i tokens with count <=%i (before %i, after %i)",
        old_len - len(vocab), min_reduce, old_len, len(vocab)
    )
    return result

def main(args):

    vocab = {}

    max_vocab_size = 40000000

    sentence_no, total_words, min_reduce = -1, 0, 1

    delimiter = '_'

    connector_words = frozenset(
        " a an the "  # articles; we never care about these in MWEs
        " for of with without at from to in on by "  # prepositions; incomplete on purpose, to minimize FNs
        " and or "  # conjunctions; incomplete on purpose, to minimize FNs
        .split()    
    )

    with open(args.input_file, 'r') as file:

        for line in file:

            sentence = line.strip().split()

            if sentence_no % 10_000 == 0:
                print(
                    "PROGRESS: at sentence #%i, processed %i words and %i word types",
                    sentence_no, total_words, len(vocab),
                )
            start_token, in_between = None, []
            for word in sentence:
                if word not in connector_words:
                    vocab[word] = vocab.get(word, 0) + 1
                    if start_token is not None:
                        phrase_tokens = itertools.chain([start_token], in_between, [word])
                        joined_phrase_token = delimiter.join(phrase_tokens)
                        vocab[joined_phrase_token] = vocab.get(joined_phrase_token, 0) + 1
                    start_token, in_between = word, []  # treat word as both end of a phrase AND beginning of another
                elif start_token is not None:
                    in_between.append(word)
                total_words += 1

            if len(vocab) > max_vocab_size:
                prune_vocab(vocab, min_reduce)
                min_reduce += 1

        print(
            "collected %i token types (unigram + bigrams) from a corpus of %i words and %i sentences",
            len(vocab), total_words, sentence_no + 1,
        )

    vocab["min_reduce_number"] = min_reduce
    
    with open(args.output_file, 'w') as file:
        file.write(json.dumps(vocab))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')

    args = parser.parse_args()
    main(args)