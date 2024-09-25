import json

import argparse

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


def merge_counts(dict1, dict2):
    """Merge `dict1` of (word, freq1) and `dict2` of (word, freq2) into `dict1` of (word, freq1+freq2).
    Parameters
    ----------
    dict1 : dict of (str, int)
        First dictionary.
    dict2 : dict of (str, int)
        Second dictionary.
    Returns
    -------
    result : dict
        Merged dictionary with sum of frequencies as values.
    """
    for word, freq in dict2.items():
        if word in dict1:
            dict1[word] += freq
        else:
            dict1[word] = freq

    return dict1


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
        if not keep_vocab_item(
            w, vocab[w], min_reduce, trim_rule
        ):  # vocab[w] <= min_reduce:
            result += vocab[w]
            del vocab[w]
    print(
        "pruned out %i tokens with count <=%i (before %i, after %i)",
        old_len - len(vocab),
        min_reduce,
        old_len,
        len(vocab),
    )
    return result


def main(args):

    total_counts = {}

    min_reduces = []

    for input_file in args.input_files:

        with open(input_file) as f:
            data = json.load(f)

        min_reduces.append(data["min_reduce_number"])

        del data["min_reduce_number"]

        total_counts = merge_counts(total_counts, data)

    prune_vocab(total_counts, max(min_reduces))

    with open(args.output_file, "w") as file:
        file.write(json.dumps(total_counts))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs="+")
    parser.add_argument("--output_file")

    args = parser.parse_args()
    main(args)
