from typing import List

import torch
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
)

from more_itertools import chunked

from tqdm import tqdm

class FitBert:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_name="bert-large-uncased",
        mask_token="***mask***",
        disable_gpu=False,
        batch_size = 2
    ):
        self.mask_token = mask_token
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not disable_gpu else "cpu"
        )
        print("device:", self.device)

        self.bert = model 

        self.tokenizer = tokenizer

        self.bert.eval()

    def _tokens_to_masked_ids(self, tokens, mask_ind):
        masked_tokens = tokens[:]
        masked_tokens[mask_ind] = "[MASK]"
        masked_tokens = ["[CLS]"] + masked_tokens + ["[SEP]"]

        masked_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)

        return masked_ids
    
    def rank_multi(self, masked_sent: str, options: List[str]):
        masked_sentences = [masked_sent.replace(self.mask_token, x) for x in options]

        all_probs = []

        for batch in tqdm(chunked(masked_sentences, self.batch_size)):
            print("tokenizing sentences")

            tokens_batch = [self.tokenizer.tokenize(x)[:510] for x in batch]
            
            max_length = len(max(tokens_batch, key=lambda x: len(x)))

            print("padding sentences")
            
            tokens_batch = [x+[self.tokenizer.pad_token]*(max_length - len(x)) for x in tokens_batch]

            tokens_ids_batch = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens_batch]

            tokens_batch_masked = [[self._tokens_to_masked_ids(tokens, i) for i in range(len(tokens))] for tokens in tokens_batch]

            lengths_of_tokens = [len(x) for x in tokens_batch_masked]

            tokens_batch_masked = sum(tokens_batch_masked, [])

            tens = torch.tensor(tokens_batch_masked).to(self.device)

            print("pass through")

            with torch.no_grad():

                preds = self.bert(tens)[0].detach().cpu()

            print("softmax through")

            probs = torch.nn.functional.log_softmax(preds, dim=-1)

            probs_batched = torch.split(probs, lengths_of_tokens, dim=0)

            total_probs_batched = []

            print("prob calc")

            for i, token_ids in enumerate(tokens_ids_batch):
                probs_for_tokens = probs_batched[i]

                total_prob = 0

                for j, token in enumerate(token_ids):
                    if token != 0:
                        total_prob += float(probs_for_tokens[j][j + 1][token].item())

                total_probs_batched.append(total_prob)
            
            del tens, preds, probs, probs_batched

            if self.device == "cuda":
                torch.cuda.empty_cache()

            all_probs.extend(total_probs_batched)

        ranked_pairs = sorted(list(zip(options, all_probs)), key=lambda x: x[1])

        ranked_options = [x[0] for x in ranked_pairs]

        ranked_options_prob = [x[1] for x in ranked_pairs]

        return ranked_options, ranked_options_prob