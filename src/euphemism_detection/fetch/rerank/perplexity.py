from typing import List

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

class PerplexityMetric:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

        self.tokenizer = AutoTokenizer(self.model_name)

        self.model = AutoModelForCausalLM(self.model_name, device="cuda")

        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    def get_scores(self, sentences: List[str]):

        batched_sentences = batch(sentences, 32)

        self.model.eval()

        total_perplexity = []

        with torch.no_grad():

            for batch in batched_sentences:
                tokenized_text = self.tokenizer(sentences, padding="max_length", return_tensors='pt').to("cuda")

                output = self.model(**tokenized_text)

                logits = output.logits
                
                shift_logits = logits[:, :-1, : ].contiguous()

                shift_labels = batch["input_ids"][..., 1:].contiguous()

                shift_attention_mask_batch = batch["attention_mask"][..., 1:].contiguous()

                perplexity_batch = torch.exp(self.loss_func(shift_logits.transpose(1,2), shift_labels) * shift_attention_mask_batch).sum(1) / (shift_attention_mask_batch.sum(1))
                                            
                total_perplexity.extend(perplexity_batch.cpu().tolist())
        
        return total_perplexity
