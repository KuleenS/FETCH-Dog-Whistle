from typing import List

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm

class PerplexityMetric:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.model_max_length = 512

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map ="cuda")

        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    
    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def get_scores(self, sentences: List[str]):
        encodings = self.tokenizer(
            sentences,
            add_special_tokens=False,
            padding='max_length',
            truncation=True if self.tokenizer.model_max_length else False,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        self.model.eval()

        ppls = []

        for start_index in tqdm(range(0, len(encoded_texts), 4)):
            end_index = min(start_index + 4, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index].to("cuda")
            attn_mask = attn_masks[start_index:end_index].to("cuda")

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (self.loss_func(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            ).detach().cpu()

            ppls.extend(perplexity_batch.tolist())

        return ppls