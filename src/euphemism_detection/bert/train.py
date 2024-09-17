import os

from typing import List

import datasets

import pandas as pd

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

class TrainBERT:

    def __init__(self, model_name: str, lr: float, weight_decay: float, batch_size: int, epochs: int, output_folder: str) -> None:

        self.model_name = model_name
        self.lr = lr 
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_folder = output_folder

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.labels = ["dogwhistle", "no_dogwhistle"]

        self.label2id, self.id2label = dict(), dict()
        for i, label in enumerate(self.labels):
            self.label2id[label] = str(i)
            self.id2label[str(i)] = label
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.labels), label2id=self.label2id, id2label=self.id2label
        )

        self.max_length = self.model.config.max_position_embeddings

    
    def tokenize(self, batch):
        return self.tokenizer(batch['text'], padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
    
    def train(self, X: List[str], y: List[str]) -> None:

        df = pd.DataFrame({"text": X, "labels": y})

        dataset = datasets.Dataset.from_pandas(df)

        tokenized_dataset = dataset.map(self.tokenize, batched=True, remove_columns=["text"])

        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_folder, self.model_name.replace("/", "-")),
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.lr,
            num_train_epochs=self.epochs,
            # PyTorch 2.0 specifics 
            bf16=True, # bfloat16 training 
            torch_compile=True, # optimizations
            optim="adamw_torch_fused", # improved optimizer 
            # logging & evaluation strategies
            logging_dir=os.path.join(self.output_folder, self.model_name.replace("/", "-"), "logs"),
            logging_strategy="steps",
            logging_steps=200,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
        )

        # Create a Trainer instance
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
        )

        trainer.train()

        self.tokenizer.save_pretrained(os.path.join(self.output_folder, self.model_name.replace("/", "-")))
