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

        if self.model_name == "tomh/toxigen_hatebert":
            self.tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
        elif self.model_name in ["adediu25/subtle-toxicgenconprompt-all-no-lora", "adediu25/implicit-toxicgenconprompt-all-no-lora"]:
            self.tokenizer = AutoTokenizer.from_pretrained("youngggggg/ToxiGen-ConPrompt") 
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.labels = ["dogwhistle", "no_dogwhistle"]

        self.label2id, self.id2label = dict(), dict()
        for i, label in enumerate(self.labels):
            self.label2id[label] = i
            self.id2label[i] = label
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.labels), label2id=self.label2id, id2label=self.id2label
        )

        self.max_length = self.model.config.max_position_embeddings

    
    def tokenize(self, batch):
        return self.tokenizer(batch['text'], padding='max_length', truncation=True, max_length=self.max_length - 2, return_tensors="pt")
    
    def train(self, X: List[str], y: List[str]) -> None:

        y_modified = [self.label2id[x] for x in y]

        df = pd.DataFrame({"text": X, "label": y_modified})

        dataset = datasets.Dataset.from_pandas(df)

        tokenized_dataset = dataset.map(self.tokenize, batched=True)

        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_folder, self.model_name.replace("/", "-")),
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.lr,
            num_train_epochs=self.epochs,
            logging_dir=os.path.join(self.output_folder, self.model_name.replace("/", "-"), "logs"),
            logging_strategy="steps",
            logging_steps=200,
            do_eval=False,
            evaluation_strategy="no",
            save_strategy="epoch",
            save_total_limit=2,
        )

        # Create a Trainer instance
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        trainer.train()

        self.tokenizer.save_pretrained(os.path.join(self.output_folder, self.model_name.replace("/", "-")))

        trainer.save_model(os.path.join(self.output_folder, self.model_name.replace("/", "-")))
