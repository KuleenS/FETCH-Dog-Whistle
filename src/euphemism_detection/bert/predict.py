from typing import List

from transformers import pipeline

class PredictBERT:

    def __init__(self, model_folder: str, model_name: str = None):

        self.model_folder = model_folder

        self.max_length = {
            "tomh/toxigen_hatebert": 510,
            "GroNLP/hateBERT": 510,
            "adediu25/subtle-toxicgenconprompt-all-no-lora": 510,
            "adediu25/implicit-toxicgenconprompt-all-no-lora": 510,
            "facebook/roberta-hate-speech-dynabench-r4-target": 512, 
            "cardiffnlp/twitter-roberta-base-hate": 512,
            "Hate-speech-CNERG/bert-base-uncased-hatexplain" :510,
        }

        if model_folder in self.max_length:
            max_length = self.max_length[model_folder]
        
        elif model_name == "jhu-clsp/bernice" or model_name == "vinai/bertweet-base":
            max_length = 128
        
        if self.model_folder== "tomh/toxigen_hatebert":
            self.classifier = pipeline("text-classification", model=self.model_folder, tokenizer="GroNLP/hateBERT", device="cuda:0", max_length = max_length, truncation=True)
        elif self.model_folder in ["adediu25/subtle-toxicgenconprompt-all-no-lora", "adediu25/implicit-toxicgenconprompt-all-no-lora"]:
            self.classifier = pipeline("text-classification", model=self.model_folder, tokenizer="youngggggg/ToxiGen-ConPrompt", device="cuda:0", max_length = max_length, truncation=True)
        else:
            self.classifier = pipeline("text-classification", model=self.model_folder, tokenizer=self.model_folder, device="cuda:0", max_length = max_length, truncation=True)

        self.classifier.tokenizer.pad_token = self.classifier.tokenizer.eos_token
    
    def prediction(self, X: List[str]) -> List[str]:
        results = self.classifier(X)

        if self.model_folder in ["tomh/toxigen_hatebert", "GroNLP/hateBERT"]:
            results = [1 if x["label"] == "LABEL_1" else 0 for x in results]
        elif self.model_folder in ["adediu25/subtle-toxicgenconprompt-all-no-lora", "adediu25/implicit-toxicgenconprompt-all-no-lora"]:
            results = [0 if x["label"] == "Non-HS" else 1 for x in results]
        elif self.model_folder in ["facebook/roberta-hate-speech-dynabench-r4-target", "cardiffnlp/twitter-roberta-base-hate-latest"]:
            results = [1 if x["label"] == "hate" else 0 for x in results]
        elif self.model_folder in ["Hate-speech-CNERG/bert-base-uncased-hatexplain"]:
            results = [0 if x["label"] == "normal" else 1 for x in results]
        elif self.model_folder in ["badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification"]:
            results = [0 if x["label"] == "NEITHER" else 1 for x in results]
        else:
            results = [1 if x["label"] == "LABEL_1" else 0 for x in results]

        return results