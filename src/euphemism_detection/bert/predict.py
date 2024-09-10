from typing import List

from transformers import pipeline

class PredictBERT:

    def __init__(self, model_folder: str):

        self.model_folder = model_folder

        self.classifier = pipeline( model=self.model_folder, tokenizer=self.model_folder, device=0)
    
    def prediction(self, X: List[str]) -> List[str]:
        results = []

        for item in X:
            results.append(self.classifier(item)["label"])
        
        return results