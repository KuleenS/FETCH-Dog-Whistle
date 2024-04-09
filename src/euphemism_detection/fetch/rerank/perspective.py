import os

from typing import List, Dict, Optional

from urllib.error import HTTPError

import time

from googleapiclient import discovery
import json

from tqdm import tqdm



class ToxicityPerspective:
    def __init__(self):
        self.api = discovery.build(
          "commentanalyzer",
          "v1alpha1",
          developerKey=os.environ["PERSPECTIVE_API_KEY"],
          discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
          static_discovery=False,
        )
    
    def get_scores(self, sentences: List[str]) -> List[Optional[Dict[str, float]]]:
        toxicity = []

        for sentence in tqdm(sentences):

            try:
            
              analyze_request = {
                'comment': { 'text': sentence },
                'requestedAttributes': {'TOXICITY': {}, "SEVERE_TOXICITY" : {}, "IDENTITY_ATTACK" : {}, "INSULT" : {}, "PROFANITY" : {}, "THREAT" : {}}
              }
              response = self.api.comments().analyze(body=analyze_request).execute()
              
              scores = {}
              
              for key in response["attributeScores"]:
                  scores[key] = response["attributeScores"][key]["summaryScore"]["value"]

              toxicity.append(scores)
            
            except Exception as e:
                print(e)
            
            time.sleep(2)
                
        return toxicity
