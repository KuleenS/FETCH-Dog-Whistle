from typing import List

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDF:

    def __init__(self, ngram_range: tuple[int, int]):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)

    def get_most_important_ngrams(self, corpus: List[str], top_k: int):
        X = self.tfidf_vectorizer.fit_transform(corpus)

        importance = np.argsort(np.asarray(X.sum(axis=0)).ravel())[::-1]
        
        tfidf_feature_names = np.array( self.tfidf_vectorizer.get_feature_names())

        return tfidf_feature_names[importance[:top_k]]