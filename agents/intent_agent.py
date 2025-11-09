"""
Handles intent classification for user queries using a trained ML model.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class IntentAgent:
    def __init__(self, model_path=None):
        # Load trained model if available, otherwise initialize a new one
        if model_path:
            self.vec, self.clf = joblib.load(model_path)
        else:
            self.vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
            self.clf = LogisticRegression(max_iter=1000)

    def train(self, texts, labels, save_path):
        """
        Train a simple text classification model and save it to disk.
        """
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, labels)
        joblib.dump((self.vec, self.clf), save_path)

    def predict(self, query):
        """
        Predict the intent of a given user query.
        """
        X = self.vec.transform([query])
        return self.clf.predict(X)[0]
