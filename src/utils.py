import re
import torch
import json
import numpy as np
from typing import List
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from models import (
    DummyModel,
    DummyModelConfig,
    TfidfClassifier,
    TfidfClassifierConfig,
    SentenceEmbeddingModel, 
    SentenceEmbeddingConfig,
    Tfidf,
)
from constants import DUMMY_MODEL_CONFIG_PATH, ALL_STOPWORDS

def remove_strings(text: str, strings: List[str]) -> str:
    for s in strings:
        s = str(s)
        if s in text:
            text = text.replace(s, "")

    return text

def remove_numbers(text: str, remove_string: bool = False) -> str:
    text = text.split()
    text = [t for t in text if not re.search(r"\d", t)] if remove_string else [re.sub(r"\d+", "", t) for t in text]

    return " ".join(text)

def remove_stopwords(text: str):
    text = text.split()
    text = [t for t in text if t not in ALL_STOPWORDS or t == "can"]

    return " ".join(text)

def remove_punctuations(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)

    return " ".join(text.strip().split()) 

def clean_text(row) -> str:
    text = row.Item_Name
    brand = row.Brand
    unit = str(row.Unit)
    text = remove_strings(text, [brand])
    text = remove_punctuations(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    if unit not in text and text == "":
        text += unit

    return  text

def load_tfidf_model():
    config = TfidfClassifierConfig()
    model = TfidfClassifier(config)

    return model

def load_tfidf():
    config = TfidfClassifierConfig()
    model = Tfidf(config)

    return model

def load_dummy_model():
    with open(DUMMY_MODEL_CONFIG_PATH, "r") as f:
        config_dict = json.load(f)

    try:
        config = DummyModelConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys in {DUMMY_MODEL_CONFIG_PATH}: {e}.")

    model = DummyModel(config)

    return model

def load_embedding_model(config_path: str):
    with open(config_path, "r") as f:

        config_dict = json.load(f)
    try:
        config = SentenceEmbeddingConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")

    model = SentenceEmbeddingModel(config)

    return model

def load_bagging_ensemble_model(config_path: str):
    class BaggingEnsemble:
        def __init__(self, config_path: str):
            self.config_path = config_path
            self.tfidf_model = None
            self.embedding_model = None
            
        def fit(self, X, y):
            self.tfidf_model = load_tfidf()
            self.embedding_model = load_embedding_model(self.config_path)
            self.classes_ = list(set(y))
            
            self.tfidf_model.fit(X, self.classes_)
            return self
            
        def predict(self, X):
            n_samples = len(X)
            n_estimators = 5
            predictions = []
            
            for _ in range(n_estimators):
                indices = np.random.choice(n_samples, size=int(0.4 * n_samples), replace=False)
                X_subset = [X[i] for i in indices]
                
                tfidf_pred = self.tfidf_model.predict(X_subset)
                
                scores = self.embedding_model.get_scores(X_subset, self.classes_)
                embedding_pred = [self.classes_[torch.argmax(scores[i]).item()] for i in range(len(X_subset))]
                
                predictions.append((tfidf_pred, embedding_pred, indices))
            
            final_predictions = []
            for i in range(n_samples):
                votes = []
                for tfidf_pred, embedding_pred, indices in predictions:
                    if i in indices:
                        idx_in_subset = list(indices).index(i)
                        votes.extend([tfidf_pred[idx_in_subset], embedding_pred[idx_in_subset]])
                
                if votes:
                    final_predictions.append(max(set(votes), key=votes.count))
                else:
                    # tfidf_pred = self.tfidf_model.predict([X[i]])[0]
                    # final_predictions.append(tfidf_pred)
                    scores = self.embedding_model.get_scores([X[i]], self.classes_)
                    embedding_pred = self.classes_[torch.argmax(scores[0]).item()]
                    final_predictions.append(embedding_pred)
                    
            return final_predictions
    
    return BaggingEnsemble(config_path)

def load_boosting_ensemble_model(config_path: str):
    class BoostingEnsemble:
        def __init__(self, config_path: str):
            self.config_path = config_path
            self.tfidf_model = None
            self.embedding_model = None
            
        def fit(self, X, y):
            self.tfidf_model = load_tfidf()
            self.embedding_model = load_embedding_model(self.config_path)
            self.classes_ = list(set(y))
            
            self.tfidf_model.fit(X, self.classes_)
            return self
            
        def predict(self, X, y, k):
            tfidf_predictions = self.tfidf_model.predict(X)
            
            misclassified_indices = []
            final_predictions = tfidf_predictions.copy()
            
            for i in range(len(X)):
                if X[i] != y[i]:
                    misclassified_indices.append(i)
            
            if misclassified_indices:
                misclassified_texts = [X[i] for i in misclassified_indices]
                embedding_scores = self.embedding_model.get_scores(misclassified_texts, self.classes_)
                
                for idx, scores_tensor in enumerate(embedding_scores):
                    embedding_scores_np = scores_tensor.cpu().numpy()
                    if isinstance(embedding_scores_np, np.ndarray) and embedding_scores_np.ndim == 0:
                        embedding_scores_np = embedding_scores_np.reshape(-1)
                    elif len(embedding_scores_np.shape) > 1:
                        embedding_scores_np = embedding_scores_np.flatten()
                    

                    topk_indices = np.argsort(embedding_scores_np)[-k:][::-1]
                    topk_classes = [self.classes_[i] for i in topk_indices]
                    
                    pred = final_predictions[misclassified_indices[idx]]
                    
                    if pred in topk_classes:
                        final_predictions[misclassified_indices[idx]] = pred
                    else:
                        final_predictions[misclassified_indices[idx]] = topk_classes[0]
                        
            return final_predictions
    
    return BoostingEnsemble(config_path)