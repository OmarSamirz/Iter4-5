import torch
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
from models import TfidfSimilarityModel

import time
import statistics
from typing import List, Optional

from models import SentenceEmbeddingModel
from utils import load_embedding_model, load_dummy_model

def evaluation_score(y_true: List[str], y_pred: List[str], average: str) -> float:
    return f1_score(y_true, y_pred, average=average)

def multi_column_embedding_model_evaluation(
    df: pd.DataFrame,
    config_path: str,
    columns_name: List[str],
    n_samples: Optional[int] = None, 
) -> None:
    model = load_embedding_model(config_path)
    for column_name in columns_name:
        score = evaluate_embedding_model(df=df, column_name=column_name, n_samples=n_samples, model=model)
        print(f"The score of using {column_name}: {score}")

def evaluate_embedding_model(
        df: pd.DataFrame,
        column_name: str,
        config_path: Optional[str] = None,
        n_samples: Optional[int] = None,
        model: Optional[SentenceEmbeddingModel] = None,
    ):
    if not ((model is None) ^ (config_path is None)):
        raise ValueError("You must specify model or config_path.")
    if model is None:
        model = load_embedding_model(config_path)

    print(f"You are evaluating: {model.model_id}")
    num_samples = n_samples if n_samples is not None else len(df)
    product_names = df[column_name].tolist()[:num_samples]
    classes = sorted(set(df["class"].tolist()))
    
    scores = []
    classes_idx = []
    runtime = []
    for product_name in product_names:
        start = time.time()
        score = model.get_scores([product_name], classes)
        end = time.time()
        runtime.append(end-start)
        class_idx = torch.argmax(score, dim=1)
        scores.append(score)
        classes_idx.append(class_idx)

    print(f"Average time taken for a single example: {statistics.mean(runtime)} seconds\nNumber of examples: {len(runtime)}")
    y_pred = [classes[idx] for idx in classes_idx]
    y_true =  df["class"].tolist()[:num_samples]

    model_score = evaluation_score(y_true, y_pred, "weighted")

    return model_score

def evaluate_dummy_model(df: pd.DataFrame, column_name: str, n_samples: Optional[int] = None):
    model = load_dummy_model()

    num_samples = n_samples if n_samples is not None else len(df)
    product_names = df[column_name].tolist()[:num_samples]
    labels = df["class"].tolist()[:num_samples]

    model.fit_model(product_names, labels)
    y_pred = model.predict(product_names)
    y_true =  df["class"].tolist()[:num_samples]

    model_score = evaluation_score(y_true, y_pred, "weighted")

    return model_score

def evaluate_tfidf_model(
    df: pd.DataFrame,
    column_name: str,
    n_samples: Optional[int] = None,
) -> float:
    model = TfidfSimilarityModel()
    num_samples = n_samples if n_samples is not None else len(df)
    product_names = df[column_name].astype(str).tolist()[:num_samples]
    labels = df["class"].astype(str).tolist()[:num_samples]

    model.fit_corpus(product_names)
    centroid_mat, classes_ordered = model.build_class_centroids(product_names, labels)
    scores = model.get_scores_against_centroids(product_names, centroid_mat)
    pred_idx = np.argmax(scores, axis=1)
    y_pred = [classes_ordered[i] for i in pred_idx]
    y_true = labels
    return evaluation_score(y_true, y_pred, "weighted")


def evaluate_ensemble_model(
    df: pd.DataFrame,
    column_name: str,
    paraphraser_config_path: str,
    alpha: float = 0.2,  # start lower since TF-IDF is stronger
    n_samples: Optional[int] = None,
) -> float:
    """
    Ensemble TF-IDF (centroids) with embedding similarities.
    combined = alpha * emb_probs + (1 - alpha) * tfidf_probs
    """
    num_samples = n_samples if n_samples is not None else len(df)
    product_names = df[column_name].astype(str).tolist()[:num_samples]
    labels = df["class"].astype(str).tolist()[:num_samples]
    classes = sorted(set(labels))

    # 1) TF-IDF scores using the SAME centroid approach as in evaluate_tfidf_model
    tfidf_model = TfidfSimilarityModel()
    tfidf_model.fit_corpus(product_names)
    centroid_mat, classes_ordered = tfidf_model.build_class_centroids(product_names, labels)

    # Reorder centroids to match the "classes" list used everywhere else
    idx_map = {c: i for i, c in enumerate(classes_ordered)}
    order = [idx_map[c] for c in classes]
    centroid_mat_aligned = centroid_mat[order]
    tfidf_scores = tfidf_model.get_scores_against_centroids(product_names, centroid_mat_aligned)  # (N, C) numpy

    # 2) Embedding scores (batched)
    emb_model = load_embedding_model(paraphraser_config_path)
    emb_scores = emb_model.get_scores(product_names, classes).cpu().numpy()  # (N, C)

    # 3) Row-wise softmax normalization for each model
    def row_softmax(a: np.ndarray) -> np.ndarray:
        a = a - a.max(axis=1, keepdims=True)
        ea = np.exp(a)
        return ea / ea.sum(axis=1, keepdims=True)

    tfidf_probs = row_softmax(tfidf_scores)
    emb_probs = row_softmax(emb_scores)

    # 4) Weighted blend
    combined = alpha * emb_probs + (1.0 - alpha) * tfidf_probs
    pred_idx = np.argmax(combined, axis=1)
    y_pred = [classes[i] for i in pred_idx]
    y_true = labels

    return evaluation_score(y_true, y_pred, "weighted")