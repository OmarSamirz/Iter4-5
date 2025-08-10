import torch
import pandas as pd
from sklearn.metrics import f1_score

import time
import statistics
from typing import List, Optional

from models import SentenceEmbeddingModel
from utils import load_random_forest_embedding_model, load_embedding_model, load_dummy_model, load_tfidf, load_tfidf_model

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
    classes = list(set(df["class"].tolist()))

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
    labels = df["label"].tolist()[:num_samples]

    model.fit_model(product_names, labels)
    y_pred = model.predict(product_names)
    y_true =  df["label"].tolist()[:num_samples]

    model_score = evaluation_score(y_true, y_pred, "weighted")

    return model_score

def evaluate_random_forest_embedding_model(train_data_path: str, test_data_path: str):
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)
    model = load_random_forest_embedding_model()

    X_train, y_train = df_train["cleaned_text"].tolist(), df_train["label"].tolist()
    X_test, y_test = df_test["cleaned_text"].tolist(), df_test["label"].tolist()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    model_score = evaluation_score(y_test, y_pred, "weighted")

    return model_score, y_pred


def evaluate_tfidf_model(train_data_path: str, test_data_path: str):
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)
    model = load_tfidf_model()

    X_train, y_train = df_train["cleaned_text"].tolist(), df_train["label"].tolist()
    X_test, y_test = df_test["cleaned_text"].tolist(), df_test["label"].tolist()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    model_score = evaluation_score(y_test, y_pred, "weighted")

    return model_score

def evaluate_tfidf(train_data_path: str, test_data_path: str):
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)
    model = load_tfidf()

    X_train, y_train = df_train["cleaned_text"].tolist(), list(set(df_train["class"].tolist()))
    X_test, y_test = df_test["cleaned_text"].tolist(), df_test["class"].tolist()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    model_score = evaluation_score(y_test, y_pred, "weighted")

    return model_score

def evaluate_model_topk(y_true: List[str], y_pred_topk: List[List[str]], average: str, k: int) -> float:
    y_pred_adjusted = []
    for true_label, pred_list in zip(y_true, y_pred_topk):
        if true_label in pred_list:
            y_pred_adjusted.append(true_label)  
        else:
            y_pred_adjusted.append(pred_list[0])  

    return f1_score(y_true, y_pred_adjusted, average=average)

def evaluate_embedding_topk_model(
        df: pd.DataFrame, 
        column_name: str,
        config_path: str,
        n_samples: Optional[int] = None,
        k: int = 3,
    ):
    """
    Evaluate an embedding model using Top-k accuracy logic for F1 score.
    """
    model = load_embedding_model(config_path)

    num_samples = n_samples if n_samples is not None else len(df)
    product_names = df[column_name].tolist()[:num_samples]
    classes = list(set(df["class"].tolist()))

    topk_preds = []
    scores = []
    for product_name in product_names:
        score = model.get_scores([product_name], classes)  
        topk_indices = torch.topk(score, k, dim=1).indices.squeeze(0).tolist()
        topk_labels = [classes[idx] for idx in topk_indices]
        scores.append(score)
        topk_preds.append(topk_labels)

    y_true = df["class"].tolist()[:num_samples]

    model_score = evaluate_model_topk(y_true, topk_preds, "weighted", k)
    return model_score
