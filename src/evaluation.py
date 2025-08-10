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


def evaluate_qwen_llm(
    df: pd.DataFrame,
    config_path: str = "config/qwen_config.json",
    column_name: str = "cleaned_text",
    n_samples: Optional[int] = None,
    print_every: int = 50,
    log_csv_path: str = "qwen_predictions_log.csv",
) -> float:
    from utils import load_qwen_model
    
    num = n_samples if n_samples is not None else len(df)
    texts = df[column_name].astype(str).tolist()[:num]
    y_true = df["class"].astype(str).tolist()[:num]
    cats = sorted(set(df["class"].astype(str).tolist()))

    print(f"[Qwen] config={config_path} N={num}")
    clf = load_qwen_model(config_path)

    rows, y_pred, times = [], [], []
    t_all0 = time.time()
    for i, text in enumerate(texts, 1):
        t0 = time.time()
        pred, raw = clf.predict_one(text, cats, return_raw=True)
        t1 = time.time()
        dt = t1 - t0
        y_pred.append(pred)
        times.append(dt)
        avg = sum(times) / len(times)

        print(f"[Qwen] {i}/{num} t={dt:.3f}s avg={avg:.3f}s")

        if i % print_every == 0:
            run_f1 = evaluation_score(y_true[:i], y_pred, "weighted")
            print(f"[Qwen] running F1@{i} = {run_f1:.4f}")

        rows.append({
            "idx": i,
            "text": text,
            "true": y_true[i-1],
            "parsed": pred,
            "raw": raw,
            "latency_s": dt,
        })

    t_all = time.time() - t_all0
    avg_all = (sum(times) / len(times)) if times else 0.0
    final_f1 = evaluation_score(y_true, y_pred, "weighted")

    print(f"[Qwen] FINAL F1 = {final_f1:.4f}")
    print(f"[Qwen] avg_time_per_example = {avg_all:.3f}s | total_time = {t_all:.3f}s | N = {num}")

    try:
        pd.DataFrame(rows).to_csv(log_csv_path, index=False, encoding="utf-8-sig")
        print(f"[Qwen] wrote detailed log to {log_csv_path}")
    except Exception as e:
        print(f"[Qwen] failed to save log to CSV: {e}")

    return final_f1