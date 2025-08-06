import torch
import pandas as pd
from sklearn.metrics import f1_score

import re
import json
from typing import List, Optional

from models import (
    BaseEmbeddingModel,
    Qwen3EmbeddingConfig,
    DummyModel,
    DummyModelConfig,
)
from constants import (
    QWEN3_EMBEDDING_CONFIG_PATH, 
    PROMPTS_PATH, 
    TRAIN_VAL_SAMPLE_DATA_PATH,
    DUMMY_MODEL_CONFIG_PATH,
)

def clean_text(row) -> str:
    text = row.Item_Name
    pack = str(row.Pack)
    unit = str(row.Unit)
    weight = str(row.Weight)
    if pack in text:
        text = text.replace(pack, "")
    if unit in text:
        text = text.replace(unit, "")
    if weight in text:
        text = text.replace(weight, "")
    text = re.sub(r'[^\w\s]', '', text)
    text = text.split()
    text = [t for t in text if not re.search(r"\d", t)]

    return " ".join(text)

def load_dummy_model():
    with open(DUMMY_MODEL_CONFIG_PATH, "r") as f:
        config_dict = json.load(f)

    try:
        config = DummyModelConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys in {DUMMY_MODEL_CONFIG_PATH}: {e}.")

    model = DummyModel(config)

    return model

def load_qwen3_embedding(model_class: BaseEmbeddingModel):
    with open(QWEN3_EMBEDDING_CONFIG_PATH, "r") as f:
        config_dict = json.load(f)

    try:
        config = Qwen3EmbeddingConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys in {QWEN3_EMBEDDING_CONFIG_PATH}: {e}.")

    model = model_class(config)

    return model

def evaluate_model(y_true: List[str], y_pred: List[str], average: str) -> float:
    return f1_score(y_true, y_pred, average=average)

def evaluate_qwen3_embedding(
        df: pd.DataFrame, 
        column_name: str,
        model_class: BaseEmbeddingModel, 
        n_samples: Optional[int] = None,
    ):
    model = load_qwen3_embedding(model_class)

    num_samples = n_samples if n_samples is not None else len(df)
    product_names = df[column_name].tolist()[:num_samples]
    classes = list(set(df["class"].tolist()))

    scores = []
    classes_idx = []
    for product_name in product_names:
        score = model.get_scores([product_name], classes)
        class_idx = torch.argmax(score, dim=1)
        scores.append(score)
        classes_idx.append(class_idx)

    y_pred = [classes[idx] for idx in classes_idx]
    y_true =  df["class"].tolist()[:num_samples]

    model_score = evaluate_model(y_true, y_pred, "weighted")

    return model_score

def evaluate_dummy_model(df: pd.DataFrame, column_name: str, n_samples: Optional[int] = None):
    model = load_dummy_model()

    num_samples = n_samples if n_samples is not None else len(df)
    product_names = df[column_name].tolist()[:num_samples]
    labels = df["class"].tolist()[:num_samples]

    model.fit_model(product_names, labels)
    y_pred = model.predict(product_names)
    y_true =  df["class"].tolist()[:num_samples]

    model_score = evaluate_model(y_true, y_pred, "weighted")

    return model_score