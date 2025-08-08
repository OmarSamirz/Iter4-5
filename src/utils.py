
import re
import json
from typing import List

from models import (
    DummyModel,
    DummyModelConfig,
    SentenceEmbeddingModel, 
    SentenceEmbeddingConfig,
)  
from constants import (
    QWEN3_EMBEDDING_CONFIG_PATH,
    PARAPHRASER_EMBEDDING_CONFIG_PATH,
    DUMMY_MODEL_CONFIG_PATH,
    BGE_M3_CONFIG_PATH,
    ALL_STOPWORDS
)

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
    return re.sub(r'[^\w\s]', '', text)

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