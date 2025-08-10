import torch
import numpy as np
from torch import Tensor
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

import re
import json

from typing import List, Optional
from dataclasses import dataclass

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

@dataclass
class SentenceEmbeddingConfig:
    device: str
    dtype: str
    model_id: str
    truncate_dim: int
    convert_to_numpy: bool
    convert_to_tensor: bool

class SentenceEmbeddingModel:

    def __init__(self, config: SentenceEmbeddingConfig):
        super().__init__()
        self.config = config
        self.model_id = config.model_id
        self.device = config.device
        self.dtype = config.dtype
        self.truncate_dim = config.truncate_dim

        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            truncate_dim=self.truncate_dim,
            model_kwargs={"torch_dtype": self.dtype}
        )

    def get_embeddings(self, texts: List[str], prompt_name: Optional[str] = None) -> Tensor:
        embeddings = self.model.encode(
            texts, 
            prompt_name=prompt_name, 
            convert_to_numpy=self.config.convert_to_numpy,
            convert_to_tensor=self.config.convert_to_tensor
        )

        return embeddings

    def calculate_scores(self, query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
        scores = self.model.similarity(query_embeddings, document_embeddings)

        return scores

    def get_scores(self, queries: List[str], documents: List[str]) -> Tensor:
        query_embeddings = self.get_embeddings(queries, "query")
        document_embeddings = self.get_embeddings(documents)
        scores = self.calculate_scores(query_embeddings, document_embeddings)
        
        return scores


@dataclass
class TfidfClassifierConfig:
    analyzer: str = "char_wb"
    ngram_range: tuple = (3, 5)
    min_df: int = 2
    max_df: float = 0.9
    lowercase: bool = True
    sublinear_tf: bool = True
    smooth_idf: bool = True
    norm: str = "l2"
    strip_accents: Optional[str] = None
    stop_words: Optional[set] = None


class Tfidf:

    def __init__(self, config: Optional[TfidfClassifierConfig]):
        self.vectorizer = TfidfVectorizer(
            analyzer=config.analyzer,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            max_df=config.max_df,
            lowercase=config.lowercase,
            sublinear_tf=config.sublinear_tf,
            smooth_idf=config.smooth_idf,
            norm=config.norm,
            strip_accents=config.strip_accents,
            stop_words=config.stop_words,
            token_pattern=None if config.analyzer in ("char", "char_wb") else r'(?u)\b\w+\b',
        )

        self.product_vectors = None
        self.class_vectors = None
        self.class_names = None

    def fit(self, product_names, classes):
        corpus = product_names + classes

        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        self.class_names = classes
        self.product_vectors = tfidf_matrix[:len(product_names)]
        self.class_vectors = tfidf_matrix[len(product_names):]

    def predict(self, product_name):
        vector = self.vectorizer.transform(product_name)
        scores = cosine_similarity(vector, self.class_vectors)
        class_idx = np.argmax(scores, axis=1)
        class_name = [self.class_names[idx] for idx in class_idx]

        return class_name


class TfidfClassifier:

    def __init__(self, config: Optional[TfidfClassifierConfig]):
        self.vectorizer = TfidfVectorizer(
            analyzer=config.analyzer,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            max_df=config.max_df,
            lowercase=config.lowercase,
            sublinear_tf=config.sublinear_tf,
            smooth_idf=config.smooth_idf,
            norm=config.norm,
            strip_accents=config.strip_accents,
            stop_words=config.stop_words,
            token_pattern=None if config.analyzer in ("char", "char_wb") else r'(?u)\b\w+\b',
        )

        self.clf = None

    def fit(self, X_train, y_train):
        self.clf = Pipeline(
            [
                ("vectorizer_tfidf", self.vectorizer),
                ("random_forest", RandomForestClassifier())
            ]
        )
        self.clf.fit(X_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


class RandomForestEmbeddingModel:

    def __init__(self, config: SentenceEmbeddingConfig) -> None:
        self.random_forest = RandomForestClassifier()
        self.embedding_model = SentenceEmbeddingModel(config)

    def fit(self, X_train, y_train):
        embeddings = self.embedding_model.get_embeddings(X_train, "query").tolist()
        self.random_forest.fit(embeddings, y_train)

    def predict(self, x):
        embeddings = self.embedding_model.get_embeddings(x).tolist()
        pred = self.random_forest.predict(embeddings)

        return pred


@dataclass
class DummyModelConfig:
    strategy: str


class DummyModel:

    def __init__(self, config: DummyModelConfig):
        self.model = DummyClassifier(strategy=config.strategy)

    def fit_model(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
@dataclass
class QwenConfig:
    model_id: str
    enable_thinking: bool = False
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0
    device_map: str = "cuda"

class QwenLLMClassifier:
    def __init__(self, config: QwenConfig, prompt_file: str = "prompt.txt"):
        self.config = config
        # Load prompt from file
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read().strip()
            
        quant_cfg = None
        if _HAS_BNB:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map=config.device_map,
            torch_dtype="auto",
            quantization_config=quant_cfg,
        )

    def _messages(self, item_name: str, categories: List[str]) -> List[dict]:
        cats_str = " | ".join(categories)
        usr = (
            f"Allowed categories (choose EXACTLY one): {cats_str}\n"
            f'Item name: "{item_name}"\n'
            'Respond as JSON: {"category": "<one_of_the_allowed_categories>"}'
        )
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": usr}]

    def _gen(self, item_name: str, categories: List[str]) -> str:
        msgs = self._messages(item_name, categories)
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=self.config.enable_thinking
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        gen = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    @staticmethod
    def _parse_category(text: str, categories: List[str]) -> str:
        text = re.sub(r"^```.*?\n|\n```$", "", text, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                cand = str(obj.get("category", "")).strip()
                if cand:
                    clower = [c.lower() for c in categories]
                    if cand.lower() in clower:
                        return categories[clower.index(cand.lower())]
                    for c in categories:
                        if c.lower() in cand.lower() or cand.lower() in c.lower():
                            return c
            except Exception:
                pass
        tlower = text.lower()
        for c in categories:
            if c.lower() in tlower:
                return c
        return categories[0]

    def predict_one(self, item_name: str, categories: List[str], return_raw: bool = False):
        raw = self._gen(item_name, categories)
        parsed = self._parse_category(raw, categories)
        return (parsed, raw) if return_raw else parsed

    def predict_batch(self, items: List[str], categories: List[str]) -> List[str]:
        return [self.predict_one(x, categories) for x in items]
    
