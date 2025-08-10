
import numpy as np
from torch import Tensor
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SentenceEmbeddingConfig:
    device: str
    dtype: str
    model_id: str
    truncate_dim: int


class SentenceEmbeddingModel:

    def __init__(self, config: SentenceEmbeddingConfig):
        super().__init__()
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
        embeddings = self.model.encode(texts, prompt_name=prompt_name, convert_to_numpy=False, convert_to_tensor=True)

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
    
