from torch import Tensor
from sklearn.dummy import DummyClassifier
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass

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
        safe_texts = ["" if t is None else str(t) for t in texts]
        embeddings = self.model.encode(
            safe_texts,
            prompt_name=prompt_name,
            convert_to_numpy=False,
            convert_to_tensor=True,
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
class TfidfConfig:
    analyzer: str = "char_wb"          # char-level with word-boundaries
    ngram_range: tuple = (3, 5)        # 3–5 char n-grams
    min_df: int = 2                    # ignore ultra-rare noise
    max_df: float = 0.9                # ignore very common boilerplate
    lowercase: bool = True
    sublinear_tf: bool = True          # sublinear TF scaling
    smooth_idf: bool = True            # IDF smoothing
    norm: str = "l2"                   # cosine-friendly
    strip_accents: Optional[str] = None
    stop_words: Optional[set] = None   # not used for char analyzer
    use_svd: bool = False
    svd_components: int = 200
    random_state: int = 42

class TfidfSimilarityModel:
    def __init__(self, config: Optional[TfidfConfig] = None):
        self.config = config or TfidfConfig()
        self.vectorizer = TfidfVectorizer(
            analyzer=self.config.analyzer,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            lowercase=self.config.lowercase,
            sublinear_tf=self.config.sublinear_tf,
            smooth_idf=self.config.smooth_idf,
            norm=self.config.norm,
            strip_accents=self.config.strip_accents,
            stop_words=self.config.stop_words,
            token_pattern=None if self.config.analyzer in ("char", "char_wb") else r'(?u)\b\w+\b',
            dtype=np.float32,
        )
        self._svd = None
        self._docs = None
        self._doc_matrix = None
        self.model_id = "tfidf"

    def fit_corpus(self, texts: List[str]) -> None:
        texts = ["" if t is None else str(t) for t in texts]
        X = self.vectorizer.fit_transform(texts)
        if self.config.use_svd:
            from sklearn.decomposition import TruncatedSVD
            self._svd = TruncatedSVD(
                n_components=self.config.svd_components,
                random_state=self.config.random_state
            )
            self._svd.fit(X)

    def transform(self, texts: List[str]):
        texts = ["" if t is None else str(t) for t in texts]
        X = self.vectorizer.transform(texts)
        if self.config.use_svd and self._svd is not None:
            X = self._svd.transform(X)
        return X

    def build_class_centroids(self, texts: List[str], labels: List[str]):
        import numpy as np
        X = self.transform(texts)  # after fit_corpus
        classes = sorted(set(labels))
        centroids = []
        for c in classes:
            mask = np.array([lbl == c for lbl in labels])
            Xc = X[mask]
            if hasattr(Xc, "toarray"):  # sparse
                vec = Xc.mean(axis=0)
                vec = np.asarray(vec).ravel()
            else:
                vec = Xc.mean(axis=0)
            centroids.append(vec)
        centroid_mat = np.vstack(centroids)
        return centroid_mat, classes

    def get_scores_against_centroids(self, queries: List[str], centroid_matrix) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity
        Q = self.transform(queries)
        return cosine_similarity(Q, centroid_matrix)

    def fit_documents(self, documents: List[str]):
        self._docs = list(map(lambda x: "" if x is None else str(x), documents))
        self._doc_matrix = self.vectorizer.fit_transform(self._docs)

    def get_scores(self, queries: List[str], documents: List[str]) -> np.ndarray:
        # Fit TF-IDF on documents (class labels) so query and label share the same vocab space
        self.fit_documents(documents)
        Q = self.vectorizer.transform(list(map(lambda x: "" if x is None else str(x), queries)))
        return cosine_similarity(Q, self._doc_matrix)