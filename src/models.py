from torch import Tensor
from sklearn.dummy import DummyClassifier
from sentence_transformers import SentenceTransformer

from typing import List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing_extensions import override

@dataclass
class BaseEmbeddingConfig:
    device: str
    dtype: str
    model_id: str

@dataclass
class Qwen3EmbeddingConfig(BaseEmbeddingConfig):
    truncate_dim: int

@dataclass
class ParaphraserConfig(BaseEmbeddingConfig):
    truncate_dim: int


class BaseEmbeddingModel(ABC):

    def __init__(self, config: BaseEmbeddingConfig):
        super().__init__()
        self.config = config
        self.model_id = self.config.model_id
        self.device = self.config.device
        self.dtype = self.config.dtype

    @abstractmethod
    def get_embeddings(self, texts: List[str], prompt_name: Optional[str] = None) -> Tensor:
        embeddings = self.model.encode(texts, prompt_name=prompt_name, convert_to_numpy=False, convert_to_tensor=True)

        return embeddings

    @abstractmethod
    def calculate_scores(self, query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
        scores = self.model.similarity(query_embeddings, document_embeddings)

        return scores

    @abstractmethod
    def get_scores(self, queries: List[str], documents: List[str]) -> Tensor:
        query_embeddings = self.get_embeddings(queries, "query")
        document_embeddings = self.get_embeddings(documents)
        scores = self.calculate_scores(query_embeddings, document_embeddings)
        
        return scores

class ParaphraserModel(BaseEmbeddingModel):

    def __init__(self, config: ParaphraserConfig):
        super().__init__(config)
        self.truncate_dim = self.config.truncate_dim
        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            truncate_dim=self.truncate_dim,
            model_kwargs={"torch_dtype": self.dtype}
        )
    
    @override
    def get_embeddings(self, texts: List[str], prompt_name: Optional[str] = None) -> Tensor:
        return super().get_embeddings(texts, prompt_name)

    @override
    def calculate_scores(self, query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
        return super().calculate_scores(query_embeddings, document_embeddings)
    
    @override
    def get_scores(self, queries: List[str], documents: List[str]) -> Tensor:
        return super().get_scores(queries, documents)

class Qwen3EmbeddingModel(BaseEmbeddingModel):

    def __init__(self, config: Qwen3EmbeddingConfig):
        super().__init__(config)
        self.truncate_dim = self.config.truncate_dim
        self.model = SentenceTransformer(
            self.model_id, 
            device=self.device,
            truncate_dim=self.truncate_dim, 
            model_kwargs={"torch_dtype": self.dtype},
        )

    @override
    def get_embeddings(self, texts: List[str], prompt_name: Optional[str] = None) -> Tensor:
        return super().get_embeddings(texts, prompt_name)

    @override
    def calculate_scores(self, query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
        return super().calculate_scores(query_embeddings, document_embeddings)
    
    @override
    def get_scores(self, queries: List[str], documents: List[str]) -> Tensor:
        return super().get_scores(queries, documents)


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
    
