
from torch import Tensor
from sklearn.dummy import DummyClassifier
from sentence_transformers import SentenceTransformer

from typing import List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing_extensions import override


@dataclass
class Qwen3EmbeddingConfig:
    device: str
    dtype: str
    model_id: str
    tokenizer_max_length: int
    padding: bool
    truncation: bool
    truncate_dim: int
    task_prompt: str


class BaseEmbeddingModel(ABC):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_id = self.config.model_id
        self.device = self.config.device
        self.dtype = self.config.dtype

    @abstractmethod
    def get_embeddings(self, *args, **kwargs) -> Tensor:
        ...

    @abstractmethod
    def calculate_scores(self, *args, **kwargs) -> Tensor:
        ...

    @abstractmethod
    def get_scores(self, *args, **kwargs) -> Tensor:
        ...


class Qwen3EmbeddingModel2(BaseEmbeddingModel):

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
        embeddings = self.model.encode(texts, prompt_name=prompt_name, convert_to_numpy=False, convert_to_tensor=True)

        return embeddings

    @override
    def calculate_scores(self, query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
        scores = self.model.similarity(query_embeddings, document_embeddings)

        return scores
    
    @override
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
    
