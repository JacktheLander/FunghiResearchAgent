from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class EmbeddingBackend(ABC):
    name: str

    @abstractmethod
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError


class HashingEmbeddingBackend(EmbeddingBackend):
    """Deterministic local fallback for tests and offline demos."""

    def __init__(self, n_features: int = 384) -> None:
        self.name = f"hashing-{n_features}"
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            stop_words="english",
        )

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(list(texts))
        return matrix.astype("float32").toarray()


class SentenceTransformersBackend(EmbeddingBackend):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), normalize_embeddings=True),
            dtype="float32",
        )


class OpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is not installed") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")
        self.name = model_name
        self.client = OpenAI()

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        batch = list(texts)
        response = self.client.embeddings.create(model=self.name, input=batch)
        vectors = [item.embedding for item in response.data]
        return np.asarray(vectors, dtype="float32")


def build_embedding_backend(backend: str, model: str) -> EmbeddingBackend:
    if backend == "sentence_transformers":
        try:
            return SentenceTransformersBackend(model)
        except RuntimeError:
            return HashingEmbeddingBackend()
    if backend == "openai":
        return OpenAIEmbeddingBackend(model)
    return HashingEmbeddingBackend()
