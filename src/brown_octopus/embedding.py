"""Embedding provider contracts and local/HTTP implementations."""

import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from brown_octopus.device import detect_compute_device


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


class EmbeddingProvider(Protocol):
    """Provider used for both index and query embeddings."""

    @property
    def model_id(self) -> str: ...

    def embed(self, texts: str | list[str]): ...

    def healthcheck(self) -> None: ...


class LocalEmbeddingProvider:
    """Load any compatible local SentenceTransformers embedding model.

    ``model_name`` may be a model identifier or a local model directory.
    ``model_path`` is provided as an explicit, readable alias for local paths.
    """

    model_id = MODEL_NAME

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        *,
        model_path: str | Path | None = None,
        model_id: str | None = None,
        device: str | None = None,
    ) -> None:
        if model_path is not None and model_name != MODEL_NAME:
            raise ValueError("Specify either model_name or model_path, not both.")
        self.model_name = str(model_path) if model_path is not None else model_name
        self.model_id = model_id or self.model_name
        self.device = device
        self.model: Any = None
        self.dimension: int | None = None

    def initialize(self) -> None:
        if self.model is not None:
            return

        allow_download = os.getenv("BROWN_OCTOPUS_ALLOW_MODEL_DOWNLOAD", "0") == "1"
        try:
            from sentence_transformers import SentenceTransformer

            self.device = self.device or detect_compute_device()
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                local_files_only=not allow_download,
                device=self.device,
            )
            get_dimension = getattr(self.model, "get_embedding_dimension", None)
            if get_dimension is None:
                get_dimension = getattr(
                    self.model,
                    "get_sentence_embedding_dimension",
                    None,
                )
            if get_dimension is not None:
                self.dimension = get_dimension()
        except Exception as exc:
            action = (
                "set BROWN_OCTOPUS_ALLOW_MODEL_DOWNLOAD=1 for explicit bootstrap"
                if not allow_download
                else "verify the model installation and embedding dependencies"
            )
            raise RuntimeError(
                f"Brown Octopus could not load {self.model_name} from the local model cache; "
                f"{action}."
            ) from exc

    def embed(self, texts: str | list[str]):
        self.initialize()
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    def healthcheck(self) -> None:
        self.initialize()


class HttpEmbeddingProvider:
    """Embedding provider for an OpenAI-compatible HTTP embeddings endpoint."""

    def __init__(
        self,
        url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.model_id = model
        self.api_key = api_key
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.dimension: int | None = None

    def embed(self, texts: str | list[str]):
        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)
        if not inputs:
            return self._tensor([])

        request_headers = {
            "Content-Type": "application/json",
            **self.headers,
        }
        if self.api_key:
            request_headers.setdefault(
                "Authorization",
                f"Bearer {self.api_key}",
            )

        body = json.dumps({"model": self.model_id, "input": inputs}).encode()
        request = urllib.request.Request(
            self.url,
            data=body,
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Embedding API request failed for model '{self.model_id}'."
            ) from exc

        vectors = self._vectors(payload, len(inputs))
        result = self._tensor(vectors)
        return result[0] if single else result

    @staticmethod
    def _vectors(payload: Any, expected: int) -> list[list[float]]:
        if isinstance(payload, dict) and isinstance(payload.get("embeddings"), list):
            vectors = payload["embeddings"]
            if len(vectors) != expected:
                raise RuntimeError(
                    "Embedding API response must contain one embedding per input."
                )
            if not all(isinstance(vector, list) and vector for vector in vectors):
                raise RuntimeError(
                    "Embedding API response items must contain numeric embeddings."
                )
            dimension = len(vectors[0])
            if any(len(vector) != dimension for vector in vectors):
                raise RuntimeError("Embedding API returned inconsistent dimensions.")
            return vectors

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != expected:
            raise RuntimeError(
                "Embedding API response must contain one data item per input."
            )

        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in ordered]
        if not all(isinstance(vector, list) and vector for vector in vectors):
            raise RuntimeError(
                "Embedding API response items must contain numeric embeddings."
            )
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise RuntimeError("Embedding API returned inconsistent dimensions.")
        return vectors

    def _tensor(self, vectors: list[list[float]]):
        import torch

        if not vectors:
            return torch.empty((0, 0), dtype=torch.float32)
        self.dimension = len(vectors[0])
        tensor = torch.tensor(vectors, dtype=torch.float32)
        return torch.nn.functional.normalize(tensor, p=2, dim=1)

    def healthcheck(self) -> None:
        self.embed(["brown octopus healthcheck"])
