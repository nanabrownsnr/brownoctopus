from time import perf_counter

import torch
from sentence_transformers import SentenceTransformer

from evals.common.models import RetrievalResult


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


class DenseTopK:
    name = "dense_top16"

    def __init__(self, embeddings=None, model=None, k: int = 16):
        self.k = k
        self.name = f"dense_{k}"
        self.model = model
        self.embeddings = embeddings

    def initialize(self, tools: list[dict], embeddings=None) -> None:
        if self.model is None:
            self.model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        if embeddings is not None:
            self.embeddings = embeddings
        if self.embeddings is None:
            texts = [
                f"{tool['name']}: {tool.get('description') or ''}"
                for tool in tools
            ]
            self.embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        if len(self.embeddings) != len(tools):
            raise ValueError("Dense embeddings do not match the tool universe.")

    def retrieve(self, query: str, tools: list[dict], state=None) -> RetrievalResult:
        if self.model is None or self.embeddings is None:
            raise RuntimeError("DenseTopK is not initialized.")

        start = perf_counter()
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        scores = query_embedding @ self.embeddings.T
        ranked_indices = scores.argsort(descending=True)[: self.k].tolist()
        tool_ids = [tools[index]["name"] for index in ranked_indices]
        score_map = {tool_id: float(scores[index]) for tool_id, index in zip(tool_ids, ranked_indices)}

        return RetrievalResult(
            tool_ids=tool_ids,
            scores=score_map,
            latency_ms=(perf_counter() - start) * 1000.0,
            metadata={"k": self.k, "representation": "name + description"},
        )
