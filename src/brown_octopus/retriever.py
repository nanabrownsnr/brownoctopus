from collections.abc import Iterable
from typing import Any, TYPE_CHECKING

from brown_octopus.analyzer import analyze_intents
from brown_octopus.tool_registry import capability_id, get_all_tools
from brown_octopus.selection import select_min4_bounded_max_gap
from brown_octopus.capability_scope import (
    normalize_allowed_mcp_urls,
    tool_is_allowed,
)
from brown_octopus.embedding import (
    MODEL_NAME,
    EmbeddingProvider,
    LocalEmbeddingProvider,
)
from brown_octopus.device import detect_compute_device

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


DEFAULT_PER_INTENT_K = 4
RETRIEVAL_METHOD = "brown_octopus_v3"


model: Any = None
TOOLS: list[dict] = []
tool_embeddings = None
model_device = "cpu"
embedding_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider | None:
    """Return the provider currently backing the shared retriever."""
    return embedding_provider


def _active_provider(
    provider: EmbeddingProvider | None = None,
) -> EmbeddingProvider | None:
    """Resolve a provider without breaking legacy model monkeypatch seams."""
    if provider is not None:
        return provider
    current = embedding_provider
    if current is None:
        return None
    provider_model = getattr(current, "model", None)
    if model is None or provider_model is None or model is provider_model:
        return current
    return None


class QwenCandidateRetriever:
    """Candidate retrieval only; relevance selection is a separate strategy."""

    name = "qwen_dense"

    def __init__(
        self,
        candidate_k: int | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.candidate_k = candidate_k
        self.embedding_provider = embedding_provider

    def retrieve(
        self,
        query: str,
        intent: dict,
        allowed_mcp_urls: Iterable[str] | None = None,
    ) -> list[dict]:
        retrieval_query = (
            intent.get("retrieval_text")
            or intent.get("text")
            or query
        )
        if allowed_mcp_urls is None:
            if self.embedding_provider is None:
                ranked = rank_tools(retrieval_query)
            else:
                ranked = rank_tools(
                    retrieval_query,
                    embedding_provider=self.embedding_provider,
                )
        else:
            kwargs = {"allowed_mcp_urls": allowed_mcp_urls}
            if self.embedding_provider is not None:
                kwargs["embedding_provider"] = self.embedding_provider
            ranked = rank_tools(retrieval_query, **kwargs)
        return ranked if self.candidate_k is None else ranked[: self.candidate_k]


def initialize_retriever(provider: EmbeddingProvider | None = None) -> None:
    """Load the embedding model once."""
    global model, model_device, embedding_provider

    if provider is not None:
        embedding_provider = provider
        healthcheck = getattr(provider, "healthcheck", None)
        if healthcheck is not None:
            healthcheck()
        model = getattr(provider, "model", None)
        model_device = getattr(provider, "device", None) or model_device
        return

    if model is None:
        provider = LocalEmbeddingProvider(
            model_name=MODEL_NAME,
            device=detect_compute_device(),
        )
        provider.initialize()
        embedding_provider = provider
        model = provider.model
        model_device = provider.device or "cpu"


def retrieve_tools(
    query: str,
    allowed_mcp_urls: Iterable[str] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[dict]:
    intents = analyze_intents(query)

    selected_tools = []
    seen_tools = set()

    for intent in intents:
        retrieval_query = (
            intent.get("retrieval_text")
            or intent.get("text")
            or query
        )
        if allowed_mcp_urls is None:
            if embedding_provider is None:
                ranked_tools = rank_tools(retrieval_query)
            else:
                ranked_tools = rank_tools(
                    retrieval_query,
                    embedding_provider=embedding_provider,
                )
        else:
            kwargs = {"allowed_mcp_urls": allowed_mcp_urls}
            if embedding_provider is not None:
                kwargs["embedding_provider"] = embedding_provider
            ranked_tools = rank_tools(retrieval_query, **kwargs)

        intent_tools = select_min4_bounded_max_gap(ranked_tools)

        for tool in intent_tools:
            tool_name = capability_id(tool)

            if tool_name in seen_tools:
                continue

            seen_tools.add(tool_name)
            selected_tools.append(tool)

    return selected_tools


def retrieve_candidates(
    query: str,
    candidate_retriever: QwenCandidateRetriever | None = None,
) -> list[dict]:
    """Return candidates for all intents without applying a selector."""
    analyzer_intents = analyze_intents(query)
    retriever = candidate_retriever or QwenCandidateRetriever()
    candidates = []
    for intent in analyzer_intents:
        candidates.append({
            "intent": intent,
            "candidates": retriever.retrieve(query, intent),
        })
    return candidates


def refresh_index(
    provider: EmbeddingProvider | None = None,
):
    """
    Rebuild the tool embedding index from the current registry.
    """
    global TOOLS, tool_embeddings

    if model is None and provider is None and embedding_provider is None:
        raise RuntimeError("Retriever is not initialized. " "Call initialize() first.")

    TOOLS = get_all_tools()

    if not TOOLS:
        tool_embeddings = None
        return

    tool_embeddings = build_embeddings(TOOLS, embedding_provider=provider)

    return tool_embeddings


def build_embeddings(
    tools: list[dict],
    embedding_provider: EmbeddingProvider | None = None,
):
    """Encode a candidate universe without changing the active index."""
    provider = _active_provider(embedding_provider)
    if model is None and provider is None:
        raise RuntimeError("Retriever is not initialized. Call initialize() first.")
    tool_texts = [f"{tool['name']}: {tool.get('description') or ''}" for tool in tools]
    if provider is not None:
        return provider.embed(tool_texts)
    return model.encode(
        tool_texts, convert_to_tensor=True, normalize_embeddings=True
    )


def load_index(
    tools: list[dict],
    embeddings,
) -> None:
    global TOOLS, tool_embeddings

    TOOLS = tools.copy()
    tool_embeddings = embeddings


def rank_tools(
    query: str,
    allowed_mcp_urls: Iterable[str] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[dict]:
    provider = _active_provider(embedding_provider)
    if model is None and provider is None:
        raise RuntimeError("Retriever is not initialized. " "Call initialize() first.")

    if tool_embeddings is None:
        raise RuntimeError("Tool index is not loaded.")

    if not query.strip():
        return []

    allowed_urls = normalize_allowed_mcp_urls(allowed_mcp_urls)
    eligible_indices = [
        index
        for index, tool in enumerate(TOOLS)
        if tool_is_allowed(tool, allowed_urls)
    ]

    if not eligible_indices:
        return []

    if provider is not None:
        query_embedding = provider.embed(query)
    else:
        query_embedding = model.encode(
            query, convert_to_tensor=True, normalize_embeddings=True
        )

    scores = query_embedding @ tool_embeddings[eligible_indices].T

    ranked_indices = scores.argsort(descending=True)

    ranked_tools = []

    for rank, local_index in enumerate(
        ranked_indices.tolist(),
        start=1,
    ):
        source_tool = TOOLS[eligible_indices[local_index]]
        ranked_tool = dict(source_tool)
        ranked_tool.update(
            {
                "rank": rank,
                "score": float(scores[local_index]),
            }
        )
        ranked_tools.append(ranked_tool)

    return ranked_tools
