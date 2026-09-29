import os
from collections.abc import Iterable
from typing import Any, TYPE_CHECKING

from brown_octopus.analyzer import analyze_intents
from brown_octopus.tool_registry import capability_id, get_all_tools
from brown_octopus.selection import select_min4_bounded_max_gap
from brown_octopus.capability_scope import (
    normalize_allowed_mcp_urls,
    tool_is_allowed,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_PER_INTENT_K = 4
RETRIEVAL_METHOD = "brown_octopus_v3"


model: Any = None
TOOLS: list[dict] = []
tool_embeddings = None


class QwenCandidateRetriever:
    """Candidate retrieval only; relevance selection is a separate strategy."""

    name = "qwen_dense"

    def __init__(self, candidate_k: int | None = None) -> None:
        self.candidate_k = candidate_k

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
            ranked = rank_tools(retrieval_query)
        else:
            ranked = rank_tools(
                retrieval_query,
                allowed_mcp_urls=allowed_mcp_urls,
            )
        return ranked if self.candidate_k is None else ranked[: self.candidate_k]


def initialize_retriever() -> None:
    """Load the embedding model once."""
    global model

    if model is None:
        allow_download = os.getenv("BROWN_OCTOPUS_ALLOW_MODEL_DOWNLOAD", "0") == "1"
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                MODEL_NAME,
                trust_remote_code=True,
                local_files_only=not allow_download,
            )
        except Exception as exc:
            action = (
                "set BROWN_OCTOPUS_ALLOW_MODEL_DOWNLOAD=1 for explicit bootstrap"
                if not allow_download
                else "verify the model installation and embedding dependencies"
            )
            raise RuntimeError(
                f"Brown Octopus could not load {MODEL_NAME} from the local model cache; "
                f"{action}."
            ) from exc


def retrieve_tools(
    query: str,
    allowed_mcp_urls: Iterable[str] | None = None,
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
            ranked_tools = rank_tools(retrieval_query)
        else:
            ranked_tools = rank_tools(
                retrieval_query,
                allowed_mcp_urls=allowed_mcp_urls,
            )

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


def refresh_index():
    """
    Rebuild the tool embedding index from the current registry.
    """
    global TOOLS, tool_embeddings

    if model is None:
        raise RuntimeError("Retriever is not initialized. " "Call initialize() first.")

    TOOLS = get_all_tools()

    if not TOOLS:
        tool_embeddings = None
        return

    tool_embeddings = build_embeddings(TOOLS)

    return tool_embeddings


def build_embeddings(tools: list[dict]):
    """Encode a candidate universe without changing the active index."""
    if model is None:
        raise RuntimeError("Retriever is not initialized. Call initialize() first.")
    tool_texts = [f"{tool['name']}: {tool.get('description') or ''}" for tool in tools]
    return model.encode(
        tool_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
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
) -> list[dict]:
    if model is None:
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

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
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
