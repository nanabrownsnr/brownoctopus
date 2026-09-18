from sentence_transformers import SentenceTransformer, util

from octopus.analyzer import analyze_intents
from octopus.tool_registry import get_all_tools


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_PER_INTENT_K = 4


model: SentenceTransformer | None = None
TOOLS: list[dict] = []
tool_embeddings = None


def initialize_retriever() -> None:
    """Load the embedding model once."""
    global model

    if model is None:
        model = SentenceTransformer(
            MODEL_NAME,
            trust_remote_code=True,
        )


def refresh_index() -> None:
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

    tool_texts = [f"{tool['name']}: {tool.get('description') or ''}" for tool in TOOLS]

    tool_embeddings = model.encode(
        tool_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )


def retrieve_tools(
    query: str,
    per_intent_k: int = DEFAULT_PER_INTENT_K,
) -> list[dict]:
    """
    Retrieve tools independently for each operational intent,
    then merge and deduplicate the results.

    Intent embeddings are generated as a single batch so compound
    requests do not require a separate model.encode() call for
    every intent.
    """
    if model is None:
        raise RuntimeError(
            "Retriever is not initialized. "
            "Call initialize() before processing requests."
        )

    if not query.strip():
        return []

    if not TOOLS or tool_embeddings is None:
        return []

    intents = analyze_intents(query)

    intent_texts = [
        intent["text"].strip() for intent in intents if intent["text"].strip()
    ]

    if not intent_texts:
        return []

    # Encode all operational intents in one model call.
    query_embeddings = model.encode(
        intent_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    selected_indices = []
    seen_indices = set()

    # Each intent still gets its own independent top-K search.
    for query_embedding in query_embeddings:
        scores = util.cos_sim(
            query_embedding,
            tool_embeddings,
        )[0]

        top_k = min(
            per_intent_k,
            len(TOOLS),
        )

        ranked_indices = scores.argsort(descending=True)[:top_k]

        for index in ranked_indices:
            index = int(index)

            if index in seen_indices:
                continue

            seen_indices.add(index)
            selected_indices.append(index)

    return [TOOLS[index] for index in selected_indices]
