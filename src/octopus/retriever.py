from sentence_transformers import SentenceTransformer, util

from octopus.analyzer import analyze_intents
from octopus.tool_registry import get_all_tools
from octopus.selection import select_bounded_max_gap


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


def retrieve_tools(
    query: str,
) -> list[dict]:
    intents = analyze_intents(query)

    selected_tools = []
    seen_tools = set()

    for intent in intents:
        ranked_tools = rank_tools(intent["text"])

        intent_tools = select_bounded_max_gap(ranked_tools)

        for tool in intent_tools:
            tool_name = tool["name"]

            if tool_name in seen_tools:
                continue

            seen_tools.add(tool_name)
            selected_tools.append(tool)

    return selected_tools


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

    tool_texts = [f"{tool['name']}: {tool.get('description') or ''}" for tool in TOOLS]

    tool_embeddings = model.encode(
        tool_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    return tool_embeddings


def load_index(
    tools: list[dict],
    embeddings,
) -> None:
    global TOOLS, tool_embeddings

    TOOLS = tools.copy()
    tool_embeddings = embeddings


def rank_tools(
    query: str,
) -> list[dict]:
    if model is None:
        raise RuntimeError("Retriever is not initialized. " "Call initialize() first.")

    if tool_embeddings is None:
        raise RuntimeError("Tool index is not loaded.")

    if not query.strip():
        return []

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    scores = query_embedding @ tool_embeddings.T

    ranked_indices = scores.argsort(descending=True)

    ranked_tools = []

    for rank, index in enumerate(
        ranked_indices.tolist(),
        start=1,
    ):
        ranked_tools.append(
            {
                "rank": rank,
                "name": TOOLS[index]["name"],
                "description": TOOLS[index].get("description"),
                "score": float(scores[index]),
            }
        )

    return ranked_tools
