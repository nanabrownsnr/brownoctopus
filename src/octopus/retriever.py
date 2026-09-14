from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim, dot_score

from octopus.tools import TOOLS
from octopus.analyzer import analyze_intents


model = SentenceTransformer("BAAI/bge-small-en-v1.5")

tool_texts = [f"{tool['name']}: {tool['description']}" for tool in TOOLS]

tool_embeddings = model.encode(
    tool_texts,
    convert_to_tensor=True,
)


# def retrieve_tools(query: str, top_k: int = 8) -> list[dict]:
#     query_embedding = model.encode(
#         query,
#         convert_to_tensor=True,
#     )

#     scores = cos_sim(query_embedding, tool_embeddings)[0]

#     ranked_indices = scores.argsort(descending=True)[:top_k]

#     return [TOOLS[index.item()] for index in ranked_indices]


def retrieve_tools(query: str, per_intent_k: int = 4) -> list[dict]:
    intents = analyze_intents(query)

    selected_indices = []

    for intent in intents:

        # query_embedding = model.encode(
        #     intent_query,
        #     convert_to_tensor=True,
        # )

        action_embedding = model.encode(
            intent["action"],
            convert_to_tensor=True,
        )

        target_embedding = model.encode(
            intent["target"] or "",
            convert_to_tensor=True,
        )

        context_embedding = model.encode(
            intent["context"],
            convert_to_tensor=True,
        )

        query_embedding = (
            1.5 * action_embedding + 1.0 * target_embedding + 1.0 * context_embedding
        )

        scores = dot_score(query_embedding, tool_embeddings)[0]

        ranked_indices = scores.argsort(descending=True)[:per_intent_k]

        print(
            f"\nINTENT: "
            f"action={intent['action']!r}, "
            f"target={intent['target']!r}, "
            f"context={intent['context']!r}"
        )

        for index in ranked_indices:
            i = index.item()
            print(f"{TOOLS[i]['name']:<30} " f"{float(scores[i]):.4f}")

        selected_indices.extend(index.item() for index in ranked_indices)

    seen = set()
    results = []

    for index in selected_indices:
        if index not in seen:
            seen.add(index)
            results.append(TOOLS[index])

    return results
