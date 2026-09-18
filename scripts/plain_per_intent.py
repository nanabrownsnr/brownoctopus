import asyncio

from sentence_transformers import SentenceTransformer, util

from octopus.analyzer import analyze_intents
from octopus.mcp_discovery import discover_universe
from octopus.sources import load_mcp_urls


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
TOP_K_PER_INTENT = 4
MCP_CATALOG_PATH = "data/mcps.json"


TEST_CASES = [
    {
        "query": "create a Word document summarizing the sprint",
    },
    {
        "query": "send an email to Tom",
    },
    {
        "query": "find the latest Nvidia news",
    },
    {
        "query": "search the web for the latest Nvidia news and send what you find to Tom",
    },
    {
        "query": "check my leave balance, search the web for Nvidia news, and email the results to Tom",
    },
    {
        "query": "get the sprint work items and create a Word document summarizing them",
    },
]


async def main():
    # ---------------------------------------------------------
    # Discover the real tool universe
    # ---------------------------------------------------------

    mcp_urls = load_mcp_urls(MCP_CATALOG_PATH)

    tools = await discover_universe(mcp_urls)

    print(f"\nDiscovered tools: {len(tools)}")

    # ---------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------

    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # ---------------------------------------------------------
    # Build tool index
    # ---------------------------------------------------------

    tool_texts = [f"{tool['name']}: {tool['description'] or ''}" for tool in tools]

    tool_embeddings = model.encode(
        tool_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    print("Tool index ready.")

    # ---------------------------------------------------------
    # Run tests
    # ---------------------------------------------------------

    for test_case in TEST_CASES:
        query = test_case["query"]

        print("\n" + "=" * 100)
        print(f"QUERY: {query}")
        print("=" * 100)

        # -----------------------------------------------------
        # Analyze the raw user query
        # -----------------------------------------------------

        intents = analyze_intents(query)

        print(f"\nINTENTS: {len(intents)}")

        merged_tools = []
        seen_tools = set()

        # -----------------------------------------------------
        # Retrieve independently for each detected intent
        # -----------------------------------------------------

        for intent_number, intent in enumerate(
            intents,
            start=1,
        ):
            action = intent["action"]
            target = intent["target"]

            # Build a clean local retrieval query.
            #
            # Example:
            #
            # action = "create"
            # target = "Word document"
            #
            # becomes:
            #
            # "create Word document"
            #
            # We intentionally do NOT include the full original
            # context here because each operational intent gets
            # its own retrieval neighborhood.
            intent_text = intent["text"]

            print("\n" + "-" * 80)
            print(f"INTENT {intent_number}")
            print("-" * 80)

            print(f"ACTION:         {action}")
            print(f"TARGET:         {target}")
            print(f"RETRIEVAL TEXT: {intent_text}")

            # -------------------------------------------------
            # Embed this intent independently
            # -------------------------------------------------

            query_embedding = model.encode(
                intent_text,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

            scores = util.cos_sim(
                query_embedding,
                tool_embeddings,
            )[0]

            ranked_indices = scores.argsort(descending=True)[:TOP_K_PER_INTENT]

            print("\nTOP TOOLS:")

            # -------------------------------------------------
            # Collect local top-K
            # -------------------------------------------------

            for rank, index in enumerate(
                ranked_indices,
                start=1,
            ):
                index = int(index)

                tool = tools[index]
                score = scores[index].item()

                print(f"{rank:>2}. " f"{tool['name']:<60} " f"{score:.4f}")

                # Merge results from all intents while
                # preserving each intent's local ranking.
                #
                # There is deliberately NO global reranking.
                if tool["name"] not in seen_tools:
                    merged_tools.append(
                        {
                            "name": tool["name"],
                            "score": score,
                            "intent": intent_text,
                            "local_rank": rank,
                            "intent_number": intent_number,
                        }
                    )

                    seen_tools.add(tool["name"])

        # -----------------------------------------------------
        # Final merged capability set
        # -----------------------------------------------------

        print("\n" + "-" * 80)
        print("FINAL TOOL SET")
        print("-" * 80)

        for tool in merged_tools:
            print(
                f"[intent {tool['intent_number']} "
                f"rank {tool['local_rank']}] "
                f"{tool['name']:<60} "
                f"{tool['score']:.4f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
