import asyncio

from octopus import Octopus
from octopus.analyzer import analyze_intents
from octopus.retriever import (
    rank_tools,
    retrieve_tools_adaptive,
)


QUERIES = [
    "Send an email to Tom",
    "Create a Word report",
    "Get my leave applications",
    "Submit a leave application",
    "Find a GitHub repository",
    "Check the deployment status on Railway",
    "Get the current sprint work items",
    "Find the latest Nvidia news and email a summary to Tom",
]


async def main():
    octopus = Octopus(
        index_path="data/indexes/default",
        catalog_path="data/mcps.json",
    )

    await octopus.initialize()

    for query in QUERIES:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")

        intents = analyze_intents(query)

        print(f"\nINTENTS ({len(intents)}):")

        for intent in intents:
            print(f"  - {intent['text']}")

        v1_tools = octopus.retrieve(query)
        v2_tools = retrieve_tools_adaptive(query)

        print(f"\nV1: {len(v1_tools)} tools")
        for tool in v1_tools:
            print(f"  - {tool['name']}")

        print(f"\nV2: {len(v2_tools)} tools")
        for tool in v2_tools:
            print(
                f"  - {tool['name']}"
                f"  score={tool['score']:.4f}"
            )


if __name__ == "__main__":
    asyncio.run(main())