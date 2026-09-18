import asyncio

from octopus.sources import load_mcp_urls
from octopus.mcp_discovery import discover_universe
from octopus.tool_registry import set_tools, get_all_tools
from octopus.retriever import refresh_index, retrieve_tools


QUERIES = [
    # 1. Clean independent actions
    (
        "clean_compound",
        "check my leave balance, search the web for Nvidia news, "
        "and email the results to Tom",
    ),
    # 2. Second action depends on output of first
    (
        "dependent_output",
        "search the web for the latest Nvidia news " "and send what you find to Tom",
    ),
    # 3. Pronoun/reference dependency
    (
        "pronoun_reference",
        "find the latest Nvidia news and email it to Tom",
    ),
    # 4. Shared object between actions
    (
        "shared_object",
        "find and email Tom the latest Nvidia news",
    ),
    # 5. Second action depends on information from first
    (
        "dependent_information",
        "check my leave balance and let my manager know " "how many days I have left",
    ),
    # 6. Output from one tool becomes input to another
    (
        "cross_tool_output",
        "get the sprint work items and create a Word document " "summarizing them",
    ),
    # 7. Nested/content clause
    (
        "nested_clause",
        "email Tom to tell him that I will check " "the sprint board tomorrow",
    ),
    # 8. Conditional action
    (
        "conditional",
        "check my leave balance and if I have enough days " "submit a leave request",
    ),
    # 9. Single-intent email baseline
    (
        "email_baseline",
        "send an email to Tom",
    ),
    # 10. Single-intent web-search baseline
    (
        "web_baseline",
        "search the web for the latest Nvidia news",
    ),
]


async def main():
    print("=" * 80)
    print("BROWN OCTOPUS — REAL MCP RETRIEVAL TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load MCP URLs
    # ---------------------------------------------------------

    urls = load_mcp_urls("data/mcps.json")

    print(f"\nConfigured MCPs: {len(urls)}")

    # ---------------------------------------------------------
    # 2. Discover real tools from MCP servers
    # ---------------------------------------------------------

    print("\nDiscovering MCP universe...\n")

    tools = await discover_universe(urls)

    print(f"\nDiscovered unique tools: {len(tools)}")

    # ---------------------------------------------------------
    # 3. Replace Octopus registry with real discovered universe
    # ---------------------------------------------------------

    set_tools(tools)

    print(f"Registry tools: {len(get_all_tools())}")

    # ---------------------------------------------------------
    # 4. Build semantic retrieval index
    # ---------------------------------------------------------

    print("\nBuilding retrieval index...")

    refresh_index()

    print("Index ready.")

    # ---------------------------------------------------------
    # 5. Run retrieval experiments
    # ---------------------------------------------------------

    for number, (name, query) in enumerate(QUERIES, start=1):

        print("\n\n")
        print("=" * 80)
        print(f"TEST {number}: {name}")
        print("=" * 80)

        print(f"\nQUERY:\n{query}\n")

        results = retrieve_tools(query)

        print("\nFINAL RETRIEVED TOOLS:")

        if not results:
            print("  <none>")
            continue

        for index, tool in enumerate(results, start=1):
            print(f"{index:>2}. " f"{tool['name']}")

    # ---------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------

    print("\n\n")
    print("=" * 80)
    print("TEST RUN COMPLETE")
    print("=" * 80)

    print(
        f"""
MCPs configured:     {len(urls)}
Unique tools:        {len(get_all_tools())}
Queries tested:      {len(QUERIES)}
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
