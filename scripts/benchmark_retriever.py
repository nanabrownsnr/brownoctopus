import asyncio
from statistics import mean, median
from time import perf_counter

from octopus.analyzer import initialize_analyzer
from octopus.mcp_discovery import discover_universe
from octopus.pipeline import process_turn
from octopus.retriever import (
    initialize_retriever,
    refresh_index,
)
from octopus.sources import load_mcp_urls
from octopus.tool_registry import set_tools


TEST_CASES = [
    {
        "name": "single_email",
        "query": "send an email to Tom",
    },
    {
        "name": "single_web_search",
        "query": "find the latest Nvidia news",
    },
    {
        "name": "single_leave",
        "query": "check my leave balance",
    },
    {
        "name": "nested_word_and_sprint",
        "query": "create a Word document summarizing the sprint",
    },
    {
        "name": "compound_web_and_email",
        "query": (
            "search the web for the latest Nvidia news " "and send what you find to Tom"
        ),
    },
    {
        "name": "compound_work_items_and_word",
        "query": (
            "get the sprint work items and " "create a Word document summarizing them"
        ),
    },
    {
        "name": "three_capabilities",
        "query": (
            "check my leave balance, "
            "search the web for the latest Nvidia news, "
            "and email the results to Tom"
        ),
    },
]


WARMUP_RUNS = 1
BENCHMARK_RUNS = 5
MCP_CATALOG = "data/mcps.json"


def benchmark_query(
    query: str,
    runs: int,
) -> tuple[list[float], dict]:
    timings = []
    last_result = None

    for turn in range(1, runs + 1):
        start = perf_counter()

        last_result = process_turn(
            query=query,
            active_tools={},
            turn=turn,
        )

        elapsed_ms = (perf_counter() - start) * 1000

        timings.append(elapsed_ms)

    return timings, last_result


async def initialize_with_timings() -> tuple[list[dict], dict]:
    timings = {}

    print("\n" + "=" * 80)
    print("STARTUP PROFILE")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Analyzer model
    # ---------------------------------------------------------

    print("\nLoading analyzer...")

    start = perf_counter()

    initialize_analyzer()

    timings["analyzer"] = perf_counter() - start

    print(f"Analyzer initialization: " f"{timings['analyzer']:.2f} s")

    # ---------------------------------------------------------
    # 2. Retriever model
    # ---------------------------------------------------------

    print("\nLoading retriever...")

    start = perf_counter()

    initialize_retriever()

    timings["retriever"] = perf_counter() - start

    print(f"Retriever initialization: " f"{timings['retriever']:.2f} s")

    # ---------------------------------------------------------
    # 3. Load MCP configuration
    # ---------------------------------------------------------

    print("\nLoading MCP catalog...")

    start = perf_counter()

    mcp_urls = load_mcp_urls(MCP_CATALOG)

    timings["catalog"] = perf_counter() - start

    print(f"MCP catalog loading: " f"{timings['catalog']:.4f} s")

    print(f"Configured MCPs: {len(mcp_urls)}")

    # ---------------------------------------------------------
    # 4. Discover real MCP tool universe
    # ---------------------------------------------------------

    print("\nDiscovering MCP universe...")

    start = perf_counter()

    tools = await discover_universe(mcp_urls)

    timings["discovery"] = perf_counter() - start

    print(f"\nMCP discovery: " f"{timings['discovery']:.2f} s")

    print(f"Discovered tools: {len(tools)}")

    # ---------------------------------------------------------
    # 5. Populate registry
    # ---------------------------------------------------------

    print("\nPopulating tool registry...")

    start = perf_counter()

    set_tools(tools)

    timings["registry"] = perf_counter() - start

    print(f"Registry population: " f"{timings['registry']:.4f} s")

    # ---------------------------------------------------------
    # 6. Build embedding index
    # ---------------------------------------------------------

    print("\nBuilding tool embedding index...")

    start = perf_counter()

    refresh_index()

    timings["index"] = perf_counter() - start

    print(f"Tool index construction: " f"{timings['index']:.2f} s")

    # ---------------------------------------------------------
    # Startup summary
    # ---------------------------------------------------------

    timings["total"] = sum(timings.values())

    print("\n" + "-" * 80)
    print("STARTUP BREAKDOWN")
    print("-" * 80)

    print(f"Analyzer model:      " f"{timings['analyzer']:.2f} s")

    print(f"Retriever model:     " f"{timings['retriever']:.2f} s")

    print(f"MCP catalog:         " f"{timings['catalog']:.4f} s")

    print(f"MCP discovery:       " f"{timings['discovery']:.2f} s")

    print(f"Registry population: " f"{timings['registry']:.4f} s")

    print(f"Tool indexing:       " f"{timings['index']:.2f} s")

    print(f"Measured startup:    " f"{timings['total']:.2f} s")

    return tools, timings


async def main():
    print("=" * 80)
    print("BROWN OCTOPUS BENCHMARK")
    print("=" * 80)

    # ---------------------------------------------------------
    # Cold startup
    # ---------------------------------------------------------

    tools, startup_timings = await initialize_with_timings()

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("WARM RUNTIME")
    print("=" * 80)

    print("\nWarming up runtime...")

    for _ in range(WARMUP_RUNS):
        process_turn(
            query="send an email to Tom",
            active_tools={},
            turn=1,
        )

    print("Warm-up complete.")

    # ---------------------------------------------------------
    # Warm request benchmark
    # ---------------------------------------------------------

    all_timings = []

    for case in TEST_CASES:
        timings, result = benchmark_query(
            query=case["query"],
            runs=BENCHMARK_RUNS,
        )

        all_timings.extend(timings)

        retrieved_names = [tool["name"] for tool in result["retrieved_tools"]]

        context_names = [tool["name"] for tool in result["tools"]]

        print("\n" + "-" * 80)
        print(case["name"])
        print("-" * 80)

        print(f"Query: {case['query']}")

        print("Runs: " + ", ".join(f"{timing:.2f} ms" for timing in timings))

        print(f"Mean:   {mean(timings):.2f} ms")

        print(f"Median: {median(timings):.2f} ms")

        print(f"Min:    {min(timings):.2f} ms")

        print(f"Max:    {max(timings):.2f} ms")

        print(f"Retrieved tools: " f"{len(retrieved_names)}")

        for name in retrieved_names:
            print(f"  - {name}")

        print(f"Agent context tools: " f"{len(context_names)}")

    # ---------------------------------------------------------
    # Overall summary
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Analyzer startup:    " f"{startup_timings['analyzer']:.2f} s")

    print(f"Retriever startup:   " f"{startup_timings['retriever']:.2f} s")

    print(f"MCP discovery:       " f"{startup_timings['discovery']:.2f} s")

    print(f"Tool indexing:       " f"{startup_timings['index']:.2f} s")

    print(f"Total startup:       " f"{startup_timings['total']:.2f} s")

    print(f"Tool universe:       " f"{len(tools)}")

    print(f"Warm mean latency:   " f"{mean(all_timings):.2f} ms")

    print(f"Warm median latency: " f"{median(all_timings):.2f} ms")

    print(f"Warm min latency:    " f"{min(all_timings):.2f} ms")

    print(f"Warm max latency:    " f"{max(all_timings):.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())
