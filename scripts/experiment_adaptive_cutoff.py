import asyncio
import csv
from pathlib import Path
from statistics import mean, median

from octopus import Octopus
from octopus.analyzer import analyze_intents
from octopus.retriever import rank_tools


QUERIES = [
    "Send an email to Tom",
    "Reply to Tom's email",
    "Search the web for the latest Nvidia news",
    "Create a Word document",
    "Get the current sprint work items",
    "Find the latest Nvidia news and email a summary to Tom",
    "Get the current sprint work items and create a Word report",
    "Search for Nvidia news, create a report, and email it to Tom",
    "Schedule a meeting with Tom",
    "Get my leave applications",
    "Submit a leave application",
    "Find a GitHub repository",
    "Get a file from SharePoint",
    "Run a query against the database",
    "Check the deployment status on Railway",
]


OUTPUT_DIR = Path("data/experiments/adaptive_cutoff")

MAX_TOOLS = 15

RECURSIVE_THRESHOLDS = [
    6.0,
    4.0,
    2.0,
]

MIN_MAX_GAP = 2.0

V1_K = 4


def add_gap_statistics(
    ranked_tools: list[dict],
) -> list[dict]:
    """
    Add adjacent score gaps.

    Gap percentage is measured relative
    to the top score for the intent.
    """

    if not ranked_tools:
        return []

    peak_score = ranked_tools[0]["score"]

    results = []

    for index, tool in enumerate(ranked_tools):
        gap = None
        gap_percent = None

        if index + 1 < len(ranked_tools):
            next_score = ranked_tools[index + 1]["score"]

            gap = tool["score"] - next_score

            if peak_score != 0:
                gap_percent = (gap / peak_score) * 100

        results.append(
            {
                **tool,
                "gap": gap,
                "gap_percent": gap_percent,
            }
        )

    return results


def fixed_k(
    ranked_tools: list[dict],
    k: int = V1_K,
) -> dict:
    selected_k = min(
        k,
        len(ranked_tools),
    )

    return {
        "k": selected_k,
        "reason": f"fixed-{k}",
        "boundary": (ranked_tools[selected_k - 1] if selected_k > 0 else None),
    }


def recursive_first_gap(
    ranked_tools: list[dict],
    max_tools: int = MAX_TOOLS,
    thresholds: list[float] = (RECURSIVE_THRESHOLDS),
) -> dict:
    """
    User hypothesis.

    Search the bounded head repeatedly:

        first >= 6%
        else first >= 4%
        else first >= 2%
        else max_tools

    Important:
    this deliberately uses FIRST qualifying
    gap because otherwise the threshold ladder
    collapses into maximum-gap selection.
    """

    head = ranked_tools[:max_tools]

    for threshold in thresholds:
        for tool in head:
            gap_percent = tool.get("gap_percent")

            if gap_percent is None:
                continue

            if gap_percent >= threshold:
                return {
                    "k": tool["rank"],
                    "reason": (f"first >= " f"{threshold:.0f}%"),
                    "boundary": tool,
                }

    selected_k = min(
        max_tools,
        len(ranked_tools),
    )

    return {
        "k": selected_k,
        "reason": "fallback-cap",
        "boundary": (ranked_tools[selected_k - 1] if selected_k > 0 else None),
    }


def bounded_max_gap(
    ranked_tools: list[dict],
    max_tools: int = MAX_TOOLS,
    minimum_gap: float = MIN_MAX_GAP,
) -> dict:
    """
    Occam hypothesis.

    Within the bounded head:

        find the single largest adjacent gap

    If that gap is at least minimum_gap,
    cut there.

    Otherwise return the full bounded head.
    """

    head = [
        tool for tool in ranked_tools[:max_tools] if tool.get("gap_percent") is not None
    ]

    if not head:
        return {
            "k": 0,
            "reason": "empty",
            "boundary": None,
        }

    strongest = max(
        head,
        key=lambda tool: (tool["gap_percent"]),
    )

    if strongest["gap_percent"] >= minimum_gap:
        return {
            "k": strongest["rank"],
            "reason": "max-gap",
            "boundary": strongest,
        }

    selected_k = min(
        max_tools,
        len(ranked_tools),
    )

    return {
        "k": selected_k,
        "reason": "fallback-cap",
        "boundary": (ranked_tools[selected_k - 1] if selected_k > 0 else None),
    }


def reduction_percent(
    selected_k: int,
    universe_size: int,
) -> float:
    if universe_size == 0:
        return 0.0

    return (1 - selected_k / universe_size) * 100


def boundary_gap(
    result: dict,
) -> float | None:
    boundary = result["boundary"]

    if boundary is None:
        return None

    return boundary.get("gap_percent")


def boundary_tool(
    result: dict,
) -> str:
    boundary = result["boundary"]

    if boundary is None:
        return "-"

    return boundary["name"]


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def print_policy_statistics(
    name: str,
    rows: list[dict],
    k_field: str,
    universe_size: int,
) -> None:
    values = [row[k_field] for row in rows]

    average_k = mean(values)
    median_k = median(values)

    average_reduction = mean(
        reduction_percent(
            value,
            universe_size,
        )
        for value in values
    )

    cap_count = sum(value == MAX_TOOLS for value in values)

    print(
        f"{name:<24}"
        f"{average_k:<12.2f}"
        f"{median_k:<12.1f}"
        f"{average_reduction:<14.2f}%"
        f"{cap_count}"
    )


async def main():
    octopus = Octopus()

    print("Loading persisted Octopus V1...")

    tools = await octopus.initialize()

    universe_size = len(tools)

    print(f"Loaded {universe_size} tools.")

    print(f"Maximum tools per intent: " f"{MAX_TOOLS}")

    print("Policies:")

    print("  1. V1 fixed K=4")

    print("  2. Recursive first gap " "6% -> 4% -> 2% -> cap")

    print("  3. Bounded maximum gap " ">=2% -> otherwise cap")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []
    ranking_rows = []

    total_intents = 0

    for query_number, query in enumerate(
        QUERIES,
        start=1,
    ):
        intents = analyze_intents(query)

        for intent_number, intent in enumerate(
            intents,
            start=1,
        ):
            total_intents += 1

            intent_text = intent["text"]

            ranked = rank_tools(intent_text)

            ranked = add_gap_statistics(ranked)

            fixed = fixed_k(ranked)

            recursive = recursive_first_gap(ranked)

            max_gap = bounded_max_gap(ranked)

            recursive_gap = boundary_gap(recursive)

            max_gap_value = boundary_gap(max_gap)

            summary_rows.append(
                {
                    "query_number": (query_number),
                    "query": query,
                    "intent_number": (intent_number),
                    "intent": (intent_text),
                    "peak_score": (ranked[0]["score"]),
                    "fixed_k": (fixed["k"]),
                    "recursive_k": (recursive["k"]),
                    "recursive_reason": (recursive["reason"]),
                    "recursive_gap_percent": (recursive_gap),
                    "recursive_boundary_tool": (boundary_tool(recursive)),
                    "max_gap_k": (max_gap["k"]),
                    "max_gap_reason": (max_gap["reason"]),
                    "max_gap_percent": (max_gap_value),
                    "max_gap_boundary_tool": (boundary_tool(max_gap)),
                    "policies_agree": (recursive["k"] == max_gap["k"]),
                }
            )

            for tool in ranked:
                ranking_rows.append(
                    {
                        "query_number": (query_number),
                        "query": query,
                        "intent_number": (intent_number),
                        "intent": (intent_text),
                        "rank": (tool["rank"]),
                        "tool_name": (tool["name"]),
                        "score": (tool["score"]),
                        "gap": (tool["gap"]),
                        "gap_percent": (tool["gap_percent"]),
                    }
                )

    summary_path = OUTPUT_DIR / "simple_policy_comparison.csv"

    ranking_path = OUTPUT_DIR / "simple_policy_rankings.csv"

    write_csv(
        summary_path,
        summary_rows,
    )

    write_csv(
        ranking_path,
        ranking_rows,
    )

    # -------------------------------------------------
    # EVERYTHING IMPORTANT PRINTS AT THE END
    # -------------------------------------------------

    print()
    print("=" * 150)
    print("FINAL POLICY COMPARISON")
    print("=" * 150)

    print(
        f"{'Intent':<38}"
        f"{'Peak':<8}"
        f"{'V1':<6}"
        f"{'Rec':<6}"
        f"{'Rec rule':<14}"
        f"{'Max':<6}"
        f"{'Max gap':<10}"
        f"{'Same':<7}"
        f"Max-gap boundary"
    )

    print("-" * 150)

    for row in summary_rows:
        max_gap_value = row["max_gap_percent"]

        max_gap_text = f"{max_gap_value:.2f}%" if max_gap_value is not None else "-"

        print(
            f"{row['intent'][:36]:<38}"
            f"{row['peak_score']:<8.4f}"
            f"{row['fixed_k']:<6}"
            f"{row['recursive_k']:<6}"
            f"{row['recursive_reason']:<14}"
            f"{row['max_gap_k']:<6}"
            f"{max_gap_text:<10}"
            f"{str(row['policies_agree']):<7}"
            f"{row['max_gap_boundary_tool']}"
        )

    print()
    print("=" * 150)
    print("POLICY AGGREGATES")
    print("=" * 150)

    print(
        f"{'Policy':<24}"
        f"{'Average K':<12}"
        f"{'Median K':<12}"
        f"{'Avg reduction':<14}"
        f"At cap"
    )

    print("-" * 80)

    print_policy_statistics(
        "V1 fixed K=4",
        summary_rows,
        "fixed_k",
        universe_size,
    )

    print_policy_statistics(
        "Recursive gap",
        summary_rows,
        "recursive_k",
        universe_size,
    )

    print_policy_statistics(
        "Bounded max gap",
        summary_rows,
        "max_gap_k",
        universe_size,
    )

    print()
    print("=" * 150)
    print("POLICY DISAGREEMENTS")
    print("=" * 150)

    disagreements = [row for row in summary_rows if not row["policies_agree"]]

    if not disagreements:
        print("Recursive and max-gap " "agree on every intent.")
    else:
        for row in disagreements:
            print()
            print(f"Intent: {row['intent']}")

            print(
                "  Recursive: "
                f"K={row['recursive_k']} "
                f"({row['recursive_reason']})"
            )

            print("  Recursive boundary: " f"{row['recursive_boundary_tool']}")

            recursive_gap = row["recursive_gap_percent"]

            if recursive_gap is not None:
                print("  Recursive gap: " f"{recursive_gap:.2f}%")

            print("  Max gap:   " f"K={row['max_gap_k']} " f"({row['max_gap_reason']})")

            print("  Max-gap boundary: " f"{row['max_gap_boundary_tool']}")

            max_gap_value = row["max_gap_percent"]

            if max_gap_value is not None:
                print("  Max gap size: " f"{max_gap_value:.2f}%")

    print()
    print("=" * 150)
    print("EXPERIMENT COMPLETE")
    print("=" * 150)

    print(f"Queries: " f"{len(QUERIES)}")

    print(f"Operational intents: " f"{total_intents}")

    print(f"Universe: " f"{universe_size}")

    print(f"Maximum tools: " f"{MAX_TOOLS}")

    print(f"Summary: " f"{summary_path}")

    print(f"Rankings: " f"{ranking_path}")


if __name__ == "__main__":
    asyncio.run(main())
