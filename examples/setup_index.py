"""Build or update a Brown Octopus capability index from an MCP catalog.

Run after installing Brown Octopus and preparing its models:

    brown-octopus setup-models
    python examples/setup_index.py

The default catalog is data/mcps.json and the default output is
data/indexes/default. Both paths can be overridden for a separate catalog or
index.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from brown_octopus import OctopusIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/mcps.json"),
        help="JSON MCP catalog to discover (default: data/mcps.json)",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/indexes/default"),
        help="Directory for the persisted index (default: data/indexes/default)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Synchronize an existing index instead of creating it initially",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    from brown_octopus.sources import LocalMcpCatalogSource

    index = OctopusIndex.from_sources(
        [LocalMcpCatalogSource(args.catalog)],
        index_path=args.index,
    )
    report = await (index.update() if args.update else index.create())

    operation = "updated" if args.update else "created"
    print(f"Index {operation}: {args.index}")
    print(f"Capabilities: {report.tool_count}")
    print(f"Added: {len(report.added)}")
    print(f"Changed: {len(report.changed)}")
    print(f"Removed: {len(report.removed)}")
    if report.failed_sources:
        print(f"Sources unavailable: {len(report.failed_sources)}")


if __name__ == "__main__":
    asyncio.run(main())
