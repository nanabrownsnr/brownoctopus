import asyncio
import time

from octopus import Octopus


async def main():
    octopus = Octopus(
        index_path="data/indexes/default",
        catalog_path="data/mcps.json",
    )

    print("Starting Octopus from persisted index...")

    start = time.perf_counter()

    tools = await octopus.initialize()

    elapsed = time.perf_counter() - start

    print()
    print("Octopus initialized.")
    print(f"Tools loaded: {len(tools)}")
    print(f"Startup time: {elapsed:.2f}s")


asyncio.run(main())
