import asyncio
import time

from octopus import Octopus


async def main():
    octopus = Octopus(
        index_path="data/indexes/default",
        catalog_path="data/mcps.json",
    )

    print("Building Octopus index...")

    start = time.perf_counter()

    tools = await octopus.update()

    elapsed = time.perf_counter() - start

    print()
    print("Index build complete.")
    print(f"Tools indexed: {len(tools)}")
    print(f"Build time: {elapsed:.2f}s")
    print("Index path: data/indexes/default")


asyncio.run(main())
