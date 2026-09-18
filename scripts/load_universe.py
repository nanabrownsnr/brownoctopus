import asyncio

from octopus.bootstrap import initialize


async def main() -> None:
    tools = await initialize()

    print(f"\nBrown Octopus ready: {len(tools)} tools")

    for tool in tools:
        print(f"- {tool['name']}")


if __name__ == "__main__":
    asyncio.run(main())
