import asyncio

from octopus import Octopus


async def main():
    octopus = Octopus()

    print("Initializing Octopus...")
    tools = await octopus.initialize()

    print(f"\nLoaded {len(tools)} tools")

    query = "Find the latest Nvidia news " "and email a summary to Tom"

    print(f"\nQuery: {query}")

    retrieved = octopus.retrieve(query)

    print("\nRetrieved tools:")
    for tool in retrieved:
        print(f"- {tool['name']}")

    result = octopus.process(query)

    print("\nActive context:")
    for tool in result["tools"]:
        print(f"- {tool['name']}")

    print(f"\nTurn: {octopus.turn}")
    print(f"Active tools: {len(octopus.active_tools)}")


if __name__ == "__main__":
    asyncio.run(main())
