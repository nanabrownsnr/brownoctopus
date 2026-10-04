"""Small LangGraph integration example.

This file is intentionally outside the Octopus package. In a real application,
replace ``make_demo_tool`` with adapters that execute the corresponding MCP or
application capability.
"""

import asyncio
import json
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from brown_octopus import Octopus
from brown_octopus.tool_registry import capability_id


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def make_demo_tool(definition: dict) -> StructuredTool:
    """Create a local demo executor for one exposed Octopus capability."""
    identity = capability_id(definition)
    name = definition["name"]
    description = definition.get("description") or f"Execute capability {identity}."

    def execute(payload: str = "") -> str:
        # Replace this body with the real application/MCP call. Octopus never
        # executes capabilities itself; this adapter belongs to the harness.
        return json.dumps({
            "capability_id": identity,
            "tool_name": name,
            "payload": payload,
            "status": "demo execution",
        })

    return StructuredTool.from_function(
        execute,
        name=name,
        description=description,
    )


def build_turn_graph(model, exposed_tools: list[dict]):
    """Bind only this turn's Octopus context to a LangGraph agent."""
    tools = [make_demo_tool(definition) for definition in exposed_tools]
    bound_model = model.bind_tools(tools)

    def call_model(state: AgentState):
        return {"messages": [bound_model.invoke(state["messages"])]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(), tools


async def run_turn(octopus: Octopus, model, user_message: str, history=None):
    history = list(history or [])
    result = octopus.process(user_message)
    exposed = result["tools"]
    exposed_ids = [capability_id(tool) for tool in exposed]
    print(json.dumps({
        "query": user_message,
        "exposed_capabilities": exposed_ids,
        "exposed_count": len(exposed_ids),
        "active_state": result["active_state"],
    }))

    graph, _ = build_turn_graph(model, exposed)
    state = await graph.ainvoke({
        "messages": history + [HumanMessage(content=user_message)]
    })
    return state["messages"], result


async def main():
    octopus = Octopus(
        index_path=os.getenv("OCTOPUS_INDEX_PATH", "data/indexes/default")
    )
    await octopus.initialize()
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    history, _ = await run_turn(octopus, model, "Find the latest Nvidia news")
    history, _ = await run_turn(
        octopus,
        model,
        "Find the latest Nvidia news and email a summary to Tom",
        history,
    )
    await run_turn(octopus, model, "Email it to Tom", history)


if __name__ == "__main__":
    asyncio.run(main())
