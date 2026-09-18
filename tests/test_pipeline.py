import pytest

from octopus.analyzer import initialize_analyzer
from octopus.pipeline import process_turn
from octopus.retriever import (
    initialize_retriever,
    refresh_index,
)
from octopus.tool_registry import (
    get_all_tools,
    set_tools,
)


TEST_TOOLS = [
    {
        "name": "tool_get_leave_balance",
        "description": (
            "Get an employee's current leave or vacation balance "
            "and remaining available days."
        ),
        "input_schema": {},
    },
    {
        "name": "tool_send_email",
        "description": ("Send a new email message to one or more recipients."),
        "input_schema": {},
    },
    {
        "name": "tool_search_web",
        "description": ("Search the public web for current information and news."),
        "input_schema": {},
    },
]


@pytest.fixture(scope="module", autouse=True)
def initialized_pipeline():
    initialize_analyzer()
    initialize_retriever()

    original_tools = get_all_tools()

    set_tools(TEST_TOOLS)
    refresh_index()

    yield

    set_tools(original_tools)
    refresh_index()


def test_process_turn_returns_retrieved_tools_state_and_definitions():
    result = process_turn(
        query="check my leave balance",
        active_tools={},
        turn=1,
    )

    retrieved_names = [tool["name"] for tool in result["retrieved_tools"]]

    active_names = list(result["active_state"])

    context_names = [tool["name"] for tool in result["tools"]]

    assert "tool_get_leave_balance" in retrieved_names
    assert "tool_get_leave_balance" in active_names
    assert "tool_get_leave_balance" in context_names

    assert result["active_state"]["tool_get_leave_balance"] == 1


def test_process_turn_preserves_previous_tool_context():
    first_result = process_turn(
        query="check my leave balance",
        active_tools={},
        turn=1,
    )

    second_result = process_turn(
        query="what about next week?",
        active_tools=first_result["active_state"],
        turn=2,
    )

    assert "tool_get_leave_balance" in second_result["active_state"]

    context_names = [tool["name"] for tool in second_result["tools"]]

    assert "tool_get_leave_balance" in context_names


def test_process_turn_adds_new_capability_without_losing_previous_context():
    first_result = process_turn(
        query="check my leave balance",
        active_tools={},
        turn=1,
    )

    second_result = process_turn(
        query="send my manager an email",
        active_tools=first_result["active_state"],
        turn=2,
    )

    context_names = [tool["name"] for tool in second_result["tools"]]

    assert "tool_get_leave_balance" in context_names
    assert "tool_send_email" in context_names


def test_retrieved_tools_only_describe_current_turn_retrieval():
    first_result = process_turn(
        query="check my leave balance",
        active_tools={},
        turn=1,
    )

    second_result = process_turn(
        query="send my manager an email",
        active_tools=first_result["active_state"],
        turn=2,
    )

    retrieved_names = [tool["name"] for tool in second_result["retrieved_tools"]]

    context_names = [tool["name"] for tool in second_result["tools"]]

    assert "tool_send_email" in retrieved_names

    assert "tool_get_leave_balance" in context_names


def test_process_turn_can_add_multiple_new_capabilities():
    result = process_turn(
        query=(
            "search the web for the latest Nvidia news " "and email the results to Tom"
        ),
        active_tools={},
        turn=1,
    )

    context_names = [tool["name"] for tool in result["tools"]]

    assert "tool_search_web" in context_names
    assert "tool_send_email" in context_names


def test_pipeline_rotates_old_tools_when_active_context_is_full(monkeypatch):
    existing_tools = [
        {
            "name": f"tool_{index}",
            "description": f"Tool {index}",
        }
        for index in range(1, 31)
    ]

    new_tools = [
        {
            "name": f"tool_{index}",
            "description": f"Tool {index}",
        }
        for index in range(31, 35)
    ]

    all_tools = existing_tools + new_tools

    set_tools(all_tools)

    active_state = {f"tool_{index}": 10 for index in range(1, 31)}

    # Make these the four oldest capabilities.
    active_state["tool_1"] = 6
    active_state["tool_2"] = 7
    active_state["tool_3"] = 8
    active_state["tool_4"] = 9

    monkeypatch.setattr(
        "octopus.pipeline.retrieve_tools",
        lambda query: new_tools,
    )

    result = process_turn(
        query="use some new capabilities",
        active_tools=active_state,
        turn=11,
    )

    active_names = set(result["active_state"])
    context_names = {tool["name"] for tool in result["tools"]}

    assert len(active_names) == 30
    assert len(context_names) == 30

    for tool_name in [
        "tool_1",
        "tool_2",
        "tool_3",
        "tool_4",
    ]:
        assert tool_name not in active_names
        assert tool_name not in context_names

    for tool_name in [
        "tool_31",
        "tool_32",
        "tool_33",
        "tool_34",
    ]:
        assert tool_name in active_names
        assert tool_name in context_names
