import pytest

from octopus.pipeline import process_turn
from octopus.retriever import (
    initialize_retriever,
    refresh_index,
)
from octopus.analyzer import initialize_analyzer
from octopus.tool_registry import (
    get_all_tools,
    set_tools,
)


TEST_TOOLS = [
    {
        "name": "carrot_check_leave_balance",
        "description": "Check an employee's remaining leave balance and available leave days.",
        "mcp_name": "Carrot",
        "mcp_url": "https://example.com/carrot",
        "tool_name": "check_leave_balance",
        "input_schema": {},
    },
    {
        "name": "outlook_send_email",
        "description": "Send an email message to a recipient.",
        "mcp_name": "Outlook",
        "mcp_url": "https://example.com/outlook",
        "tool_name": "send_email",
        "input_schema": {},
    },
    {
        "name": "web_search",
        "description": "Search the web for current information and news.",
        "mcp_name": "Web Search",
        "mcp_url": "https://example.com/web-search",
        "tool_name": "web_search",
        "input_schema": {},
    },
    {
        "name": "calendar_create_event",
        "description": "Create a calendar event or meeting.",
        "mcp_name": "Calendar",
        "mcp_url": "https://example.com/calendar",
        "tool_name": "create_event",
        "input_schema": {},
    },
    {
        "name": "word_create_document",
        "description": "Create a Microsoft Word document.",
        "mcp_name": "Word",
        "mcp_url": "https://example.com/word",
        "tool_name": "create_document",
        "input_schema": {},
    },
    {
        "name": "github_create_issue",
        "description": "Create an issue in a GitHub repository.",
        "mcp_name": "GitHub",
        "mcp_url": "https://example.com/github",
        "tool_name": "create_issue",
        "input_schema": {},
    },
    {
        "name": "sharepoint_upload_file",
        "description": "Upload a file to SharePoint.",
        "mcp_name": "SharePoint",
        "mcp_url": "https://example.com/sharepoint",
        "tool_name": "upload_file",
        "input_schema": {},
    },
    {
        "name": "database_execute_query",
        "description": "Execute a read-only database query.",
        "mcp_name": "Database",
        "mcp_url": "https://example.com/database",
        "tool_name": "execute_query",
        "input_schema": {},
    },
]


@pytest.fixture(scope="module", autouse=True)
def setup_retriever():
    original_tools = get_all_tools()

    initialize_analyzer()
    initialize_retriever()

    set_tools(TEST_TOOLS)
    refresh_index()

    yield

    set_tools(original_tools)

    if original_tools:
        refresh_index()


def tool_names(result: dict) -> set[str]:
    return {tool["name"] for tool in result["tools"]}


def test_conversation_preserves_capabilities_across_turns():
    active_state = {}

    # Turn 1: conversation begins with a leave task.
    result = process_turn(
        query="check my leave balance",
        active_tools=active_state,
        turn=1,
    )

    active_state = result["active_state"]

    assert "carrot_check_leave_balance" in tool_names(result)

    # Turn 2: the conversation moves to email.
    result = process_turn(
        query="email it to Tom",
        active_tools=active_state,
        turn=2,
    )

    active_state = result["active_state"]

    names = tool_names(result)

    assert "outlook_send_email" in names
    assert "carrot_check_leave_balance" in names

    # Turn 3: the user modifies the existing email task.
    result = process_turn(
        query="actually send it to Sarah instead",
        active_tools=active_state,
        turn=3,
    )

    active_state = result["active_state"]

    names = tool_names(result)

    assert "outlook_send_email" in names
    assert "carrot_check_leave_balance" in names

    # Turn 4: the conversation introduces another capability.
    result = process_turn(
        query="find the latest Nvidia news",
        active_tools=active_state,
        turn=4,
    )

    names = tool_names(result)

    assert "web_search" in names
    assert "outlook_send_email" in names
    assert "carrot_check_leave_balance" in names


def test_conversation_respects_ttl_over_24_turns():
    active_state = {}

    conversation = [
        # Turn 1
        "check my leave balance",
        # Turns 2-9
        "find the latest Nvidia news",
        "search the web for AI news",
        "find recent Microsoft news",
        "search for the latest OpenAI news",
        "find today's technology news",
        "search for recent cloud computing news",
        "find the latest cybersecurity news",
        "search the web for Nvidia updates",
        # Turns 10-17
        "email the Nvidia update to Tom",
        "send the AI news to Sarah",
        "email the Microsoft update to Tom",
        "send the OpenAI news to Sarah",
        "email the technology update to Tom",
        "send the cloud news to Sarah",
        "email the cybersecurity update to Tom",
        "send the Nvidia news to Sarah",
        # Turns 18-24
        "find the latest AMD news",
        "search for recent Google news",
        "find today's semiconductor news",
        "search for recent robotics news",
        "find the latest Intel news",
        "search for recent Apple news",
        "find today's enterprise AI news",
    ]

    assert len(conversation) == 24

    tracked_tool = "carrot_check_leave_balance"
    last_retrieved_turn = None

    for turn, query in enumerate(
        conversation,
        start=1,
    ):
        result = process_turn(
            query=query,
            active_tools=active_state,
            turn=turn,
        )

        retrieved_names = {tool["name"] for tool in result["retrieved_tools"]}

        if tracked_tool in retrieved_names:
            last_retrieved_turn = turn

        active_state = result["active_state"]
        names = tool_names(result)

        print(
            f"\nTurn {turn}: {query}"
            f"\n  Retrieved: {sorted(retrieved_names)}"
            f"\n  Leave last retrieved: {last_retrieved_turn}"
            f"\n  Leave active: {tracked_tool in names}"
        )

        if last_retrieved_turn is None:
            assert tracked_tool not in names
            continue

        age = turn - last_retrieved_turn

        if age <= 8:
            assert tracked_tool in names
        else:
            assert tracked_tool not in names


def test_conversation_context_grows_to_cap_and_rotates_old_capabilities(
    monkeypatch,
):
    tools = [
        {
            "name": f"tool_{index}",
            "description": f"Capability {index}",
        }
        for index in range(1, 35)
    ]

    set_tools(tools)

    retrieval_by_turn = {
        "turn_1": [
            tool
            for tool in tools
            if tool["name"] in {f"tool_{index}" for index in range(1, 11)}
        ],
        "turn_2": [
            tool
            for tool in tools
            if tool["name"] in {f"tool_{index}" for index in range(11, 21)}
        ],
        "turn_3": [
            tool
            for tool in tools
            if tool["name"] in {f"tool_{index}" for index in range(21, 31)}
        ],
        "turn_4": [
            tool
            for tool in tools
            if tool["name"] in {f"tool_{index}" for index in range(31, 35)}
        ],
    }

    monkeypatch.setattr(
        "octopus.pipeline.retrieve_tools",
        lambda query: retrieval_by_turn[query],
    )

    active_state = {}

    result = process_turn(
        query="turn_1",
        active_tools=active_state,
        turn=1,
    )
    active_state = result["active_state"]

    assert len(result["tools"]) == 10

    result = process_turn(
        query="turn_2",
        active_tools=active_state,
        turn=2,
    )
    active_state = result["active_state"]

    assert len(result["tools"]) == 20

    result = process_turn(
        query="turn_3",
        active_tools=active_state,
        turn=3,
    )
    active_state = result["active_state"]

    assert len(result["tools"]) == 30

    result = process_turn(
        query="turn_4",
        active_tools=active_state,
        turn=4,
    )

    active_names = set(result["active_state"])
    context_names = {tool["name"] for tool in result["tools"]}

    assert len(active_names) == 30
    assert len(context_names) == 30

    # The newest capabilities must enter.
    for index in range(31, 35):
        tool_name = f"tool_{index}"

        assert tool_name in active_names
        assert tool_name in context_names

    # Four capabilities from the oldest retrieval batch
    # must have been displaced.
    oldest_batch = {f"tool_{index}" for index in range(1, 11)}

    assert len(oldest_batch - active_names) == 4

    # More recent capabilities must remain.
    for index in range(11, 31):
        tool_name = f"tool_{index}"

        assert tool_name in active_names
        assert tool_name in context_names
