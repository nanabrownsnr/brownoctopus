import pytest

from octopus.analyzer import initialize_analyzer
from octopus.retriever import (
    initialize_retriever,
    refresh_index,
    retrieve_tools,
)
from octopus.tool_registry import (
    get_all_tools,
    set_tools,
)
import torch

import octopus.retriever as retriever


def test_refresh_index_returns_built_embeddings(
    monkeypatch,
):
    tools = [
        {
            "name": "web_search",
            "description": "Search the web",
        },
        {
            "name": "send_email",
            "description": "Send an email",
        },
    ]

    expected_embeddings = object()

    class FakeModel:
        def encode(
            self,
            texts,
            convert_to_tensor,
            normalize_embeddings,
        ):
            assert texts == [
                "web_search: Search the web",
                "send_email: Send an email",
            ]
            assert convert_to_tensor is True
            assert normalize_embeddings is True

            return expected_embeddings

    monkeypatch.setattr(
        retriever,
        "model",
        FakeModel(),
    )

    monkeypatch.setattr(
        retriever,
        "get_all_tools",
        lambda: tools,
    )

    monkeypatch.setattr(
        retriever,
        "TOOLS",
        [],
    )

    monkeypatch.setattr(
        retriever,
        "tool_embeddings",
        None,
    )

    result = retriever.refresh_index()

    assert result is expected_embeddings
    assert retriever.tool_embeddings is expected_embeddings


def test_load_index_uses_precomputed_embeddings(
    monkeypatch,
):
    tools = [
        {
            "name": "web_search",
            "description": "Search the web",
        },
        {
            "name": "send_email",
            "description": "Send an email",
        },
    ]

    embeddings = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )

    monkeypatch.setattr(
        retriever,
        "TOOLS",
        [],
    )

    monkeypatch.setattr(
        retriever,
        "tool_embeddings",
        None,
    )

    retriever.load_index(
        tools=tools,
        embeddings=embeddings,
    )

    assert retriever.TOOLS == tools
    assert retriever.tool_embeddings is embeddings


TEST_TOOLS = [
    {
        "name": "outlook_mail_management_send_email",
        "description": (
            "Send a new email message to one or more recipients " "using Outlook."
        ),
        "input_schema": {},
    },
    {
        "name": "outlook_mail_management_reply_to_email",
        "description": ("Reply to an existing email message in Outlook."),
        "input_schema": {},
    },
    {
        "name": "outlook_calendar_list_events",
        "description": (
            "List meetings, appointments, and events from " "an Outlook calendar."
        ),
        "input_schema": {},
    },
    {
        "name": "web_search_tools_web_search",
        "description": (
            "Search the public web for current information, "
            "news, and external information."
        ),
        "input_schema": {},
    },
    {
        "name": "project_management_tools_get_sprint",
        "description": (
            "Retrieve information about the current or " "specified project sprint."
        ),
        "input_schema": {},
    },
    {
        "name": "project_management_tools_list_work_items",
        "description": (
            "List project work items, tasks, user stories, " "bugs, or tickets."
        ),
        "input_schema": {},
    },
    {
        "name": "office_word_mcp_server_api_create_word_document",
        "description": ("Create a new Microsoft Word document."),
        "input_schema": {},
    },
    {
        "name": "office_word_mcp_server_api_add_paragraph",
        "description": ("Add a paragraph of text to a Microsoft Word document."),
        "input_schema": {},
    },
    {
        "name": "carrot_mcp_server_api_list_leave_application",
        "description": ("List employee leave applications and leave information."),
        "input_schema": {},
    },
    {
        "name": "carrot_mcp_server_api_get_hr_dashboard_summary",
        "description": (
            "Get an employee HR dashboard summary including "
            "leave and workforce information."
        ),
        "input_schema": {},
    },
]


@pytest.fixture(scope="module", autouse=True)
def initialized_retriever():
    initialize_analyzer()
    initialize_retriever()

    original_tools = get_all_tools()

    set_tools(TEST_TOOLS)
    refresh_index()

    yield

    set_tools(original_tools)
    refresh_index()


def tool_names(query: str) -> list[str]:
    return [tool["name"] for tool in retrieve_tools(query)]


def test_send_email_capability_is_retrieved():
    names = tool_names("send an email to Tom")

    assert any("send_email" in name for name in names)


def test_web_search_capability_is_retrieved():
    names = tool_names("find the latest Nvidia news")

    assert any("web_search" in name for name in names)


def test_calendar_and_reply_capabilities_are_retrieved():
    names = tool_names("check my calendar and reply to Tom")

    assert any("calendar" in name for name in names)

    assert any("reply" in name for name in names)


def test_leave_and_email_capabilities_are_retrieved():
    names = tool_names("check my leave balance and email my manager")

    assert any("leave" in name or "hr_" in name for name in names)

    assert any("email" in name for name in names)


def test_web_and_email_capabilities_survive_compound_request():
    names = tool_names(
        "search the web for the latest Nvidia news " "and send what you find to Tom"
    )

    assert any("web_search" in name for name in names)

    assert any("email" in name for name in names)


def test_word_and_sprint_capabilities_survive_nested_request():
    names = tool_names("create a Word document summarizing the sprint")

    assert any("word" in name for name in names)

    assert any("sprint" in name for name in names)


def test_work_items_and_word_capabilities_survive_compound_request():
    names = tool_names(
        "get the sprint work items and " "create a Word document summarizing them"
    )

    assert any("work_item" in name for name in names)

    assert any("word" in name for name in names)


def test_three_capabilities_survive_three_action_request():
    names = tool_names(
        "check my leave balance, "
        "search the web for Nvidia news, "
        "and email the results to Tom"
    )

    assert any("leave" in name or "hr_" in name for name in names)

    assert any("web_search" in name for name in names)

    assert any("email" in name for name in names)


def test_results_are_deduplicated():
    names = tool_names("send an email to Tom and reply to Sarah")

    assert len(names) == len(set(names))


def test_empty_query_returns_no_tools():
    assert retrieve_tools("") == []
    assert retrieve_tools("   ") == []


def test_retriever_uses_current_registry():
    original_tools = get_all_tools()

    try:
        tools = [
            {
                "name": "custom_search_current_information",
                "description": (
                    "Search the internet for current information " "and recent news."
                ),
                "input_schema": {},
            }
        ]

        set_tools(tools)
        refresh_index()

        names = tool_names("search the internet for the latest AI news")

        assert "custom_search_current_information" in names

    finally:
        set_tools(original_tools)
        refresh_index()


def test_retrieval_uses_bounded_max_gap(
    monkeypatch,
):
    ranked_tools = [
        {"name": "tool_1", "score": 0.90},
        {"name": "tool_2", "score": 0.89},
        {"name": "tool_3", "score": 0.88},
        {"name": "tool_4", "score": 0.70},
        {"name": "tool_5", "score": 0.69},
    ]

    monkeypatch.setattr(
        "octopus.retriever.rank_tools",
        lambda query: ranked_tools,
    )

    from octopus.retriever import retrieve_tools

    selected = retrieve_tools("test query")

    assert [tool["name"] for tool in selected] == [
        "tool_1",
        "tool_2",
        "tool_3",
    ]


def test_retrieval_selects_per_intent_then_merges(
    monkeypatch,
):
    intents = [
        {
            "text": "Search for Nvidia news",
        },
        {
            "text": "email a summary to Tom",
        },
    ]

    monkeypatch.setattr(
        "octopus.retriever.analyze_intents",
        lambda query: intents,
    )

    rankings = {
        "Search for Nvidia news": [
            {"name": "github_search", "score": 0.90},
            {"name": "web_search", "score": 0.89},
            {"name": "railway_info", "score": 0.70},
        ],
        "email a summary to Tom": [
            {"name": "send_email", "score": 0.90},
            {"name": "find_email", "score": 0.89},
            {"name": "search_mail", "score": 0.70},
        ],
    }

    monkeypatch.setattr(
        "octopus.retriever.rank_tools",
        lambda query: rankings[query],
    )

    from octopus.retriever import retrieve_tools

    selected = retrieve_tools(
        "Find the latest Nvidia news " "and email a summary to Tom"
    )

    assert [tool["name"] for tool in selected] == [
        "github_search",
        "web_search",
        "send_email",
        "find_email",
    ]


def test_retrieval_deduplicates_tools_across_intents(
    monkeypatch,
):
    intents = [
        {
            "text": "Find Nvidia news",
        },
        {
            "text": "Research Nvidia",
        },
    ]

    monkeypatch.setattr(
        "octopus.retriever.analyze_intents",
        lambda query: intents,
    )

    rankings = {
        "Find Nvidia news": [
            {"name": "web_search", "score": 0.90},
            {"name": "github_search", "score": 0.89},
            {"name": "unrelated_1", "score": 0.60},
        ],
        "Research Nvidia": [
            {"name": "web_search", "score": 0.91},
            {"name": "company_search", "score": 0.90},
            {"name": "unrelated_2", "score": 0.60},
        ],
    }

    monkeypatch.setattr(
        "octopus.retriever.rank_tools",
        lambda query: rankings[query],
    )

    from octopus.retriever import retrieve_tools

    selected = retrieve_tools("Find and research Nvidia news")

    assert [tool["name"] for tool in selected] == [
        "web_search",
        "github_search",
        "company_search",
    ]
