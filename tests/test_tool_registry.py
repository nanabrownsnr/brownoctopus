import pytest

from octopus.tool_registry import (
    get_all_tools,
    get_tools,
    set_tools,
)


@pytest.fixture(autouse=True)
def clean_registry():
    original_tools = get_all_tools()

    set_tools([])

    yield

    set_tools(original_tools)


def test_registry_starts_empty_when_cleared():
    set_tools([])

    assert get_all_tools() == []


def test_set_tools_replaces_registry_contents():
    tools = [
        {
            "name": "tool_send_email",
            "description": "Send an email message.",
            "input_schema": {},
        },
        {
            "name": "tool_search_web",
            "description": "Search the public web.",
            "input_schema": {},
        },
    ]

    set_tools(tools)

    assert get_all_tools() == tools


def test_get_tools_returns_requested_definitions():
    tools = [
        {
            "name": "tool_send_email",
            "description": "Send an email message.",
            "input_schema": {},
        },
        {
            "name": "tool_search_web",
            "description": "Search the public web.",
            "input_schema": {},
        },
    ]

    set_tools(tools)

    result = get_tools(
        [
            "tool_search_web",
            "tool_send_email",
        ]
    )

    assert [tool["name"] for tool in result] == [
        "tool_search_web",
        "tool_send_email",
    ]


def test_get_tools_ignores_unknown_names():
    set_tools(
        [
            {
                "name": "tool_send_email",
                "description": "Send an email message.",
                "input_schema": {},
            }
        ]
    )

    result = get_tools(
        [
            "tool_that_does_not_exist",
            "tool_send_email",
        ]
    )

    assert [tool["name"] for tool in result] == [
        "tool_send_email",
    ]


def test_set_tools_copies_input_list():
    tools = [
        {
            "name": "tool_send_email",
            "description": "Send an email message.",
            "input_schema": {},
        }
    ]

    set_tools(tools)

    tools.append(
        {
            "name": "tool_search_web",
            "description": "Search the public web.",
            "input_schema": {},
        }
    )

    assert [tool["name"] for tool in get_all_tools()] == [
        "tool_send_email",
    ]


def test_get_all_tools_returns_copy():
    set_tools(
        [
            {
                "name": "tool_send_email",
                "description": "Send an email message.",
                "input_schema": {},
            }
        ]
    )

    returned_tools = get_all_tools()
    returned_tools.clear()

    assert len(get_all_tools()) == 1
