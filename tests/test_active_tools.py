from octopus.active_tools import (
    ACTIVE_TOOL_TTL,
    update_active_tools,
)


def test_previous_tools_remain_active_when_no_new_tools_are_retrieved():
    active_tools = {}

    active_tools = update_active_tools(
        active_tools=active_tools,
        retrieved_tools=[
            "tool_create_leave",
            "tool_submit_leave",
        ],
        turn=1,
    )

    active_tools = update_active_tools(
        active_tools=active_tools,
        retrieved_tools=[],
        turn=2,
    )

    assert list(active_tools) == [
        "tool_create_leave",
        "tool_submit_leave",
    ]


def test_new_tools_are_added_without_duplicating_existing_tools():
    active_tools = {
        "tool_create_leave": 1,
        "tool_submit_leave": 1,
    }

    active_tools = update_active_tools(
        active_tools=active_tools,
        retrieved_tools=[
            "tool_submit_leave",
            "tool_send_email",
        ],
        turn=2,
    )

    assert list(active_tools) == [
        "tool_create_leave",
        "tool_submit_leave",
        "tool_send_email",
    ]


def test_tool_remains_active_through_ttl_boundary():
    active_tools = update_active_tools(
        active_tools={},
        retrieved_tools=["tool_get_leave_balance"],
        turn=1,
    )

    boundary_turn = 1 + ACTIVE_TOOL_TTL

    active_tools = update_active_tools(
        active_tools=active_tools,
        retrieved_tools=[],
        turn=boundary_turn,
    )

    assert "tool_get_leave_balance" in active_tools


def test_tool_expires_after_ttl_boundary():
    active_tools = update_active_tools(
        active_tools={},
        retrieved_tools=["tool_get_leave_balance"],
        turn=1,
    )

    expired_turn = 1 + ACTIVE_TOOL_TTL + 1

    active_tools = update_active_tools(
        active_tools=active_tools,
        retrieved_tools=[],
        turn=expired_turn,
    )

    assert "tool_get_leave_balance" not in active_tools


def test_retrieving_active_tool_refreshes_last_retrieved_turn():
    active_tools = update_active_tools(
        active_tools={},
        retrieved_tools=["tool_get_leave_balance"],
        turn=1,
    )

    active_tools = update_active_tools(
        active_tools=active_tools,
        retrieved_tools=["tool_get_leave_balance"],
        turn=8,
    )

    assert active_tools["tool_get_leave_balance"] == 8

    active_tools = update_active_tools(
        active_tools=active_tools,
        retrieved_tools=[],
        turn=10,
    )

    assert "tool_get_leave_balance" in active_tools


def test_retrieving_tool_after_ttl_would_refresh_it():
    active_tools = {
        "tool_get_leave_balance": 1,
    }

    expired_turn = 1 + ACTIVE_TOOL_TTL + 1

    active_tools = update_active_tools(
        active_tools=active_tools,
        retrieved_tools=["tool_get_leave_balance"],
        turn=expired_turn,
    )

    assert active_tools["tool_get_leave_balance"] == expired_turn


def test_active_tools_evicts_oldest_when_context_exceeds_cap():
    active_tools = {f"tool_{index}": 10 for index in range(1, 31)}

    # Give tool_1 the oldest timestamp while keeping it
    # safely inside the TTL window.
    active_tools["tool_1"] = 9

    updated = update_active_tools(
        active_tools=active_tools,
        retrieved_tools=["tool_31"],
        turn=11,
    )

    assert len(updated) == 30
    assert "tool_1" not in updated
    assert "tool_31" in updated


def test_active_tools_evicts_multiple_oldest_when_batch_exceeds_cap():
    active_tools = {f"tool_{index}": 10 for index in range(1, 30)}

    # Give three existing tools progressively older timestamps.
    # All are still safely inside the TTL window.
    active_tools["tool_1"] = 7
    active_tools["tool_2"] = 8
    active_tools["tool_3"] = 9

    retrieved_tools = [
        "tool_30",
        "tool_31",
        "tool_32",
        "tool_33",
    ]

    updated = update_active_tools(
        active_tools=active_tools,
        retrieved_tools=retrieved_tools,
        turn=11,
    )

    # 29 existing + 4 new = 33.
    # The cap should reduce this back to 30.
    assert len(updated) == 30

    # The three least recently retrieved tools
    # should be evicted.
    assert "tool_1" not in updated
    assert "tool_2" not in updated
    assert "tool_3" not in updated

    # Every tool retrieved on the current turn
    # should survive.
    for tool_name in retrieved_tools:
        assert tool_name in updated
