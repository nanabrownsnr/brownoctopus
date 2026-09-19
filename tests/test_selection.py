from octopus.selection import select_bounded_max_gap


def test_selects_tools_before_largest_gap():
    ranked_tools = [
        {"name": "tool_1", "score": 0.90},
        {"name": "tool_2", "score": 0.89},
        {"name": "tool_3", "score": 0.88},
        {"name": "tool_4", "score": 0.70},
        {"name": "tool_5", "score": 0.69},
    ]

    selected = select_bounded_max_gap(ranked_tools)

    assert [tool["name"] for tool in selected] == [
        "tool_1",
        "tool_2",
        "tool_3",
    ]


def test_returns_full_cap_when_no_meaningful_gap():
    ranked_tools = [
        {"name": "tool_1", "score": 0.90},
        {"name": "tool_2", "score": 0.895},
        {"name": "tool_3", "score": 0.890},
        {"name": "tool_4", "score": 0.885},
        {"name": "tool_5", "score": 0.880},
        {"name": "tool_6", "score": 0.875},
    ]

    selected = select_bounded_max_gap(
        ranked_tools,
        max_tools=4,
        min_gap_percent=2.0,
    )

    assert [tool["name"] for tool in selected] == [
        "tool_1",
        "tool_2",
        "tool_3",
        "tool_4",
    ]


def test_detects_gap_after_max_tool():
    ranked_tools = [
        {
            "name": f"tool_{index}",
            "score": 1.00 - (index * 0.005),
        }
        for index in range(1, 17)
    ]

    # Tool 17 has a large drop from tool 16.
    ranked_tools.append(
        {
            "name": "tool_17",
            "score": 0.70,
        }
    )

    selected = select_bounded_max_gap(
        ranked_tools,
        max_tools=16,
        min_gap_percent=2.0,
    )

    assert len(selected) == 16

    assert selected[-1]["name"] == "tool_16"


def test_ignores_large_gaps_beyond_bounded_window():
    ranked_tools = [
        {
            "name": f"tool_{index}",
            "score": 1.00 - (index * 0.005),
        }
        for index in range(1, 18)
    ]

    # Huge gap occurs AFTER the comparison boundary.
    ranked_tools.append(
        {
            "name": "tool_18",
            "score": 0.20,
        }
    )

    selected = select_bounded_max_gap(
        ranked_tools,
        max_tools=16,
        min_gap_percent=2.0,
    )

    assert len(selected) == 16

    assert selected[-1]["name"] == "tool_16"


def test_prefers_stronger_later_gap_over_earlier_gap():
    ranked_tools = [
        {"name": "list_leave", "score": 0.90},
        {"name": "update_leave", "score": 0.83},
        {"name": "create_leave", "score": 0.82},
        {"name": "submit_leave", "score": 0.81},
        {"name": "hr_dashboard", "score": 0.65},
        {"name": "get_employee", "score": 0.63},
    ]

    selected = select_bounded_max_gap(
        ranked_tools,
        max_tools=16,
        min_gap_percent=2.0,
    )

    assert [tool["name"] for tool in selected] == [
        "list_leave",
        "update_leave",
        "create_leave",
        "submit_leave",
    ]
