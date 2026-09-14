TEST_CASES = [
    {
        "name": "single email action",
        "query": "reply to Tom saying I'll call tomorrow",
        "expected_intents": [
            {
                "action": "reply",
                "context": "Tom saying I'll call tomorrow",
                "expected_tools": [
                    "reply_to_email",
                ],
            }
        ],
    },
    {
        "name": "single leave action",
        "query": "book my Christmas leave",
        "expected_intents": [
            {
                "action": "book",
                "context": "my Christmas leave",
                "expected_tools": [
                    "create_leave_application",
                    "submit_leave_application",
                ],
            }
        ],
    },
    {
        "name": "leave lookup",
        "query": "check whether I already submitted Christmas leave",
        "expected_intents": [
            {
                "action": "check",
                "context": "whether I already submitted Christmas leave",
                "expected_tools": [
                    "list_leave_application",
                ],
            }
        ],
    },
    {
        "name": "compound leave and email",
        "query": (
            "check whether I submitted Christmas leave "
            "and email my manager if I haven't"
        ),
        "expected_intents": [
            {
                "action": "check",
                "context": "whether I submitted Christmas leave",
                "expected_tools": [
                    "list_leave_application",
                ],
            },
            {
                "action": "email",
                "context": "my manager if I haven't",
                "expected_tools": [
                    "send_email",
                    "compose_email",
                ],
            },
        ],
    },
]


def test_cases_are_defined():
    assert len(TEST_CASES) == 4

    for case in TEST_CASES:
        assert case["query"]
        assert case["expected_intents"]

        for intent in case["expected_intents"]:
            assert intent["action"]
            assert intent["context"]
            assert intent["expected_tools"]


from octopus.analyzer import analyze_intents


def test_single_email_action():
    query = "reply to Tom saying I'll call tomorrow"

    intents = analyze_intents(query)

    assert len(intents) == 1
    assert intents[0]["action"] == "reply"

    assert "Tom" in intents[0]["context"]
    assert "call tomorrow" in intents[0]["context"]


def test_polite_leave_action():
    query = "can you book my leave?"

    intents = analyze_intents(query)

    assert len(intents) == 1
    assert intents[0]["action"] == "book"
    assert "leave" in intents[0]["target"]


def test_compound_leave_and_email():
    query = (
        "check whether I submitted Christmas leave " "and email my manager if I haven't"
    )

    intents = analyze_intents(query)

    assert len(intents) == 2

    assert intents[0]["action"] == "check"
    assert "leave" in intents[0]["target"]

    assert intents[1]["action"] == "email"
    assert "manager" in intents[1]["target"]


def test_compound_email_and_leave_reversed():
    query = "email my manager and check whether " "I submitted Christmas leave"

    intents = analyze_intents(query)

    assert len(intents) == 2

    assert intents[0]["action"] == "email"
    assert "manager" in intents[0]["target"]

    assert intents[1]["action"] == "check"
    assert "leave" in intents[1]["target"]


def test_leave_action_with_different_phrasing():
    query = "could you submit my annual leave request?"

    intents = analyze_intents(query)

    assert len(intents) == 1
    assert intents[0]["action"] == "submit"
    assert "leave" in intents[0]["target"]


def test_compound_report_and_meeting():
    query = "send Tom the quarterly report and schedule a meeting with Sarah"

    intents = analyze_intents(query)

    assert len(intents) == 2

    assert intents[0]["action"] == "send"
    assert "report" in intents[0]["target"]

    assert intents[1]["action"] == "schedule"
    assert "meeting" in intents[1]["target"]


def test_conditional_compound_action():
    query = "check my leave balance and if I have enough days " "submit a leave request"

    intents = analyze_intents(query)

    assert len(intents) == 2

    assert intents[0]["action"] == "check"
    assert "leave" in intents[0]["target"]

    assert intents[1]["action"] == "submit"
    assert "leave" in intents[1]["target"]


def test_sequential_actions():
    query = "check my leave balance then email my manager"

    intents = analyze_intents(query)

    assert len(intents) == 2

    assert intents[0]["action"] == "check"
    assert "leave" in intents[0]["target"]

    assert intents[1]["action"] == "email"
    assert "manager" in intents[1]["target"]


def test_action_without_direct_object():
    query = "check my calendar and reply to Tom"

    intents = analyze_intents(query)

    assert len(intents) == 2

    assert intents[0]["action"] == "check"
    assert "calendar" in intents[0]["target"]

    assert intents[1]["action"] == "reply"
