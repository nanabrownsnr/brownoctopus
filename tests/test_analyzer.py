from octopus.analyzer import (
    analyze_intents,
    initialize_analyzer,
)


initialize_analyzer()


def test_single_email_action():
    query = "reply to Tom saying I'll call tomorrow"

    intents = analyze_intents(query)

    assert len(intents) == 1
    assert intents[0]["action"] == "reply"
    assert "Tom" in intents[0]["text"]
    assert "call tomorrow" in intents[0]["text"]


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
    query = "send Tom the quarterly report " "and schedule a meeting with Sarah"

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
    assert "Tom" in intents[1]["text"]


def test_nested_operational_action():
    query = "create a Word document summarizing the sprint"

    intents = analyze_intents(query)

    assert len(intents) == 2

    assert intents[0]["action"] == "create"
    assert "Word document" in intents[0]["target"]

    assert intents[1]["action"] == "summarize"
    assert "sprint" in intents[1]["target"]


def test_nested_action_is_removed_from_parent_local_text():
    query = "create a Word document summarizing the sprint"

    intents = analyze_intents(query)

    assert intents[0]["text"] == "Create a Word document"
    assert intents[1]["text"] == "summarizing the sprint"


def test_compound_actions_have_independent_local_text():
    query = "search the web for the latest Nvidia news " "and send what you find to Tom"

    intents = analyze_intents(query)

    assert len(intents) == 2

    assert intents[0]["text"] == ("Search the web for the latest Nvidia news")

    assert intents[1]["text"] == ("send what you find to Tom")


def test_nested_action_preserves_pronoun_context():
    query = "get the sprint work items and " "create a Word document summarizing them"

    intents = analyze_intents(query)

    assert len(intents) == 3

    assert intents[0]["text"] == ("Get the sprint work items")

    assert intents[1]["text"] == ("create a Word document")

    assert intents[2]["text"] == ("summarizing them")


def test_empty_query_returns_no_intents():
    assert analyze_intents("") == []
    assert analyze_intents("   ") == []
