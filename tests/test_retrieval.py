from octopus.retriever import retrieve_tools


def test_retrieves_multiple_capabilities_from_compound_request():
    query = "check my calendar and reply to Tom"

    results = retrieve_tools(query)
    for tool in results:
        print(tool["name"])

    names = [tool["name"] for tool in results]

    assert any("calendar" in name.lower() for name in names)
    assert any("reply" in name.lower() or "email" in name.lower() for name in names)


def test_retrieves_leave_and_email_capabilities():
    query = (
        "check whether I submitted Christmas leave and email my manager if I haven't"
    )

    results = retrieve_tools(query)

    names = [tool["name"] for tool in results]

    assert any("leave" in name.lower() for name in names)
    assert any("email" in name.lower() for name in names)


def test_explicit_tool_request_prefers_named_tool():
    query = "use the send email tool to send Tom a message saying coffee is for closers"

    results = retrieve_tools(query)

    names = [tool["name"] for tool in results]

    print(names)

    assert "send_email" in names


def test_direct_send_email_request():
    query = "send Tom an email saying coffee is for closers"

    results = retrieve_tools(query)

    names = [tool["name"] for tool in results]

    print(names)

    assert "send_email" in names


def test_three_action_request_leave_meeting_email():
    query = (
        "check my leave balance, schedule a meeting with Sarah tomorrow, "
        "and email my manager the details"
    )

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any("leave" in name.lower() for name in names)
    assert any(
        "meeting" in name.lower() or "calendar" in name.lower() for name in names
    )
    assert any("email" in name.lower() for name in names)


def test_three_action_request_file_email_meeting():
    query = (
        "find the quarterly report, send it to Tom, "
        "and if he replies schedule a meeting with him"
    )

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any("file" in name.lower() for name in names)

    assert any("email" in name.lower() or "message" in name.lower() for name in names)

    assert any(
        "meeting" in name.lower() or "calendar" in name.lower() for name in names
    )


def test_leave_balance_prefers_get_leave_balance():
    query = "check my leave balance"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert names[0] == "get_leave_balance"


def test_find_existing_report_prefers_file_search():
    query = "find the quarterly report"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert "search_files" in names