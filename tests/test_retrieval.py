from octopus.retriever import retrieve_tools


def test_compound_calendar_and_reply_request():
    query = "check my calendar and reply to Tom"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any("calendar" in name.lower() for name in names)
    assert any(
        name.endswith("reply_email") or name.endswith("reply_chat_message")
        for name in names
    )


def test_leave_and_email_request():
    query = (
        "check whether I submitted Christmas leave " "and email my manager if I haven't"
    )

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any("leave" in name.lower() for name in names)
    assert any("email" in name.lower() for name in names)


def test_explicit_tool_request_exposes_send_email_capability():
    query = (
        "use the send email tool to send Tom " "a message saying coffee is for closers"
    )

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any(name.endswith("send_email") for name in names)


def test_direct_send_email_request():
    query = "send Tom an email saying coffee is for closers"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any(name.endswith("send_email") for name in names)


def test_three_action_request_leave_meeting_email():
    query = (
        "check my leave balance, "
        "schedule a meeting with Sarah tomorrow, "
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
        "find the quarterly report, "
        "send it to Tom, "
        "and if he replies schedule a meeting with him"
    )

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any("file" in name.lower() or "knowledge" in name.lower() for name in names)

    assert any(
        name.endswith("send_email")
        or name.endswith("send_chat_message")
        or name.endswith("forward_email")
        for name in names
    )

    assert any(
        "meeting" in name.lower() or "calendar" in name.lower() for name in names
    )


def test_leave_balance_prefers_leave_balance_capability():
    query = "check my leave balance"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert names[0].endswith("get_leave_balance")


# def test_find_report_in_filesystem_prefers_filesystem():
#     query = "find the quarterly report on my computer"

#     results = retrieve_tools(query)
#     names = [tool["name"] for tool in results]

#     print(names)

#     assert "filesystem_search_files" in names


def test_employee_files_prefers_hr_platform():
    query = "find my employee files"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert names[0] == "workday_search_employee_files"


def test_employee_files_exposes_hr_file_capability():
    query = "find my employee files"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any(name.startswith("workday_") and "file" in name.lower() for name in names)


def test_workday_leave_context_prefers_workday():
    query = "check how many vacation days I have left"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any(name.startswith("workday_") for name in names)
    assert any(name.endswith("get_leave_balance") for name in names)


def test_outlook_calendar_context_prefers_outlook():
    query = "check my calendar for meetings tomorrow"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any(name.startswith("outlook_") and "calendar" in name for name in names)


def test_hubspot_customer_context_prefers_hubspot():
    query = "find the contact details for one of our customers"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any(name.startswith("hubspot_") for name in names)


def test_hubspot_customer_file_context_prefers_hubspot():
    query = "find the document attached to the customer record"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any(name.startswith("hubspot_") and "file" in name for name in names)


def test_sharepoint_policy_context_prefers_sharepoint():
    query = "find our company remote work policy"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any(name.startswith("sharepoint_") for name in names)


def test_tavily_public_web_context_prefers_tavily():
    query = "search the web for the latest news about Nvidia"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any(name.startswith("tavily_") for name in names)


def test_azure_boards_sprint_context_prefers_azure_boards():
    query = "show me the work items in the current sprint"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any(name.startswith("azure_boards_") for name in names)
    assert any("sprint" in name or "work_item" in name for name in names)


def test_teams_chat_context_prefers_teams():
    query = "send Sarah a chat message"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any(name.startswith("teams_") and "message" in name for name in names)


def test_explicit_gmail_context_prefers_gmail():
    query = "send Tom an email using Gmail"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any(
        name.startswith("gmail_") and name.endswith("send_email") for name in names
    )


def test_explicit_outlook_context_prefers_outlook():
    query = "send Tom an email using Outlook"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert any(
        name.startswith("outlook_") and name.endswith("send_email") for name in names
    )


def test_employee_contract_exposes_search_and_get_tools():
    query = "get my employee contract"

    results = retrieve_tools(query)
    names = [tool["name"] for tool in results]

    print(names)

    assert "workday_search_employee_files" in names
    assert "workday_get_employee_file" in names
