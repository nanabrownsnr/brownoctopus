TOOLS = [
    # =========================================================
    # OUTLOOK
    # =========================================================
    {
        "name": "outlook_reply_email",
        "description": "Reply to an existing email message in Outlook.",
    },
    {
        "name": "outlook_send_email",
        "description": "Send a new email message to one or more recipients using Outlook.",
    },
    {
        "name": "outlook_compose_email",
        "description": "Create a draft email message in Outlook without sending it.",
    },
    {
        "name": "outlook_forward_email",
        "description": "Forward an existing Outlook email message to another recipient.",
    },
    {
        "name": "outlook_search_emails",
        "description": "Search Outlook email messages by keyword, sender, recipient, subject, or date.",
    },
    {
        "name": "outlook_get_email",
        "description": "Retrieve the contents and details of a specific Outlook email message.",
    },
    {
        "name": "outlook_list_emails",
        "description": "List recent email messages from an Outlook mailbox.",
    },
    {
        "name": "outlook_delete_email",
        "description": "Delete an email message from Outlook.",
    },
    # =========================================================
    # GMAIL
    # =========================================================
    {
        "name": "gmail_reply_email",
        "description": "Reply to an existing email message in Gmail.",
    },
    {
        "name": "gmail_send_email",
        "description": "Send a new email message to one or more recipients using Gmail.",
    },
    {
        "name": "gmail_compose_email",
        "description": "Create a draft email message in Gmail without sending it.",
    },
    {
        "name": "gmail_forward_email",
        "description": "Forward an existing Gmail message to another recipient.",
    },
    {
        "name": "gmail_search_emails",
        "description": "Search Gmail messages by keyword, sender, recipient, subject, or date.",
    },
    {
        "name": "gmail_get_email",
        "description": "Retrieve the contents and details of a specific Gmail message.",
    },
    # =========================================================
    # OUTLOOK CALENDAR
    # =========================================================
    {
        "name": "outlook_list_calendar_events",
        "description": "List events and appointments from an Outlook calendar.",
    },
    {
        "name": "outlook_get_calendar_event",
        "description": "Retrieve details of a specific Outlook calendar event.",
    },
    {
        "name": "outlook_search_calendar",
        "description": "Search an Outlook calendar for meetings, events, and appointments.",
    },
    {
        "name": "outlook_create_calendar_event",
        "description": "Create a new event or appointment on an Outlook calendar.",
    },
    {
        "name": "outlook_update_calendar_event",
        "description": "Update an existing Outlook calendar event or appointment.",
    },
    {
        "name": "outlook_delete_calendar_event",
        "description": "Delete or cancel an Outlook calendar event.",
    },
    {
        "name": "outlook_check_calendar_availability",
        "description": "Check a person's Outlook calendar availability for a date or time range.",
    },
    {
        "name": "outlook_schedule_meeting",
        "description": "Schedule a meeting with one or more participants using Outlook.",
    },
    # =========================================================
    # WORKDAY / HR
    # =========================================================
    {
        "name": "workday_list_leave_applications",
        "description": "List employee leave applications from Workday.",
    },
    {
        "name": "workday_get_leave_application",
        "description": "Retrieve details and status of an employee leave application from Workday.",
    },
    {
        "name": "workday_create_leave_application",
        "description": "Create a new employee leave application in Workday.",
    },
    {
        "name": "workday_submit_leave_application",
        "description": "Submit an employee leave application for approval in Workday.",
    },
    {
        "name": "workday_cancel_leave_application",
        "description": "Cancel an existing employee leave application in Workday.",
    },
    {
        "name": "workday_get_leave_balance",
        "description": "Retrieve an employee's available and remaining leave balance from Workday.",
    },
    {
        "name": "workday_approve_leave_application",
        "description": "Approve an employee leave application in Workday.",
    },
    {
        "name": "workday_reject_leave_application",
        "description": "Reject an employee leave application in Workday.",
    },
    {
        "name": "workday_search_employee_files",
        "description": "Search employee HR files, personnel documents, contracts, and employment records in Workday.",
    },
    {
        "name": "workday_get_employee_file",
        "description": "Retrieve a specific employee HR file, personnel document, contract, or employment record from Workday.",
    },
    # =========================================================
    # AZURE BOARDS
    # =========================================================
    {
        "name": "azure_boards_create_work_item",
        "description": "Create a project work item, task, user story, bug, or ticket in Azure Boards.",
    },
    {
        "name": "azure_boards_update_work_item",
        "description": "Update the fields or status of an existing Azure Boards work item.",
    },
    {
        "name": "azure_boards_get_work_item",
        "description": "Retrieve details about a specific Azure Boards work item.",
    },
    {
        "name": "azure_boards_list_work_items",
        "description": "List project work items, tasks, user stories, bugs, or tickets in Azure Boards.",
    },
    {
        "name": "azure_boards_assign_work_item",
        "description": "Assign an Azure Boards work item or task to a team member.",
    },
    {
        "name": "azure_boards_list_sprint_items",
        "description": "List the Azure Boards work items currently assigned to a sprint.",
    },
    {
        "name": "azure_boards_get_sprint",
        "description": "Retrieve information about the current or specified sprint in Azure Boards.",
    },
    # =========================================================
    # FILESYSTEM
    # =========================================================
    {
        "name": "filesystem_search_files",
        "description": (
            "Find, locate, or search files stored locally on the user's "
            "computer, laptop, device, or filesystem, including documents, "
            "reports, presentations, and spreadsheets."
        ),
    },
    {
        "name": "filesystem_get_file",
        "description": "Retrieve a specific file or document from the filesystem.",
    },
    {
        "name": "filesystem_create_document",
        "description": "Create a new document in the filesystem.",
    },
    {
        "name": "filesystem_update_document",
        "description": "Update the contents of an existing filesystem document.",
    },
    {
        "name": "filesystem_delete_file",
        "description": "Delete an existing file or document from the filesystem.",
    },
    {
        "name": "filesystem_share_file",
        "description": "Share a filesystem file or document with another person.",
    },
    # =========================================================
    # HUBSPOT
    # =========================================================
    {
        "name": "hubspot_search_contacts",
        "description": "Search HubSpot for people, contacts, leads, or customers.",
    },
    {
        "name": "hubspot_get_contact",
        "description": "Retrieve contact or customer information from HubSpot.",
    },
    {
        "name": "hubspot_create_contact",
        "description": "Create a new person, lead, customer, or contact record in HubSpot.",
    },
    {
        "name": "hubspot_search_files",
        "description": "Search files and documents associated with customers, contacts, companies, or CRM records in HubSpot.",
    },
    {
        "name": "hubspot_get_file",
        "description": "Retrieve a file or document associated with a customer, contact, company, or CRM record in HubSpot.",
    },
    # =========================================================
    # TEAMS
    # =========================================================
    {
        "name": "teams_send_chat_message",
        "description": "Send a Microsoft Teams chat message to a person or channel.",
    },
    {
        "name": "teams_reply_chat_message",
        "description": "Reply to an existing Microsoft Teams chat message.",
    },
    {
        "name": "teams_list_chat_messages",
        "description": "List recent messages from a Microsoft Teams chat or channel.",
    },
    {
        "name": "teams_search_chat_messages",
        "description": "Search Microsoft Teams messages by keyword, sender, channel, or date.",
    },
    # =========================================================
    # KNOWLEDGE / WEB SEARCH
    # =========================================================
    {
        "name": "sharepoint_search_knowledge",
        "description": "Search internal organizational knowledge, documents, policies, and pages stored in SharePoint.",
    },
    {
        "name": "sharepoint_get_knowledge_article",
        "description": "Retrieve a specific internal article, page, policy, or knowledge document from SharePoint.",
    },
    {
        "name": "tavily_search_web",
        "description": "Search the public web for current or external information using Tavily.",
    },
    # =========================================================
    # DATA / REPORTING
    # =========================================================
    {
        "name": "postgres_query_database",
        "description": "Run a read-only query against a PostgreSQL database and return the results.",
    },
    {
        "name": "analytics_create_chart",
        "description": "Create a chart or visualization from structured data.",
    },
    {
        "name": "analytics_generate_report",
        "description": "Generate a report from available information and structured data.",
    },
]
