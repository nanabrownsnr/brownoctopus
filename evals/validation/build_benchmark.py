import json
from pathlib import Path


OUT = Path("data/evals/octopus_validation_benchmark_v1.json")


def task(task_id, category, query, required, supporting):
    return {
        "id": task_id,
        "category": category,
        "query": query,
        "required_tools": required,
        "supporting_tools": supporting,
        "turns": [{
            "query": query,
            "required_tools": required,
            "supporting_tools": supporting,
        }],
    }


def conversation(task_id, query1, query2, required, supporting, turn1, turn2):
    return {
        "id": task_id,
        "category": "multi_turn",
        "query": f"{query1} Then {query2}",
        "required_tools": required,
        "supporting_tools": supporting,
        "turns": [
            {"query": query1, **turn1},
            {"query": query2, **turn2},
        ],
    }


tasks = []

# These queries and labels were authored from the frozen tool descriptions and
# task semantics before running validation retrieval. They deliberately use new
# wording and new compositions rather than copying the development benchmark.
tasks.extend([
    task("single_v001", "single_capability", "Show every environment configured for my Railway workspace", ["railway_tools_project_environments"], []),
    task("single_v002", "single_capability", "Restart the payments service in its active environment", ["railway_tools_service_restart"], []),
    task("single_v003", "single_capability", "How much disk space is my account using?", ["quest_studio_file_storage_get_storage_usage"], []),
    task("single_v004", "single_capability", "Find the raw contents of the quarterly plan file", ["quest_studio_file_storage_get_file_content"], []),
    task("single_v005", "single_capability", "Look up repositories implementing event sourcing", ["github_mcp_server_search_repositories"], []),
    task("single_v006", "single_capability", "Show the commits on the release branch", ["github_mcp_server_list_commits"], []),
    task("single_v007", "single_capability", "Tell me who is on the platform team", ["project_management_tools_list_team_members"], []),
    task("single_v008", "single_capability", "Display the current Azure board for the mobile project", ["project_management_tools_view_board"], []),
    task("single_v009", "single_capability", "List the people in our organisation", ["carrot_mcp_server_api_list_employees"], []),
    task("single_v010", "single_capability", "Show the HR summary for this month", ["carrot_mcp_server_api_hr_dashboard_summary"], []),
    task("single_v011", "single_capability", "Search my inbox for messages about the renewal", ["outlook_mail_management_search_outlook_mail"], []),
    task("single_v012", "single_capability", "Remove the selected Outlook message", ["outlook_mail_management_delete_email_message"], []),
    task("single_v013", "single_capability", "List the sites I can access in SharePoint", ["sharepoint_tools_get_sharepoint_sites"], []),
    task("single_v014", "single_capability", "Send a note through Gmail to the design group", ["gmail_tools_send_google_email"], []),
    task("single_v015", "single_capability", "Find the headings in the contract document", ["office_word_mcp_server_api_get_document_outline"], []),
    task("single_v016", "single_capability", "Replace the old company name throughout the document", ["office_word_mcp_server_api_search_replace"], []),
    task("single_v017", "single_capability", "Check whether the analytics database is reachable", ["supabase_mcp_check_database_health"], []),
    task("single_v018", "single_capability", "List the tables in the reporting project", ["supabase_mcp_get_database_tables"], []),
    task("single_v019", "single_capability", "Create a persistent volume for the worker service", ["railway_tools_volume_create"], []),
    task("single_v020", "single_capability", "Show the messages in my Gmail inbox", ["gmail_tools_get_google_emails"], []),
])

tasks.extend([
    task("family_v001", "capability_family", "Draft a Word memo with a title and explanatory paragraphs", ["office_word_mcp_server_api_create_word_document"], ["office_word_mcp_server_api_add_heading", "office_word_mcp_server_api_add_paragraph"]),
    task("family_v002", "capability_family", "Inspect a Word file before exporting it as a PDF", ["office_word_mcp_server_api_convert_to_pdf"], ["office_word_mcp_server_api_get_document_info", "office_word_mcp_server_api_get_document_text"]),
    task("family_v003", "capability_family", "Open a pull request for the changes on my feature branch", ["github_mcp_server_create_pull_request"], ["github_mcp_server_list_branches", "github_mcp_server_get_file_contents"]),
    task("family_v004", "capability_family", "Remove an obsolete GitHub branch and its files", ["github_mcp_server_delete_file"], ["github_mcp_server_list_branches", "github_mcp_server_get_me"]),
    task("family_v005", "capability_family", "Prepare a new Railway service from a container image", ["railway_tools_service_create_from_image"], ["railway_tools_project_list", "railway_tools_database_list_types"]),
    task("family_v006", "capability_family", "Change the settings of an existing Railway service", ["railway_tools_service_update"], ["railway_tools_service_info", "railway_tools_service_list"]),
    task("family_v007", "capability_family", "Review and update a leave application", ["carrot_mcp_server_api_update_leave_application"], ["carrot_mcp_server_api_list_leave_application", "carrot_mcp_server_api_get_employee"]),
    task("family_v008", "capability_family", "Record project hours and show the resulting summary", ["carrot_mcp_server_api_log_hours_entry"], ["carrot_mcp_server_api_work_hours_summary"]),
    task("family_v009", "capability_family", "Move an Azure work item to a new owner", ["project_management_tools_assign_work_item"], ["project_management_tools_get_work_item", "project_management_tools_list_users"]),
    task("family_v010", "capability_family", "Add a progress comment to an Azure work item", ["project_management_tools_add_work_item_comment"], ["project_management_tools_get_work_item_comments", "project_management_tools_get_work_item"]),
    task("family_v011", "capability_family", "Create a team for the infrastructure project", ["project_management_tools_create_team"], ["project_management_tools_list_projects", "project_management_tools_list_teams"]),
    task("family_v012", "capability_family", "Copy a Word document into the archive collection", ["office_word_mcp_server_api_copy_document"], ["office_word_mcp_server_api_list_files", "office_word_mcp_server_api_get_document_info"]),
    task("family_v013", "capability_family", "Make a table of findings in the report", ["office_word_mcp_server_api_add_table"], ["office_word_mcp_server_api_get_document_info", "office_word_mcp_server_api_add_paragraph"]),
    task("family_v014", "capability_family", "Forward the incident email to the on-call address", ["outlook_mail_management_forward_email"], ["outlook_mail_management_get_email_messages", "outlook_mail_management_find_email_address"]),
    task("family_v015", "capability_family", "Reply to the customer message in Outlook", ["outlook_mail_management_reply_to_email"], ["outlook_mail_management_search_outlook_mail", "outlook_mail_management_get_email_messages"]),
    task("family_v016", "capability_family", "Create a text note in the team SharePoint library", ["sharepoint_tools_create_txt_file_in_sharepoint"], ["sharepoint_tools_get_sharepoint_sites", "sharepoint_tools_get_sharepoint_documents"]),
    task("family_v017", "capability_family", "Read the schema before selecting rows from the database", ["supabase_mcp_select_database_table_data"], ["supabase_mcp_get_table_schema", "supabase_mcp_get_database_tables"]),
    task("family_v018", "capability_family", "Update rows in the customer table", ["supabase_mcp_update_database_table_data"], ["supabase_mcp_get_table_schema", "supabase_mcp_check_database_health"]),
    task("family_v019", "capability_family", "Create a sprint and inspect its capacity", ["project_management_tools_create_sprint"], ["project_management_tools_get_user_capacity", "project_management_tools_list_sprints"]),
    task("family_v020", "capability_family", "Remove a project volume after checking its details", ["railway_tools_volume_delete"], ["railway_tools_volume_list", "railway_tools_volume_update"]),
])

tasks.extend([
    task("compound_v001", "compound", "Find repositories about graph databases and open the most promising one", ["github_mcp_server_search_repositories", "github_mcp_server_get_file_contents"], ["github_mcp_server_get_me"]),
    task("compound_v002", "compound", "Check Railway environments and restart the staging service", ["railway_tools_project_environments", "railway_tools_service_restart"], ["railway_tools_service_info", "railway_tools_service_list"]),
    task("compound_v003", "compound", "Create a Word memo and turn it into a PDF", ["office_word_mcp_server_api_create_word_document", "office_word_mcp_server_api_convert_to_pdf"], ["office_word_mcp_server_api_add_heading", "office_word_mcp_server_api_add_paragraph"]),
    task("compound_v004", "compound", "Search Outlook for the invoice and forward it to finance", ["outlook_mail_management_search_outlook_mail", "outlook_mail_management_forward_email"], ["outlook_mail_management_find_email_address"]),
    task("compound_v005", "compound", "List SharePoint sites and create a note in the selected library", ["sharepoint_tools_get_sharepoint_sites", "sharepoint_tools_create_txt_file_in_sharepoint"], ["sharepoint_tools_get_sharepoint_documents"]),
    task("compound_v006", "compound", "Check database health and retrieve the schema for the orders table", ["supabase_mcp_check_database_health", "supabase_mcp_get_table_schema"], ["supabase_mcp_get_database_tables"]),
    task("compound_v007", "compound", "Create an Azure team and list its members", ["project_management_tools_create_team", "project_management_tools_list_team_members"], ["project_management_tools_list_teams"]),
    task("compound_v008", "compound", "Log time against the migration project and review the hours summary", ["carrot_mcp_server_api_log_hours_entry", "carrot_mcp_server_api_work_hours_summary"], ["carrot_mcp_server_api_list_employees"]),
    task("compound_v009", "compound", "Find the latest support email and reply with an update", ["outlook_mail_management_search_outlook_mail", "outlook_mail_management_reply_to_email"], ["outlook_mail_management_get_email_messages"]),
    task("compound_v010", "compound", "Create a GitHub branch and push the prepared files", ["github_mcp_server_create_branch", "github_mcp_server_push_files"], ["github_mcp_server_get_me", "github_mcp_server_list_branches"]),
    task("compound_v011", "compound", "List Railway volumes and update the backup volume", ["railway_tools_volume_list", "railway_tools_volume_update"], ["railway_tools_project_list"]),
    task("compound_v012", "compound", "List Azure projects and open the board for one project", ["project_management_tools_list_projects", "project_management_tools_view_board"], ["project_management_tools_get_project"]),
    task("compound_v013", "compound", "Create a leave request and submit it for approval", ["carrot_mcp_server_api_create_leave_application", "carrot_mcp_server_api_submit_leave_application"], ["carrot_mcp_server_api_list_leave_application"]),
    task("compound_v014", "compound", "Read a report's text and replace its draft wording", ["office_word_mcp_server_api_get_document_text", "office_word_mcp_server_api_search_replace"], ["office_word_mcp_server_api_get_document_info"]),
    task("compound_v015", "compound", "Send a Gmail message and then inspect the inbox for the reply", ["gmail_tools_send_google_email", "gmail_tools_get_google_emails"], ["gmail_tools_reply_google_email"]),
    task("compound_v016", "compound", "Create a Railway project and deploy a service from a repository", ["railway_tools_project_create", "railway_tools_service_create_from_repo"], ["railway_tools_project_list"]),
    task("compound_v017", "compound", "Find the sprint details and export its work to SharePoint", ["project_management_tools_get_sprint", "project_management_tools_export_sprint_to_sharepoint"], ["project_management_tools_list_sprints", "sharepoint_tools_get_sharepoint_sites"]),
    task("compound_v018", "compound", "Query the customer data and update a matching row", ["supabase_mcp_select_database_table_data", "supabase_mcp_update_database_table_data"], ["supabase_mcp_get_table_schema", "supabase_mcp_check_database_health"]),
    task("compound_v019", "compound", "Search my files and update the selected file", ["quest_studio_file_storage_query_files", "quest_studio_file_storage_update_file"], ["quest_studio_file_storage_get_files", "quest_studio_file_storage_get_file"]),
    task("compound_v020", "compound", "Get the project team and assign its work item", ["project_management_tools_get_team", "project_management_tools_assign_work_item"], ["project_management_tools_list_team_members", "project_management_tools_get_work_item"]),
])

tasks.extend([
    task("dependency_v001", "hidden_dependency", "Send the deployment notes to the release coordinator", ["outlook_mail_management_send_email"], ["outlook_mail_management_find_email_address"]),
    task("dependency_v002", "hidden_dependency", "Reply to the person who asked about the contract", ["outlook_mail_management_reply_to_email"], ["outlook_mail_management_get_email_messages"]),
    task("dependency_v003", "hidden_dependency", "Forward the security alert to the incident lead", ["outlook_mail_management_forward_email"], ["outlook_mail_management_find_email_address"]),
    task("dependency_v004", "hidden_dependency", "Create the report document with the findings section", ["office_word_mcp_server_api_create_word_document"], ["office_word_mcp_server_api_add_heading", "office_word_mcp_server_api_add_paragraph"]),
    task("dependency_v005", "hidden_dependency", "Add the chart to the existing Word report", ["office_word_mcp_server_api_add_picture"], ["office_word_mcp_server_api_get_document_info"]),
    task("dependency_v006", "hidden_dependency", "Turn the completed proposal into a PDF", ["office_word_mcp_server_api_convert_to_pdf"], ["office_word_mcp_server_api_get_document_text"]),
    task("dependency_v007", "hidden_dependency", "Open a pull request for the authentication fix", ["github_mcp_server_create_pull_request"], ["github_mcp_server_list_branches", "github_mcp_server_get_file_contents"]),
    task("dependency_v008", "hidden_dependency", "Update the configuration file in the selected repository", ["github_mcp_server_create_or_update_file"], ["github_mcp_server_get_file_contents"]),
    task("dependency_v009", "hidden_dependency", "Deploy the service from the team's repository", ["railway_tools_service_create_from_repo"], ["railway_tools_project_list", "railway_tools_service_list"]),
    task("dependency_v010", "hidden_dependency", "Restart the service after checking its configuration", ["railway_tools_service_restart"], ["railway_tools_service_info"]),
    task("dependency_v011", "hidden_dependency", "Submit the leave request I just prepared", ["carrot_mcp_server_api_submit_leave_application"], ["carrot_mcp_server_api_create_leave_application"]),
    task("dependency_v012", "hidden_dependency", "Update the employee record for the new role", ["carrot_mcp_server_api_update_employee"], ["carrot_mcp_server_api_get_employee"]),
    task("dependency_v013", "hidden_dependency", "Assign the bug to its new owner", ["project_management_tools_assign_work_item"], ["project_management_tools_get_work_item", "project_management_tools_list_users"]),
    task("dependency_v014", "hidden_dependency", "Mark the migration ticket as finished", ["project_management_tools_update_task_status"], ["project_management_tools_get_work_item"]),
    task("dependency_v015", "hidden_dependency", "Create the note in the department's SharePoint library", ["sharepoint_tools_create_txt_file_in_sharepoint"], ["sharepoint_tools_get_sharepoint_sites", "sharepoint_tools_get_sharepoint_documents"]),
    task("dependency_v016", "hidden_dependency", "Run the revenue query in the reporting database", ["supabase_mcp_select_database_table_data"], ["supabase_mcp_check_database_health", "supabase_mcp_get_table_schema"]),
    task("dependency_v017", "hidden_dependency", "Update the customer row after checking its table definition", ["supabase_mcp_update_database_table_data"], ["supabase_mcp_get_table_schema"]),
    task("dependency_v018", "hidden_dependency", "Put the revised attachment into the file store", ["quest_studio_file_storage_update_file"], ["quest_studio_file_storage_get_file"]),
    task("dependency_v019", "hidden_dependency", "Delete the obsolete document from storage", ["quest_studio_file_storage_delete_file"], ["quest_studio_file_storage_get_files"]),
    task("dependency_v020", "hidden_dependency", "Add a volume for the newly created database service", ["railway_tools_volume_create"], ["railway_tools_service_info", "railway_tools_project_list"]),
])

tasks.extend([
    conversation("conversation_v001", "Review the open security pull requests", "Merge the approved one", ["github_mcp_server_list_pull_requests", "github_mcp_server_merge_pull_request"], [], {"required_tools": ["github_mcp_server_list_pull_requests"], "supporting_tools": []}, {"required_tools": ["github_mcp_server_merge_pull_request"], "supporting_tools": []}),
    conversation("conversation_v002", "Find the architecture document in my files", "Read its contents", ["quest_studio_file_storage_query_files", "quest_studio_file_storage_get_file_content"], ["quest_studio_file_storage_get_file"], {"required_tools": ["quest_studio_file_storage_query_files"], "supporting_tools": []}, {"required_tools": ["quest_studio_file_storage_get_file_content"], "supporting_tools": ["quest_studio_file_storage_get_file"]}),
    conversation("conversation_v003", "List the current Railway projects", "Show the services in the selected project", ["railway_tools_project_list", "railway_tools_service_list"], ["railway_tools_project_info"], {"required_tools": ["railway_tools_project_list"], "supporting_tools": []}, {"required_tools": ["railway_tools_service_list"], "supporting_tools": ["railway_tools_project_info"]}),
    conversation("conversation_v004", "Find the team's latest email", "Forward it to the customer contact", ["outlook_mail_management_search_outlook_mail", "outlook_mail_management_forward_email"], ["outlook_mail_management_find_email_address"], {"required_tools": ["outlook_mail_management_search_outlook_mail"], "supporting_tools": []}, {"required_tools": ["outlook_mail_management_forward_email"], "supporting_tools": ["outlook_mail_management_find_email_address"]}),
    conversation("conversation_v005", "Open the Word template list", "Copy the chosen template", ["office_word_mcp_server_api_list_files", "office_word_mcp_server_api_copy_document"], ["office_word_mcp_server_api_get_document_info"], {"required_tools": ["office_word_mcp_server_api_list_files"], "supporting_tools": []}, {"required_tools": ["office_word_mcp_server_api_copy_document"], "supporting_tools": ["office_word_mcp_server_api_get_document_info"]}),
    conversation("conversation_v006", "Check the orders database connection", "Show the schema for its invoices table", ["supabase_mcp_check_database_health", "supabase_mcp_get_table_schema"], ["supabase_mcp_get_database_tables"], {"required_tools": ["supabase_mcp_check_database_health"], "supporting_tools": []}, {"required_tools": ["supabase_mcp_get_table_schema"], "supporting_tools": ["supabase_mcp_get_database_tables"]}),
    conversation("conversation_v007", "List teams in the product project", "Show the members of the release team", ["project_management_tools_list_teams", "project_management_tools_list_team_members"], ["project_management_tools_get_team"], {"required_tools": ["project_management_tools_list_teams"], "supporting_tools": []}, {"required_tools": ["project_management_tools_list_team_members"], "supporting_tools": ["project_management_tools_get_team"]}),
    conversation("conversation_v008", "Open the current sprint", "Show its capacity summary", ["project_management_tools_get_sprint", "project_management_tools_get_user_capacity"], ["project_management_tools_list_sprints"], {"required_tools": ["project_management_tools_get_sprint"], "supporting_tools": []}, {"required_tools": ["project_management_tools_get_user_capacity"], "supporting_tools": ["project_management_tools_list_sprints"]}),
    conversation("conversation_v009", "Create the draft Word report", "Add a summary table to it", ["office_word_mcp_server_api_create_word_document", "office_word_mcp_server_api_add_table"], ["office_word_mcp_server_api_add_heading"], {"required_tools": ["office_word_mcp_server_api_create_word_document"], "supporting_tools": []}, {"required_tools": ["office_word_mcp_server_api_add_table"], "supporting_tools": ["office_word_mcp_server_api_add_heading"]}),
    conversation("conversation_v010", "Look up the feature repository", "Show the latest commit", ["github_mcp_server_search_repositories", "github_mcp_server_get_commit"], ["github_mcp_server_list_commits"], {"required_tools": ["github_mcp_server_search_repositories"], "supporting_tools": []}, {"required_tools": ["github_mcp_server_get_commit"], "supporting_tools": ["github_mcp_server_list_commits"]}),
    conversation("conversation_v011", "List available leave applications", "Submit the selected request", ["carrot_mcp_server_api_list_leave_application", "carrot_mcp_server_api_submit_leave_application"], ["carrot_mcp_server_api_create_leave_application"], {"required_tools": ["carrot_mcp_server_api_list_leave_application"], "supporting_tools": []}, {"required_tools": ["carrot_mcp_server_api_submit_leave_application"], "supporting_tools": ["carrot_mcp_server_api_create_leave_application"]}),
    conversation("conversation_v012", "Find the employee record", "Update the department field", ["carrot_mcp_server_api_get_employee", "carrot_mcp_server_api_update_employee"], [], {"required_tools": ["carrot_mcp_server_api_get_employee"], "supporting_tools": []}, {"required_tools": ["carrot_mcp_server_api_update_employee"], "supporting_tools": []}),
    conversation("conversation_v013", "List the files in storage", "Remove the obsolete export", ["quest_studio_file_storage_get_files", "quest_studio_file_storage_delete_file"], ["quest_studio_file_storage_get_file"], {"required_tools": ["quest_studio_file_storage_get_files"], "supporting_tools": []}, {"required_tools": ["quest_studio_file_storage_delete_file"], "supporting_tools": ["quest_studio_file_storage_get_file"]}),
    conversation("conversation_v014", "Check the worker service details", "Change its resource configuration", ["railway_tools_service_info", "railway_tools_service_update"], ["railway_tools_service_list"], {"required_tools": ["railway_tools_service_info"], "supporting_tools": []}, {"required_tools": ["railway_tools_service_update"], "supporting_tools": ["railway_tools_service_list"]}),
    conversation("conversation_v015", "Find the design message", "Reply with the review outcome", ["gmail_tools_get_google_emails", "gmail_tools_reply_google_email"], ["gmail_tools_send_google_email"], {"required_tools": ["gmail_tools_get_google_emails"], "supporting_tools": []}, {"required_tools": ["gmail_tools_reply_google_email"], "supporting_tools": ["gmail_tools_send_google_email"]}),
    conversation("conversation_v016", "List the SharePoint documents", "Create a text note beside them", ["sharepoint_tools_get_sharepoint_documents", "sharepoint_tools_create_txt_file_in_sharepoint"], ["sharepoint_tools_get_sharepoint_sites"], {"required_tools": ["sharepoint_tools_get_sharepoint_documents"], "supporting_tools": []}, {"required_tools": ["sharepoint_tools_create_txt_file_in_sharepoint"], "supporting_tools": ["sharepoint_tools_get_sharepoint_sites"]}),
    conversation("conversation_v017", "Find the sprint work items", "Add a comment to the selected item", ["project_management_tools_list_work_items", "project_management_tools_add_work_item_comment"], ["project_management_tools_get_work_item_comments"], {"required_tools": ["project_management_tools_list_work_items"], "supporting_tools": []}, {"required_tools": ["project_management_tools_add_work_item_comment"], "supporting_tools": ["project_management_tools_get_work_item_comments"]}),
    conversation("conversation_v018", "Create a new project volume", "Show its current properties", ["railway_tools_volume_create", "railway_tools_volume_list"], ["railway_tools_volume_update"], {"required_tools": ["railway_tools_volume_create"], "supporting_tools": []}, {"required_tools": ["railway_tools_volume_list"], "supporting_tools": ["railway_tools_volume_update"]}),
    conversation("conversation_v019", "Search the repository history", "Read the commit that introduced the change", ["github_mcp_server_list_commits", "github_mcp_server_get_commit"], ["github_mcp_server_get_file_contents"], {"required_tools": ["github_mcp_server_list_commits"], "supporting_tools": []}, {"required_tools": ["github_mcp_server_get_commit"], "supporting_tools": ["github_mcp_server_get_file_contents"]}),
    conversation("conversation_v020", "Check the data table definition", "Select the matching customer rows", ["supabase_mcp_get_table_schema", "supabase_mcp_select_database_table_data"], ["supabase_mcp_check_database_health"], {"required_tools": ["supabase_mcp_get_table_schema"], "supporting_tools": []}, {"required_tools": ["supabase_mcp_select_database_table_data"], "supporting_tools": ["supabase_mcp_check_database_health"]}),
])


def main() -> None:
    if len(tasks) != 100:
        raise AssertionError(f"Expected 100 tasks, got {len(tasks)}")
    counts = {}
    for item in tasks:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    if set(counts.values()) != {20}:
        raise AssertionError(counts)
    queries = {item["query"] for item in tasks}
    if len(queries) != len(tasks):
        raise AssertionError("Validation queries must be unique")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "dataset_version": "octopus_capability_validation_v1",
        "description": "Independently authored validation benchmark; labels frozen before retrieval evaluation.",
        "tool_universe": "data/indexes/default/tools.json",
        "authoring_protocol": "Queries and labels authored from canonical tool descriptions without inspecting validation rankings or selector outputs.",
        "review_status": "manually authored validation set",
        "tasks": tasks,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} with {len(tasks)} tasks")


if __name__ == "__main__":
    main()
