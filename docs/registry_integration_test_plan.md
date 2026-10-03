# Brown Octopus Registry Integration Test Plan

This document validates Brown Octopus against a capability registry that
returns MCP server records. It is intended for an external integration tester.

Use a newly built Brown Octopus wheel containing the registry-source changes.
Do not use an editable Brown Octopus install.

## What this test covers

The registry flow is:

```text
registry API
    -> paginated MCP server records
    -> MCP tool discovery per server
    -> normalized capabilities
    -> Brown Octopus index
    -> runtime retrieval
```

This is different from `OctopusIndex.from_api()`, which consumes an API that
already returns one normalized record per capability.

## Required environment

Set these values only in the test environment. Never commit them or print
their values:

```text
CAPABILITY_REGISTRY_URL   Registry endpoint returning MCP server records
CAPABILITY_REGISTRY_TOKEN Optional registry API token
TEST_MCP_URL              Optional dedicated test MCP URL
REDIS_URL                 Optional Redis URL for session-store tests
```

The registry must expose a disposable or read-only test MCP where possible.
Do not run write-capable operations against production services.

## Install the test wheel

From the consumer project:

```powershell
uv add "C:\path\to\brown_octopus-<version>-py3-none-any.whl"
uv sync
```

Verify the installed package is loaded from the virtual environment:

```powershell
uv run python -c "import brown_octopus; print(brown_octopus.__file__)"
uv run brown-octopus doctor
```

The import path must not point to the Brown Octopus source checkout.

## Test 1: registry discovery

Create a registry-backed index:

```python
import asyncio
import os

from brown_octopus import OctopusIndex


async def main():
    headers = {}
    token = os.environ.get("CAPABILITY_REGISTRY_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    index = OctopusIndex.from_mcp_registry(
        os.environ["CAPABILITY_REGISTRY_URL"],
        headers=headers,
        items_path="items",
        page_size=10,
        index_path="data/indexes/registry-test",
    )

    report = await index.create()
    print(report)

    runtime = index.runtime()
    await runtime.initialize()

    result = runtime.retrieve_result(
        "Search the web for the latest news",
        session_id="registry-test",
    )

    print("retrieved:", len(result.retrieved_tools))
    print("active:", len(result.tools))
    for tool in result.tools:
        print({
            "capability_id": tool.get("capability_id"),
            "name": tool.get("name"),
            "mcp_url": tool.get("mcp_url"),
            "tool_name": tool.get("tool_name"),
        })


if __name__ == "__main__":
    asyncio.run(main())
```

Acceptance criteria:

- all registry pages are read;
- MCP tools are discovered from the returned server URLs;
- the index is created;
- runtime initialization succeeds;
- retrieval returns the expected capability;
- `capability_id`, `mcp_url`, and `tool_name` are present where supplied;
- no token appears in output or logs.

## Test 2: pagination

Use a registry with more records than one page, or a local mock that returns
two pages. If the registry is cursor-paginated, the response should contain a
cursor such as:

```json
{
  "items": [],
  "next_cursor": "opaque-cursor-value"
}
```

The source sends that value back using the configured `cursor_param` (default:
`cursor`).

Verify:

- every page is requested;
- all server records are discovered;
- no duplicate capabilities are introduced;
- the final report contains the complete tool count.

## Test 3: expired registry token

Provide a host callback that refreshes registry headers:

```python
async def refresh_headers():
    new_token = await get_new_registry_token()
    return {"Authorization": f"Bearer {new_token}"}


index = OctopusIndex.from_mcp_registry(
    registry_url,
    headers={"Authorization": "Bearer expired-token"},
    refresh_headers=refresh_headers,
)
```

Use a test endpoint that returns 401 or 403 once and succeeds after the
refresh.

Acceptance criteria:

- the refresh callback is called at most once for the request;
- the retry succeeds;
- token values are not logged or included in raised errors.

## Test 4: unreachable registry

Point the source at an unavailable test endpoint.

Verify:

- `discover()` returns a failed/non-authoritative result or the normal update
  error contract;
- the previous valid index remains usable;
- existing capabilities are not silently deleted;
- the error identifies the source and failure type without exposing secrets.

## Test 5: one MCP server unavailable

Use a registry containing at least two servers and make one unavailable.

Verify:

- tools from the healthy server are discovered;
- the failed server appears in `failed_sources`;
- the result is non-authoritative;
- previously indexed tools belonging to the failed server remain available;
- the healthy server's new or changed tools are applied.

## Test 6: flat capability API

Test the separate `from_api()` path with an API returning normalized
capability records:

```json
{
  "items": [
    {
      "capability_id": "outlook:send_email",
      "source_id": "outlook-prod",
      "name": "outlook_send_email",
      "description": "Send an email",
      "input_schema": {"type": "object"},
      "mcp_url": "https://example.test/outlook/mcp",
      "tool_name": "send_email"
    }
  ]
}
```

Verify that `mcp_url` and `tool_name` survive normalization and are present
in `result.tools` after runtime retrieval.

## Test 7: Redis session persistence

Use the existing synchronous Redis SessionStore test plan.

Verify:

- two Octopus instances share the same session;
- different sessions remain isolated;
- concurrent same-session mutations remain atomic;
- reset/delete are visible across instances;
- TTL and active-cap behavior remain unchanged;
- Redis credentials are not logged.

Use `redis.Redis`, not `redis.asyncio.Redis`, with the current synchronous
`retrieve_result()` contract.

## Test 8: read-only MCP execution

After retrieval, pass the returned `mcp_url`, `tool_name`, and input schema to
the host-side MCP execution layer.

Verify that:

- Brown Octopus only returns capability metadata;
- the host performs execution;
- the read-only MCP operation succeeds;
- Brown Octopus does not receive or log execution credentials.

## Write-capable operations

Do not run create, update, send, or delete operations against production
services. Run them only against explicitly approved disposable test services
and record the exact operation and test data used.

## Final report

Report each item as `PASS`, `FAIL`, or `SKIPPED`:

- package/import/doctor;
- registry discovery;
- pagination;
- token refresh;
- unreachable registry handling;
- partial MCP failure preservation;
- flat capability API;
- Redis persistence;
- read-only MCP execution;
- write-capable execution.

Include:

- Python and operating system;
- installed wheel filename and hash;
- registry record count;
- discovered server count;
- discovered capability count;
- index path used;
- whether the source repository was modified;
- any secrets or write operations intentionally omitted.
