# LangGraph integration example

This is a separate application. It imports `brown-octopus` as a package and
keeps all LangGraph-specific code outside the Brown Octopus core.

When run from this repository, its development configuration points at the
local package source. A consumer application should instead depend on the
published Brown Octopus wheel or release.

## Run

From the repository root, after creating an index:

```bash
brown-octopus setup-models
python examples/setup_index.py
uv sync --project examples/langgraph_app
uv run --project examples/langgraph_app python examples/langgraph_app/app.py
```

The example expects a compatible index at `data/indexes/default`. It uses the
installed Brown Octopus runtime to retrieve capabilities; it does not discover
or execute MCP tools itself.

Set the required model and service credentials first:

```bash
set TYPESAFE_API_KEY=...
set OPENAI_API_KEY=...
```

The example initializes Octopus, processes a single-capability request, a
multi-capability request, and a follow-up request. On every turn it logs the
capability IDs exposed by Octopus and binds only those definitions to the
LangGraph model/tool node for that turn.

`make_demo_tool` is deliberately a harness adapter. It demonstrates normal
LangGraph tool selection and execution, but its body should be replaced by the
application's real MCP/tool executor. Octopus only returns capability context.
