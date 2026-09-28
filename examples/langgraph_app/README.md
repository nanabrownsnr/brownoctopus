# LangGraph integration example

This is a separate application. It imports the repository as an installed
`octopus` package and keeps all LangGraph-specific code outside the Octopus
core.

## Run

From this directory:

```bash
uv sync
uv run python app.py
```

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
