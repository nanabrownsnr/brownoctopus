# Brown Octopus roadmap

Brown Octopus is being developed in two deliberate phases.

## Product direction

Brown Octopus is the capability-control layer for production AI agents. The
library selects and maintains the capabilities exposed to an agent. A future
managed gateway can enforce policy and mediate execution at the boundary
between agents and MCP servers.

The core library should remain focused. It should not become a general agent
framework, memory platform, model gateway, scheduler, or notification system.

## Phase 1: capability-context library

The first phase is to make the library measurably excellent at selecting and
maintaining agent capability context.

Priorities:

- benchmark tool-selection recall, precision, and task completion;
- measure token reduction against exposing every tool;
- compare Brown Octopus with keyword search, basic vector search, and native
  progressive tool discovery;
- reduce indexing time and memory/GPU requirements;
- preserve incremental add/remove behavior and safe snapshot publication;
- support local, PostgreSQL, and MongoDB index stores;
- keep retrieval framework- and model-provider-independent;
- expose useful indexing reports and retrieval metadata;
- validate the library with multiple independent host applications.

The host application remains responsible for conversation history, model
execution, tool execution, credentials, authorization, and user identity.

The following are intentionally not Phase 1 priorities:

- autonomous workers;
- scheduled agent commitments;
- persistent conversation graphs;
- automatic model routing;
- proactive agent messaging;
- centralized credential ownership;
- PII mediation during tool execution.

Those ideas may become gateway or host-application capabilities later, but
they should not dilute the library's measurable core value.

## Phase 2: managed capability gateway

The gateway is the commercial control plane and execution boundary built around
the library's proven indexing and retrieval behavior.

Potential gateway capabilities:

- hosted or private capability catalogs;
- tenant, team, agent, and user identity;
- RBAC and context-aware ABAC policies;
- tool approval policies for high-risk actions;
- execution-time enforcement, not only retrieval-time filtering;
- complete audit events for capability discovery and execution;
- source health and policy status;
- usage and cost reporting;
- customer-managed or gateway-managed MCP connections;
- hybrid deployment with the data plane remaining in the customer's network.

### PII and sensitive-data mediation

For sensitive tools such as HR, finance, healthcare, and customer systems, the
gateway could inspect both directions of a tool call:

```text
agent request
    -> detect and classify sensitive values
    -> tokenize or pseudonymize approved values
    -> call MCP tool with policy-safe payload
    -> inspect and classify tool response
    -> restore values only for an authorized recipient
    -> audit the policy decision without unnecessarily storing raw PII
```

This should use scoped reversible tokenization or vault-backed references when
restoration is required. Generic irreversible masking is not sufficient for a
tool that needs the original value. Restoration must be policy-controlled,
request-scoped, authenticated, and auditable.

The core library should not attempt to provide this. It does not own tool
execution or credentials, so it cannot guarantee that a tool call was actually
filtered. The gateway or host execution boundary must enforce it.

## Commercial thesis

The open-source library proves the capability-selection engine and makes it
easy for host applications to adopt. The managed gateway creates enterprise
value through governance, enforcement, auditability, and operations.

The positioning is:

> Brown Octopus selects the capabilities an agent needs. The Brown Octopus
> gateway governs and audits what the agent is allowed to use.

The gateway should be validated with design partners before building a large
managed platform. The first discovery questions are:

1. How many agents and MCP tools does the organization operate?
2. Can it produce a complete tool-use audit today?
3. Can it enforce access by user, tenant, task, and environment?
4. Which tools require approval or PII controls?
5. Would it pay for a hosted control plane, private deployment, or both?

This roadmap is intentionally a direction, not a commitment to implement all
Phase 2 features before Phase 1 has evidence of reliable improvement.
