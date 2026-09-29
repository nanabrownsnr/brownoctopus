"""Stateful capability-context management, independent of retrieval."""

from brown_octopus.active_tools import update_active_tools
from brown_octopus.capability_scope import (
    normalize_allowed_mcp_urls,
    tool_is_allowed,
)
from brown_octopus.contracts import RetrievalResult
from brown_octopus.tool_registry import capability_id


class ActiveCapabilityContext:
    def __init__(self, tool_lookup, *, ttl: int = 8, active_cap: int = 30) -> None:
        self.tool_lookup = tool_lookup
        self.ttl = ttl
        self.active_cap = active_cap
        self.active_state: dict[str, int] = {}
        self._registered_tools: dict[str, dict] = {}

    def register_tools(self, tools: list[dict]) -> None:
        """Register current-turn definitions for custom pipeline adapters.

        The production registry remains authoritative. This small runtime
        overlay also lets injected/test pipelines provide definitions without
        requiring the SessionStore to persist tool objects.
        """
        self._registered_tools.update(
            {capability_id(tool): tool for tool in tools}
        )

    def _lookup_tools(self, ids):
        found = {capability_id(tool): tool for tool in self.tool_lookup(ids)}
        for identity in ids:
            if identity not in found and identity in self._registered_tools:
                found[identity] = self._registered_tools[identity]
        return [found[identity] for identity in ids if identity in found]

    def update(
        self,
        retrieved_tools: list[dict],
        turn: int,
        allowed_mcp_urls=None,
    ) -> RetrievalResult:
        allowed_urls = normalize_allowed_mcp_urls(allowed_mcp_urls)
        retrieved_tools = [
            tool for tool in retrieved_tools
            if tool_is_allowed(tool, allowed_urls)
        ]

        if allowed_urls is not None:
            current_active_tools = self._lookup_tools(self.active_state.keys())
            allowed_ids = {
                capability_id(tool)
                for tool in current_active_tools
                if tool_is_allowed(tool, allowed_urls)
            }
            self.active_state = {
                identity: last_turn
                for identity, last_turn in self.active_state.items()
                if identity in allowed_ids
            }

        names = [capability_id(tool) for tool in retrieved_tools]
        active_state = update_active_tools(
            self.active_state,
            names,
            turn,
            ttl=self.ttl,
            max_active_tools=self.active_cap,
        )
        self.active_state = active_state
        tools = self._lookup_tools(active_state.keys())
        tools_by_id = {capability_id(tool): tool for tool in tools}
        retrieved_ids = list(dict.fromkeys(names))
        ordered_ids = retrieved_ids + [
            identity for identity in active_state if identity not in retrieved_ids
        ]
        tools = [tools_by_id[identity] for identity in ordered_ids if identity in tools_by_id]
        return RetrievalResult(
            tool_ids=[capability_id(tool) for tool in tools],
            tools=tools,
            metadata={
                "active_state": dict(active_state),
                "retrieved_ids": retrieved_ids,
            },
            retrieved_tools=retrieved_tools,
        )

    def reset(self) -> None:
        self.active_state = {}
