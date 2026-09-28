from brown_octopus.active_tools import update_active_tools
from brown_octopus.context_manager import ActiveCapabilityContext
from brown_octopus.retriever import retrieve_tools
from brown_octopus.tool_registry import get_tools
from brown_octopus.tool_registry import capability_id
import logging
import time


logger = logging.getLogger("brown_octopus.pipeline")


class CapabilityPipeline:
    """Harness-agnostic coordinator for analysis, retrieval, selection and state."""

    def __init__(self, analyzer, candidate_retriever, selector, context_manager=None) -> None:
        self.analyzer = analyzer
        self.candidate_retriever = candidate_retriever
        self.selector = selector
        self.context_manager = context_manager

    def _retrieve_current(self, query: str) -> dict:
        """Run only the current-turn V3 retrieval and selection stages."""
        started = time.perf_counter()
        analysis_started = started
        intents = self.analyzer.analyze(query)
        analysis_ms = (time.perf_counter() - analysis_started) * 1000
        selected = []
        seen = set()
        per_intent = []
        retrieval_started = time.perf_counter()
        for intent in intents:
            candidates = self.candidate_retriever.retrieve(query, intent)
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
            selection_started = time.perf_counter()
            chosen = self.selector.select(query, intent, candidates)
            selection_ms = (time.perf_counter() - selection_started) * 1000
            per_intent.append({
                "intent": intent,
                "candidate_count": len(candidates),
                "selected_count": len(chosen),
                "retrieval_ms": retrieval_ms,
                "selection_ms": selection_ms,
            })
            for tool in chosen:
                identity = capability_id(tool)
                if identity not in seen:
                    seen.add(identity)
                    selected.append(tool)
        return {
            "retrieved_tools": selected,
            "intents": intents,
            "per_intent": per_intent,
            "strategy": getattr(self.selector, "name", type(self.selector).__name__),
            "timing_ms": {
                "analysis": analysis_ms,
                "retrieval_and_selection": (time.perf_counter() - retrieval_started) * 1000,
                "total": (time.perf_counter() - started) * 1000,
            },
        }

    def retrieve(
        self,
        query: str,
        turn: int | None = None,
        context_manager: ActiveCapabilityContext | None = None,
        session_id: str = "default",
    ) -> dict:
        """Retrieve capabilities, optionally updating active context.

        The production path supplies ``turn`` and receives both the raw
        current-turn selection and the final active context. Omitting it keeps
        a retrieval-only diagnostic path for internal callers.
        """
        retrieval = self._retrieve_current(query)
        if turn is None:
            return retrieval
        return self._with_active_context(retrieval, turn, context_manager, session_id)

    def _with_active_context(
        self,
        retrieval: dict,
        turn: int,
        context_manager: ActiveCapabilityContext | None = None,
        session_id: str = "default",
    ) -> dict:
        context_manager = context_manager or self.context_manager
        if context_manager is None:
            raise RuntimeError("A session context manager is required for stateful retrieval")
        register_tools = getattr(context_manager, "register_tools", None)
        if register_tools is not None:
            register_tools(retrieval["retrieved_tools"])
        context = context_manager.update(retrieval["retrieved_tools"], turn)
        logger.info(
            "capability_context_updated",
            extra={
                "event_data": {
                    "turn": turn,
                    "session_id": session_id,
                    "strategy": retrieval["strategy"],
                    "retrieved_count": len(retrieval["retrieved_tools"]),
                    "exposed_count": len(context.tools),
                    "timing_ms": retrieval.get("timing_ms", {}),
                }
            },
        )
        return {
            **retrieval,
            "active_state": context.metadata["active_state"],
            "active_tools": context.tools,
            "tools": context.tools,
            "tool_ids": context.tool_ids,
            "timing_ms": retrieval.get("timing_ms", {}),
        }

    def process(
        self,
        query: str,
        turn: int,
        context_manager: ActiveCapabilityContext | None = None,
        session_id: str = "default",
    ) -> dict:
        return self.retrieve(query, turn, context_manager, session_id)

    def reset(self) -> None:
        if self.context_manager is not None:
            self.context_manager.reset()


def process_turn(
    query: str,
    active_tools: dict[str, int],
    turn: int,
) -> dict:
    """
    Build the tool context for a single agent turn.

    The current query is used to retrieve relevant tools.
    Those tools are merged with the active tool working set,
    and the resulting tool definitions are returned for the
    agent to use.
    """
    retrieved_tools = retrieve_tools(query)

    retrieved_tool_names = [tool["name"] for tool in retrieved_tools]

    active_state = update_active_tools(
        active_tools=active_tools,
        retrieved_tools=retrieved_tool_names,
        turn=turn,
    )

    active_tools = get_tools(active_state.keys())

    return {
        "retrieved_tools": retrieved_tools,
        "active_state": active_state,
        "tools": active_tools,
    }
