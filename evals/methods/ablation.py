from time import perf_counter

from brown_octopus.active_tools import update_active_tools
from brown_octopus.analyzer import analyze_intents
from brown_octopus.retriever import rank_tools
from brown_octopus.selection import select_bounded_max_gap
from brown_octopus.tool_registry import get_tools
from brown_octopus import Octopus

from evals.common.models import RetrievalResult


_INTENT_CACHE = {}
_RANKING_CACHE = {}


def _cached_intents(query: str):
    if query not in _INTENT_CACHE:
        _INTENT_CACHE[query] = analyze_intents(query)
    return _INTENT_CACHE[query]


def _cached_ranking(query: str):
    if query not in _RANKING_CACHE:
        _RANKING_CACHE[query] = rank_tools(query)
    return _RANKING_CACHE[query]


def _merge_unique(tool_groups: list[list[dict]]) -> list[dict]:
    selected = []
    seen = set()
    for group in tool_groups:
        for tool in group:
            if tool["name"] not in seen:
                selected.append(tool)
                seen.add(tool["name"])
    return selected


class EvaluationOctopusMethod:
    """Evaluation-only variants of the frozen Octopus pipeline."""

    def __init__(self, name: str, mode: str, fixed_k: int | None = None):
        self.name = name
        self.mode = mode
        self.fixed_k = fixed_k
        self.active_tools: dict[str, int] = {}

    def reset(self) -> None:
        self.active_tools = {}

    def _rank_and_select(self, query: str) -> tuple[list[dict], list[dict]]:
        if self.mode == "no_intent":
            intent_specs = [{"text": query, "action": None, "target": None}]
        else:
            intent_specs = _cached_intents(query)

        details = []
        groups = []
        for index, intent in enumerate(intent_specs, start=1):
            text = intent["text"]
            ranking = _cached_ranking(text)
            if self.mode == "fixed_k":
                selected = ranking[: self.fixed_k]
                cutoff_reason = f"fixed_k_{self.fixed_k}"
            else:
                selected = select_bounded_max_gap(ranking)
                cutoff_reason = "bounded_max_gap"
            groups.append(selected)
            details.append(
                {
                    "intent_number": index,
                    "action": intent.get("action"),
                    "target": intent.get("target"),
                    "text": text,
                    "ranking": ranking,
                    "cutoff": len(selected),
                    "cutoff_reason": cutoff_reason,
                    "selected_tool_ids": [tool["name"] for tool in selected],
                }
            )
        return _merge_unique(groups), details

    def retrieve(self, query: str, tools: list[dict], state=None) -> RetrievalResult:
        start = perf_counter()
        previous_active = set(self.active_tools)
        selected, intent_details = self._rank_and_select(query)
        selected_ids = [tool["name"] for tool in selected]

        if self.mode == "no_memory":
            active_state = {tool_id: 0 for tool_id in selected_ids}
            exposed_ids = selected_ids
        else:
            active_state = update_active_tools(
                active_tools=self.active_tools,
                retrieved_tools=selected_ids,
                turn=(max(self.active_tools.values(), default=0) + 1),
            )
            exposed_ids = [tool["name"] for tool in get_tools(active_state)]

        if self.mode == "no_memory":
            retained_ids = []
            expired_ids = []
        else:
            retained_ids = sorted((previous_active & set(active_state)) - set(selected_ids))
            expired_ids = sorted(previous_active - set(active_state))

        self.active_tools = active_state if self.mode != "no_memory" else {}
        scores = {
            tool["name"]: float(tool["score"])
            for tool in selected
            if "score" in tool
        }
        return RetrievalResult(
            tool_ids=exposed_ids,
            scores=scores or None,
            latency_ms=(perf_counter() - start) * 1000.0,
            metadata={
                "newly_retrieved_tool_ids": selected_ids,
                "retained_tool_ids": retained_ids,
                "expired_tool_ids": expired_ids,
                "active_state": active_state,
                "selected_before_memory_ids": selected_ids,
                "intent_details": intent_details,
                "memory_mode": self.mode != "no_memory",
            },
        )


class OctopusFullMethod(EvaluationOctopusMethod):
    """The frozen public Octopus pipeline, with evaluation metadata attached."""

    def __init__(self, octopus: Octopus):
        super().__init__("octopus_full", "bounded")
        self.octopus = octopus

    def reset(self) -> None:
        self.octopus.reset()

    def retrieve(self, query: str, tools: list[dict], state=None) -> RetrievalResult:
        start = perf_counter()
        previous_active = set(self.octopus.active_tools)
        _, intent_details = self._rank_and_select(query)
        production_result = self.octopus.process(query)
        selected_ids = [tool["name"] for tool in production_result["retrieved_tools"]]
        active_state = production_result["active_state"]
        exposed_ids = [tool["name"] for tool in production_result["tools"]]
        retained_ids = sorted((previous_active & set(active_state)) - set(selected_ids))
        expired_ids = sorted(previous_active - set(active_state))
        self.active_tools = active_state
        scores = {
            tool["name"]: float(tool["score"])
            for tool in production_result["retrieved_tools"]
            if "score" in tool
        }
        return RetrievalResult(
            tool_ids=exposed_ids,
            scores=scores or None,
            latency_ms=(perf_counter() - start) * 1000.0,
            metadata={
                "newly_retrieved_tool_ids": selected_ids,
                "retained_tool_ids": retained_ids,
                "expired_tool_ids": expired_ids,
                "active_state": active_state,
                "selected_before_memory_ids": selected_ids,
                "intent_details": intent_details,
                "memory_mode": True,
            },
        )
