from time import perf_counter

from brown_octopus.analyzer import analyze_intents
from brown_octopus.retriever import rank_tools
from brown_octopus.selection import select_bounded_max_gap
from brown_octopus.active_tools import update_active_tools
from brown_octopus.tool_registry import get_tools

from evals.common.models import RetrievalResult


RANK_CACHE: dict[str, list[dict]] = {}
INTENT_CACHE: dict[str, list[dict]] = {}


def cached_intents(query: str) -> list[dict]:
    if query not in INTENT_CACHE:
        INTENT_CACHE[query] = analyze_intents(query)
    return INTENT_CACHE[query]


def cached_ranking(query: str) -> list[dict]:
    if query not in RANK_CACHE:
        RANK_CACHE[query] = rank_tools(query)
    return RANK_CACHE[query]


def score_window(ranking: list[dict], cutoff: int) -> list[dict]:
    start = max(0, cutoff - 2)
    end = min(len(ranking), cutoff + 2)
    return ranking[start:end]


def bounded_max_gap(ranking: list[dict]) -> tuple[list[dict], dict]:
    selected = select_bounded_max_gap(ranking)
    return selected, {"reason": "bounded_max_gap", "cutoff": len(selected)}


def relative_threshold(
    ranking: list[dict],
    alpha: float,
    max_tools: int = 16,
) -> tuple[list[dict], dict]:
    if not ranking:
        return [], {"reason": "empty", "cutoff": 0, "alpha": alpha}
    top_score = ranking[0]["score"]
    cutoff = 1
    for tool in ranking[:max_tools]:
        if tool["score"] >= alpha * top_score:
            cutoff = tool["rank"]
        else:
            break
    selected = ranking[:cutoff]
    return selected, {
        "reason": "relative_score_threshold",
        "cutoff": cutoff,
        "alpha": alpha,
        "threshold_score": alpha * top_score,
    }


def ranked_score_knee(
    ranking: list[dict],
    max_tools: int = 16,
) -> tuple[list[dict], dict]:
    if not ranking:
        return [], {"reason": "empty", "cutoff": 0}
    window = ranking[: max_tools + 1]
    if len(window) < 3:
        cutoff = min(max_tools, len(window))
        return ranking[:cutoff], {"reason": "short_window", "cutoff": cutoff}
    scores = [tool["score"] for tool in window]
    low = min(scores)
    high = max(scores)
    if high == low:
        cutoff = min(max_tools, len(window))
        return ranking[:cutoff], {"reason": "flat_curve", "cutoff": cutoff}
    distances = []
    last_index = len(window) - 1
    for index, score in enumerate(scores):
        x = index / last_index
        y = (score - low) / (high - low)
        baseline = 1.0 - x
        distances.append(y - baseline)
    knee_index = max(range(1, len(window) - 1), key=lambda index: distances[index])
    cutoff = min(max_tools, max(1, knee_index + 1))
    selected = ranking[:cutoff]
    return selected, {
        "reason": "ranked_score_knee",
        "cutoff": cutoff,
        "knee_distance": distances[knee_index],
    }


class SelectionMethod:
    def __init__(self, name: str, selector, selector_metadata: dict | None = None):
        self.name = name
        self.selector = selector
        self.selector_metadata = selector_metadata or {}
        self.active_tools: dict[str, int] = {}

    def reset(self) -> None:
        self.active_tools = {}

    def _select_for_query(self, query: str) -> tuple[list[dict], list[dict]]:
        intents = cached_intents(query)
        selected_groups = []
        details = []
        for intent_number, intent in enumerate(intents, start=1):
            ranking = cached_ranking(intent["text"])
            selected, selection = self.selector(ranking)
            details.append(
                {
                    "intent_number": intent_number,
                    "action": intent.get("action"),
                    "target": intent.get("target"),
                    "text": intent["text"],
                    "ranking": ranking,
                    "cutoff": selection["cutoff"],
                    "selection": {**self.selector_metadata, **selection},
                    "selected_tool_ids": [tool["name"] for tool in selected],
                    "scores_around_cutoff": score_window(ranking, selection["cutoff"]),
                }
            )
            selected_groups.append(selected)

        merged = []
        seen = set()
        for group in selected_groups:
            for tool in group:
                if tool["name"] not in seen:
                    merged.append(tool)
                    seen.add(tool["name"])
        return merged, details

    def retrieve(self, query: str, tools: list[dict], state=None) -> RetrievalResult:
        start = perf_counter()
        previous = set(self.active_tools)
        selected, details = self._select_for_query(query)
        selected_ids = [tool["name"] for tool in selected]
        turn = max(self.active_tools.values(), default=0) + 1
        active = update_active_tools(self.active_tools, selected_ids, turn)
        exposed = [tool["name"] for tool in get_tools(active)]
        retained = sorted((previous & set(active)) - set(selected_ids))
        expired = sorted(previous - set(active))
        self.active_tools = active
        return RetrievalResult(
            tool_ids=exposed,
            scores={tool["name"]: float(tool["score"]) for tool in selected} or None,
            latency_ms=(perf_counter() - start) * 1000.0,
            metadata={
                "newly_retrieved_tool_ids": selected_ids,
                "retained_tool_ids": retained,
                "expired_tool_ids": expired,
                "active_state": active,
                "selected_before_memory_ids": selected_ids,
                "intent_details": details,
            },
        )
