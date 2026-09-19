from time import perf_counter

from octopus import Octopus

from evals.common.models import RetrievalResult


class OctopusMethod:
    name = "brown_octopus_v3"

    def __init__(self, octopus: Octopus):
        self.octopus = octopus

    def reset(self) -> None:
        self.octopus.reset()

    def retrieve(self, query: str, tools: list[dict], state=None) -> RetrievalResult:
        start = perf_counter()
        result = self.octopus.process(query)
        exposed_ids = [tool["name"] for tool in result["tools"]]
        retrieved = result["retrieved_tools"]
        scores = {
            tool["name"]: float(tool["score"])
            for tool in retrieved
            if "score" in tool
        }
        return RetrievalResult(
            tool_ids=exposed_ids,
            scores=scores or None,
            latency_ms=(perf_counter() - start) * 1000.0,
            metadata={
                "newly_retrieved_tool_ids": [tool["name"] for tool in retrieved],
                "active_state": result["active_state"],
                "selection": "min_4_bounded_max_gap",
                "method": self.name,
                "min_tools": 4,
                "max_tools": 16,
                "min_gap_percent": 2.0,
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "tool_representation": "name + description",
                "intent_analyzer": "spaCy en_core_web_trf",
                "index_version": 1,
                "ttl": 8,
                "active_cap": 30,
            },
        )
