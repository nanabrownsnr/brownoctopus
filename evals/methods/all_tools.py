from time import perf_counter

from evals.common.models import RetrievalResult


class AllTools:
    name = "all"

    def retrieve(self, query: str, tools: list[dict], state=None) -> RetrievalResult:
        start = perf_counter()
        tool_ids = [tool["name"] for tool in tools]
        return RetrievalResult(
            tool_ids=tool_ids,
            scores=None,
            latency_ms=(perf_counter() - start) * 1000.0,
            metadata={"condition": "maximum_context_reference"},
        )
