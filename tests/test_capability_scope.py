import torch

import brown_octopus.retriever as retriever
from brown_octopus.context_manager import ActiveCapabilityContext
from brown_octopus.pipeline import CapabilityPipeline


def test_rank_tools_preserves_complete_metadata_and_prefilters_before_ranking(monkeypatch):
    tools = [
        {
            "capability_id": "alpha:search",
            "name": "alpha_search",
            "description": "Alpha search",
            "mcp_url": "https://alpha.example/mcp",
            "input_schema": {"type": "object"},
        },
        {
            "capability_id": "beta:search",
            "name": "beta_search",
            "description": "Beta search",
            "mcp_url": "https://beta.example/mcp/",
            "input_schema": {"type": "object", "properties": {"q": {}}},
        },
    ]

    class FakeModel:
        def encode(self, query, **kwargs):
            return torch.tensor([1.0, 0.0])

    monkeypatch.setattr(retriever, "model", FakeModel())
    monkeypatch.setattr(retriever, "TOOLS", tools)
    monkeypatch.setattr(
        retriever,
        "tool_embeddings",
        torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0],
        ]),
    )

    ranked = retriever.rank_tools(
        "search",
        allowed_mcp_urls=["https://beta.example/mcp"],
    )

    assert [tool["capability_id"] for tool in ranked] == ["beta:search"]
    assert ranked[0]["rank"] == 1
    assert ranked[0]["mcp_url"] == "https://beta.example/mcp/"
    assert ranked[0]["input_schema"]["properties"]["q"] == {}
    assert "alpha:search" not in [tool["capability_id"] for tool in ranked]


def test_allowlist_is_applied_to_active_context_and_retained_tools():
    tools = [
        {
            "capability_id": "alpha:tool",
            "name": "alpha_tool",
            "mcp_url": "https://alpha.example/mcp",
        },
        {
            "capability_id": "beta:tool",
            "name": "beta_tool",
            "mcp_url": "https://beta.example/mcp",
        },
    ]
    by_id = {tool["capability_id"]: tool for tool in tools}

    class Analyzer:
        def analyze(self, query):
            return [{"text": query}]

    class Retriever:
        def retrieve(self, query, intent, allowed_mcp_urls=None):
            allowed = set(allowed_mcp_urls or [])
            return [
                dict(tool, rank=1, score=1.0)
                for tool in tools
                if tool["mcp_url"] in allowed
            ]

    class Selector:
        name = "brown_octopus_v3"

        def select(self, query, intent, candidates):
            return candidates

    context = ActiveCapabilityContext(
        lambda ids: [by_id[identity] for identity in ids if identity in by_id],
        ttl=8,
        active_cap=30,
    )
    pipeline = CapabilityPipeline(Analyzer(), Retriever(), Selector(), context)

    first = pipeline.process(
        "use alpha",
        turn=1,
        allowed_mcp_urls=["https://alpha.example/mcp"],
    )
    second = pipeline.process(
        "use beta",
        turn=2,
        allowed_mcp_urls=["https://beta.example/mcp"],
    )

    assert [tool["capability_id"] for tool in first["retrieved_tools"]] == [
        "alpha:tool"
    ]
    assert [tool["capability_id"] for tool in second["retrieved_tools"]] == [
        "beta:tool"
    ]
    assert [tool["capability_id"] for tool in second["active_tools"]] == [
        "beta:tool"
    ]


def test_missing_allowlist_preserves_existing_retrieval_call_shape(monkeypatch):
    calls = []

    monkeypatch.setattr(
        retriever,
        "rank_tools",
        lambda query: calls.append(query) or [],
    )

    retriever.QwenCandidateRetriever().retrieve(
        "original",
        {"text": "capability"},
    )

    assert calls == ["capability"]
