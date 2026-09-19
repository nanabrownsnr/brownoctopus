import pytest

from octopus import Octopus

from pathlib import Path


@pytest.mark.anyio
async def test_octopus_initialize_rejects_unsupported_index_version(
    monkeypatch,
    tmp_path,
):
    index_path = tmp_path / "index"
    index_path.mkdir()

    (index_path / "tools.json").write_text("[]")
    (index_path / "embeddings.pt").touch()
    (index_path / "metadata.json").write_text('{"version": 999}')

    monkeypatch.setattr(
        "octopus.octopus.initialize_analyzer",
        lambda: None,
    )

    monkeypatch.setattr(
        "octopus.octopus.initialize_retriever",
        lambda: None,
    )

    octopus = Octopus(
        index_path=index_path,
    )

    with pytest.raises(
        RuntimeError,
        match="Unsupported Octopus index version",
    ):
        await octopus.initialize()


@pytest.mark.anyio
async def test_octopus_update_saves_index_metadata(
    monkeypatch,
    tmp_path,
):
    tools = [
        {"name": "tool_a", "description": "A"},
        {"name": "tool_b", "description": "B"},
    ]

    saved_metadata = {}

    monkeypatch.setattr(
        "octopus.octopus.initialize_analyzer",
        lambda: None,
    )

    monkeypatch.setattr(
        "octopus.octopus.initialize_retriever",
        lambda: None,
    )

    monkeypatch.setattr(
        "octopus.octopus.load_mcp_urls",
        lambda path: ["fake-mcp"],
    )

    async def fake_discover_universe(urls):
        return tools

    monkeypatch.setattr(
        "octopus.octopus.discover_universe",
        fake_discover_universe,
    )

    monkeypatch.setattr(
        "octopus.octopus.set_tools",
        lambda tools: None,
    )

    monkeypatch.setattr(
        "octopus.octopus.refresh_index",
        lambda: "fake-embeddings",
    )

    monkeypatch.setattr(
        "octopus.octopus.save_tools",
        lambda path, tools: None,
    )

    monkeypatch.setattr(
        "octopus.octopus.save_embeddings",
        lambda path, embeddings: None,
    )

    def fake_save_metadata(path, metadata):
        saved_metadata.update(metadata)

    monkeypatch.setattr(
        "octopus.octopus.save_metadata",
        fake_save_metadata,
    )

    octopus = Octopus(
        index_path=tmp_path / "index",
        catalog_path=tmp_path / "mcps.json",
    )

    await octopus.update()

    assert saved_metadata == {
        "version": 1,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "tool_count": 2,
    }


@pytest.mark.anyio
async def test_octopus_initialize_fails_clearly_when_index_is_missing(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "octopus.octopus.initialize_analyzer",
        lambda: None,
    )

    monkeypatch.setattr(
        "octopus.octopus.initialize_retriever",
        lambda: None,
    )

    octopus = Octopus(
        index_path=tmp_path / "missing-index",
    )

    with pytest.raises(
        RuntimeError,
        match="Octopus index not found",
    ):
        await octopus.initialize()


@pytest.mark.anyio
async def test_octopus_update_initializes_models(
    monkeypatch,
    tmp_path,
):
    octopus = Octopus(
        index_path=tmp_path / "index",
        catalog_path=tmp_path / "mcps.json",
    )

    initialized = {
        "analyzer": False,
        "retriever": False,
    }

    def fake_initialize_analyzer():
        initialized["analyzer"] = True

    def fake_initialize_retriever():
        initialized["retriever"] = True

    monkeypatch.setattr(
        "octopus.octopus.initialize_analyzer",
        fake_initialize_analyzer,
    )

    monkeypatch.setattr(
        "octopus.octopus.initialize_retriever",
        fake_initialize_retriever,
    )

    monkeypatch.setattr(
        "octopus.octopus.load_mcp_urls",
        lambda path: [],
    )

    async def fake_discover_universe(urls):
        return []

    monkeypatch.setattr(
        "octopus.octopus.discover_universe",
        fake_discover_universe,
    )

    monkeypatch.setattr(
        "octopus.octopus.set_tools",
        lambda tools: None,
    )

    monkeypatch.setattr(
        "octopus.octopus.refresh_index",
        lambda: None,
    )

    monkeypatch.setattr(
        "octopus.octopus.save_tools",
        lambda path, tools: None,
    )

    monkeypatch.setattr(
        "octopus.octopus.save_embeddings",
        lambda path, embeddings: None,
    )

    await octopus.update()

    assert initialized["analyzer"] is True
    assert initialized["retriever"] is True


@pytest.mark.anyio
async def test_octopus_update_rebuilds_and_saves_index(
    monkeypatch,
    tmp_path,
):
    catalog_path = tmp_path / "mcps.json"
    index_path = tmp_path / "index"

    octopus = Octopus(
        index_path=index_path,
        catalog_path=catalog_path,
    )

    mcp_urls = [
        "https://example.com/web/mcp",
        "https://example.com/email/mcp",
    ]

    discovered_tools = [
        {
            "name": "web_search",
            "description": "Search the web",
        },
        {
            "name": "send_email",
            "description": "Send an email",
        },
    ]

    built_embeddings = object()

    monkeypatch.setattr(
        "octopus.octopus.load_mcp_urls",
        lambda path: mcp_urls,
    )

    async def fake_discover_universe(urls):
        assert urls == mcp_urls
        return discovered_tools

    monkeypatch.setattr(
        "octopus.octopus.discover_universe",
        fake_discover_universe,
    )

    registered_tools = []

    def fake_set_tools(tools):
        registered_tools.extend(tools)

    monkeypatch.setattr(
        "octopus.octopus.set_tools",
        fake_set_tools,
    )

    monkeypatch.setattr(
        "octopus.octopus.refresh_index",
        lambda: built_embeddings,
    )

    saved = {}

    def fake_save_tools(path, tools):
        saved["tools_path"] = path
        saved["tools"] = tools

    def fake_save_embeddings(path, embeddings):
        saved["embeddings_path"] = path
        saved["embeddings"] = embeddings

    monkeypatch.setattr(
        "octopus.octopus.save_tools",
        fake_save_tools,
    )

    monkeypatch.setattr(
        "octopus.octopus.save_embeddings",
        fake_save_embeddings,
    )

    result = await octopus.update()

    assert result == discovered_tools

    assert registered_tools == discovered_tools

    assert saved["tools_path"] == index_path
    assert saved["tools"] == discovered_tools

    assert saved["embeddings_path"] == index_path
    assert saved["embeddings"] is built_embeddings


def test_octopus_accepts_catalog_path():
    octopus = Octopus(catalog_path="data/custom_mcps.json")

    assert octopus.catalog_path == Path("data/custom_mcps.json")


def test_octopus_accepts_index_path():
    octopus = Octopus(index_path="data/indexes/default")

    assert octopus.index_path == Path("data/indexes/default")


def test_octopus_starts_with_empty_conversation_state():
    octopus = Octopus()

    assert octopus.turn == 0
    assert octopus.active_tools == {}


def test_octopus_processes_turn_and_updates_its_state(monkeypatch):
    octopus = Octopus()

    def fake_process_turn(query, active_tools, turn):
        assert query == "check my leave"
        assert active_tools == {}
        assert turn == 1

        return {
            "retrieved_tools": [{"name": "check_leave"}],
            "active_state": {"check_leave": 1},
            "tools": [{"name": "check_leave"}],
        }

    monkeypatch.setattr(
        "octopus.octopus.process_turn",
        fake_process_turn,
    )

    result = octopus.process("check my leave")

    assert octopus.turn == 1
    assert octopus.active_tools == {"check_leave": 1}

    assert result["tools"] == [{"name": "check_leave"}]


def test_octopus_carries_state_across_turns(monkeypatch):
    octopus = Octopus()

    calls = []

    def fake_process_turn(query, active_tools, turn):
        calls.append(
            {
                "query": query,
                "active_tools": active_tools.copy(),
                "turn": turn,
            }
        )

        if turn == 1:
            return {
                "retrieved_tools": [{"name": "check_leave"}],
                "active_state": {"check_leave": 1},
                "tools": [{"name": "check_leave"}],
            }

        return {
            "retrieved_tools": [{"name": "send_email"}],
            "active_state": {
                "check_leave": 1,
                "send_email": 2,
            },
            "tools": [
                {"name": "check_leave"},
                {"name": "send_email"},
            ],
        }

    monkeypatch.setattr(
        "octopus.octopus.process_turn",
        fake_process_turn,
    )

    octopus.process("check my leave")
    result = octopus.process("email it to Tom")

    assert calls[0] == {
        "query": "check my leave",
        "active_tools": {},
        "turn": 1,
    }

    assert calls[1] == {
        "query": "email it to Tom",
        "active_tools": {"check_leave": 1},
        "turn": 2,
    }

    assert octopus.turn == 2
    assert octopus.active_tools == {
        "check_leave": 1,
        "send_email": 2,
    }

    assert result["tools"] == [
        {"name": "check_leave"},
        {"name": "send_email"},
    ]


def test_octopus_reset_clears_conversation_state():
    octopus = Octopus()

    octopus.turn = 5
    octopus.active_tools = {
        "check_leave": 2,
        "send_email": 5,
    }

    octopus.reset()

    assert octopus.turn == 0
    assert octopus.active_tools == {}


def test_octopus_retrieve_returns_tools_without_changing_state(
    monkeypatch,
):
    octopus = Octopus()

    expected_tools = [
        {"name": "web_search"},
        {"name": "send_email"},
    ]

    def fake_retrieve_tools(query):
        assert query == ("Find the latest Nvidia news and email Tom")
        return expected_tools

    monkeypatch.setattr(
        "octopus.octopus.retrieve_tools",
        fake_retrieve_tools,
    )

    result = octopus.retrieve("Find the latest Nvidia news and email Tom")

    assert result == expected_tools

    assert octopus.turn == 0
    assert octopus.active_tools == {}


@pytest.mark.anyio
async def test_octopus_can_initialize_index_created_by_update(
    monkeypatch,
    tmp_path,
):
    import torch

    index_path = tmp_path / "index"

    tools = [
        {
            "name": "tool_a",
            "description": "Tool A",
        },
        {
            "name": "tool_b",
            "description": "Tool B",
        },
    ]

    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    monkeypatch.setattr(
        "octopus.octopus.initialize_analyzer",
        lambda: None,
    )

    monkeypatch.setattr(
        "octopus.octopus.initialize_retriever",
        lambda: None,
    )

    monkeypatch.setattr(
        "octopus.octopus.load_mcp_urls",
        lambda path: ["fake-mcp"],
    )

    async def fake_discover_universe(urls):
        return tools

    monkeypatch.setattr(
        "octopus.octopus.discover_universe",
        fake_discover_universe,
    )

    monkeypatch.setattr(
        "octopus.octopus.refresh_index",
        lambda: embeddings,
    )

    builder = Octopus(
        index_path=index_path,
        catalog_path=tmp_path / "mcps.json",
    )

    await builder.update()

    loader = Octopus(
        index_path=index_path,
    )

    loaded_tools = await loader.initialize()

    assert loaded_tools == tools
