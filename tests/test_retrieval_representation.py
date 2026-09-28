from dataclasses import dataclass, field

import brown_octopus.analyzer as analyzer
from brown_octopus.retriever import QwenCandidateRetriever, retrieve_tools


@dataclass
class FakeToken:
    text: str
    lemma_: str
    dep_: str
    pos_: str = "NOUN"
    i: int = 0
    children: list = field(default_factory=list)
    subtree: list = field(default_factory=list)


def resource_action(action, resource, *, compounds=()):
    resource_token = FakeToken(
        text=resource,
        lemma_=resource.lower(),
        dep_="obj",
        i=1,
    )
    compound_tokens = [
        FakeToken(text=word, lemma_=word.lower(), dep_="compound", i=index)
        for index, word in enumerate(compounds)
    ]
    resource_token.children = compound_tokens
    action_token = FakeToken(
        text=action,
        lemma_=action.lower(),
        dep_="ROOT",
        pos_="VERB",
        i=0,
        children=[resource_token],
    )
    return action_token


def test_core_resource_builds_generic_action_resource_representation():
    cases = [
        ("create", "document", ("Word",), "create Word document"),
        ("create", "issue", (), "create issue"),
        ("create", "record", ("customer",), "create customer record"),
        ("update", "record", ("employee",), "update employee record"),
        ("schedule", "meeting", (), "schedule meeting"),
        ("search", "repositories", ("GitHub",), "search GitHub repositories"),
    ]

    for action, resource, compounds, expected in cases:
        token = resource_action(action, resource, compounds=compounds)
        assert analyzer._get_retrieval_text(token) == expected


def test_prepositional_resource_is_supported():
    email = FakeToken(
        text="email",
        lemma_="email",
        dep_="pobj",
        pos_="NOUN",
        i=2,
    )
    preposition = FakeToken(
        text="to",
        lemma_="to",
        dep_="prep",
        i=1,
        children=[email],
    )
    preposition.subtree = [preposition, email]
    action = FakeToken(
        text="reply",
        lemma_="reply",
        dep_="ROOT",
        pos_="VERB",
        children=[preposition],
    )

    assert analyzer._get_retrieval_text(action) == "reply email"


def test_analyzer_adds_retrieval_text_without_changing_existing_fields(monkeypatch):
    action = resource_action("Create", "document", compounds=("Word",))

    class FakeNLP:
        def __call__(self, text):
            return object()

    monkeypatch.setattr(analyzer, "nlp", FakeNLP())
    monkeypatch.setattr(analyzer, "_find_action_tokens", lambda doc: [action])
    monkeypatch.setattr(
        analyzer,
        "_get_direct_target",
        lambda action_token, action_tokens: "a Word document titled NVIDIA News Summary",
    )
    monkeypatch.setattr(
        analyzer,
        "_get_local_text",
        lambda action_token, action_tokens: "Create a Word document titled NVIDIA News Summary",
    )

    result = analyzer.analyze_intents("Create a Word document titled NVIDIA News Summary")

    assert result == [
        {
            "action": "create",
            "target": "a Word document titled NVIDIA News Summary",
            "text": "Create a Word document titled NVIDIA News Summary",
            "retrieval_text": "create Word document",
            "context": "Create a Word document titled NVIDIA News Summary",
        }
    ]


def test_fallback_intent_uses_original_query_as_retrieval_text(monkeypatch):
    class FakeNLP:
        def __call__(self, text):
            return object()

    monkeypatch.setattr(analyzer, "nlp", FakeNLP())
    monkeypatch.setattr(analyzer, "_find_action_tokens", lambda doc: [])

    result = analyzer.analyze_intents("please do the thing")

    assert result[0]["retrieval_text"] == "please do the thing"


def test_candidate_retriever_prefers_retrieval_text(monkeypatch):
    queries = []
    monkeypatch.setattr(
        "brown_octopus.retriever.rank_tools",
        lambda query: queries.append(query) or [],
    )

    QwenCandidateRetriever().retrieve(
        "original query",
        {"retrieval_text": "compact capability query", "text": "full clause"},
    )

    assert queries == ["compact capability query"]


def test_candidate_retriever_fallback_order(monkeypatch):
    queries = []
    monkeypatch.setattr(
        "brown_octopus.retriever.rank_tools",
        lambda query: queries.append(query) or [],
    )
    retriever = QwenCandidateRetriever()

    retriever.retrieve("original", {"text": "full clause"})
    retriever.retrieve("original", {})

    assert queries == ["full clause", "original"]


def test_legacy_retrieve_tools_uses_retrieval_text(monkeypatch):
    intents = [{"retrieval_text": "create Word document", "text": "full clause"}]
    queries = []
    monkeypatch.setattr("brown_octopus.retriever.analyze_intents", lambda query: intents)
    monkeypatch.setattr(
        "brown_octopus.retriever.rank_tools",
        lambda query: queries.append(query) or [],
    )

    assert retrieve_tools("original") == []
    assert queries == ["create Word document"]
