import json

from evals.toolret.adapter import ToolRetExample, adapt_toolret_tool, tool_embedding_text
from evals.toolret.metrics import evaluate_toolret_result


def test_toolret_adapter_preserves_id_and_schema():
    tool = adapt_toolret_tool(
        {
            "id": "web_tool_7",
            "documentation": json.dumps(
                {"name": "lookup", "description": "Find a record", "parameters": {"q": {"type": "string"}}}
            ),
        }
    )
    assert tool["id"] == "web_tool_7"
    assert tool["name"] == "web_tool_7"
    assert tool["toolret_name"] == "lookup"
    assert tool["input_schema"] == {"q": {"type": "string"}}


def test_toolret_embedding_text_excludes_parameters():
    tool = adapt_toolret_tool(
        {"id": "x", "doc": {"name": "lookup", "description": "Find a record", "parameters": {"secret": {}}}}
    )
    text = tool_embedding_text(tool)
    assert text == "lookup: Find a record"
    assert "secret" not in text


def test_toolret_label_matching_is_exact():
    metrics = evaluate_toolret_result(["tool_b", "tool_a"], {"tool_a": 1})
    assert metrics["recall@10"] == 1.0
    assert metrics["completeness@10"] == 1.0


def test_toolret_ndcg_known_order():
    first = evaluate_toolret_result(["a", "b"], {"a": 1, "b": 1})
    second = evaluate_toolret_result(["b", "a"], {"a": 1, "b": 1})
    assert first["ndcg@10"] == second["ndcg@10"] == 1.0


def test_toolret_natural_result_is_not_padded():
    metrics = evaluate_toolret_result(["a"], {"a": 1, "b": 1})
    assert metrics["recall@10"] == 0.5
    assert metrics["completeness@10"] == 0.0


def test_toolret_example_has_explicit_category_and_labels():
    example = ToolRetExample("q", "query", "instruction", {"tool": 1}, "web", "apibank")
    assert example.category == "web"
    assert example.labels == {"tool": 1}
