"""Keep the primary documentation examples syntactically valid."""

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
DOCUMENTATION = (ROOT / "README.md", ROOT / "docs" / "production.md")


def _fenced_blocks(path: Path, language: str):
    text = path.read_text(encoding="utf-8")
    pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
    return re.findall(pattern, text, flags=re.DOTALL)


def test_documented_python_examples_parse():
    for path in DOCUMENTATION:
        for number, block in enumerate(_fenced_blocks(path, "python"), start=1):
            try:
                ast.parse(block, filename=f"{path}:{number}")
            except SyntaxError as exc:
                raise AssertionError(
                    f"Invalid Python example in {path}, block {number}: {exc}"
                ) from exc


def test_documented_json_examples_parse():
    for path in DOCUMENTATION:
        for number, block in enumerate(_fenced_blocks(path, "json"), start=1):
            try:
                json.loads(block)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"Invalid JSON example in {path}, block {number}: {exc}"
                ) from exc
