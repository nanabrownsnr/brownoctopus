import math
import re
from collections import Counter
from time import perf_counter

from evals.common.models import RetrievalResult


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25:
    def __init__(self, k: int):
        self.k = k
        self.name = f"bm25_{k}"
        self._tools = None
        self._documents = []
        self._frequencies = []
        self._idf = {}
        self._average_length = 0.0

    def initialize(self, tools: list[dict]) -> None:
        self._tools = tools
        self._documents = [
            tokenize(f"{tool['name']} {tool.get('description') or ''}")
            for tool in tools
        ]
        self._frequencies = [Counter(document) for document in self._documents]
        document_count = len(self._documents)
        document_frequency = Counter(
            token
            for document in self._documents
            for token in set(document)
        )
        self._idf = {
            token: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self._average_length = (
            sum(len(document) for document in self._documents) / document_count
            if document_count
            else 0.0
        )

    def retrieve(self, query: str, tools: list[dict], state=None) -> RetrievalResult:
        if self._tools is None:
            raise RuntimeError("BM25 is not initialized.")
        start = perf_counter()
        query_tokens = tokenize(query)
        query_terms = set(query_tokens)
        k1 = 1.5
        b = 0.75
        scores = []
        for index, frequencies in enumerate(self._frequencies):
            document_length = len(self._documents[index])
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                denominator = term_frequency + k1 * (
                    1 - b + b * document_length / self._average_length
                )
                score += self._idf.get(term, 0.0) * (
                    term_frequency * (k1 + 1) / denominator
                )
            scores.append(score)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], index),
        )[: self.k]
        tool_ids = [tools[index]["name"] for index in ranked_indices]
        score_map = {tool_id: scores[index] for tool_id, index in zip(tool_ids, ranked_indices)}
        return RetrievalResult(
            tool_ids=tool_ids,
            scores=score_map,
            latency_ms=(perf_counter() - start) * 1000.0,
            metadata={"k": self.k, "representation": "name + description"},
        )
