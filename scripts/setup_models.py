"""Download/check the frozen Brown Octopus models in the active environment."""

import spacy
from sentence_transformers import SentenceTransformer


def main() -> None:
    nlp = spacy.load("en_core_web_trf")
    print(f"spaCy en_core_web_trf loaded: {nlp.meta.get('version', 'unknown')}")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    print(f"Qwen/Qwen3-Embedding-0.6B loaded on: {model.device}")


if __name__ == "__main__":
    main()
