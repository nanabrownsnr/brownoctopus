import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

from brown_octopus.model_store import register_managed_spacy_plugins, resolve_spacy_model


SPACY_MODEL_NAME = "en_core_web_trf"
OBJECT_DEPENDENCIES = {"dobj", "obj", "attr"}

nlp: Language | None = None


class DeterministicIntentAnalyzer:
    """Interface adapter for the current deterministic spaCy analyzer."""

    name = "spacy_deterministic"

    def analyze(self, text: str) -> list[dict]:
        return analyze_intents(text)


def initialize_analyzer() -> None:
    """Load the spaCy model once."""
    global nlp

    if nlp is None:
        register_managed_spacy_plugins()
        try:
            # spaCy uses CuPy for GPU execution. If the compatible CuPy
            # runtime is absent, prefer_gpu() safely returns False and the
            # analyzer remains on CPU.
            spacy.prefer_gpu()
        except Exception:
            pass
        nlp = spacy.load(resolve_spacy_model())


def _has_direct_object(token: Token) -> bool:
    return any(child.dep_ in OBJECT_DEPENDENCIES for child in token.children)


def _find_action_tokens(doc: Doc) -> list[Token]:
    root = next(token for token in doc if token.dep_ == "ROOT")

    action_tokens = [root]

    action_tokens.extend(
        token for token in doc if token.dep_ == "conj" and token.pos_ == "VERB"
    )

    action_tokens.extend(
        token
        for token in doc
        if token.dep_ == "dep"
        and token.pos_ == "VERB"
        and token.head == root
        and _has_direct_object(token)
    )

    action_tokens.extend(
        token
        for token in doc
        if token.dep_ == "acl" and token.pos_ == "VERB" and _has_direct_object(token)
    )

    return sorted(
        set(action_tokens),
        key=lambda token: token.i,
    )


def _belongs_to_nested_action(
    token: Token,
    action_token: Token,
    action_tokens: list[Token],
) -> bool:
    action_subtree = set(action_token.subtree)

    return any(
        other_action != action_token
        and other_action in action_subtree
        and (token == other_action or other_action in token.ancestors)
        for other_action in action_tokens
    )


def _get_local_text(
    action_token: Token,
    action_tokens: list[Token],
) -> str:
    local_tokens = []

    for token in action_token.subtree:
        if token.is_punct:
            continue

        if token.dep_ == "cc":
            continue

        if _belongs_to_nested_action(
            token,
            action_token,
            action_tokens,
        ):
            continue

        local_tokens.append(token)

    local_tokens.sort(key=lambda token: token.i)

    return " ".join(token.text for token in local_tokens)


def _get_direct_target(
    action_token: Token,
    action_tokens: list[Token],
) -> str | None:
    targets = []

    for child in action_token.children:
        if child.dep_ not in OBJECT_DEPENDENCIES:
            continue

        target_tokens = []

        for token in child.subtree:
            if token.is_punct:
                continue

            if _belongs_to_nested_action(
                token,
                action_token,
                action_tokens,
            ):
                continue

            target_tokens.append(token)

        target_tokens.sort(key=lambda token: token.i)

        target_text = " ".join(token.text for token in target_tokens).strip()

        if target_text:
            targets.append(target_text)

    if not targets:
        return None

    return " ".join(targets)


def _get_core_resource(action_token: Token) -> str | None:
    """Return the core grammatical resource operated on by an action.

    This intentionally uses only dependency structure.  Task-specific
    arguments such as names, titles, and dates remain in the normal intent
    fields but do not become part of the capability retrieval query.
    """

    def collect_compounds(token: Token) -> list[Token]:
        compounds: list[Token] = []
        for child in token.children:
            if child.dep_ == "compound":
                compounds.extend(collect_compounds(child))
                compounds.append(child)
        return compounds

    for child in action_token.children:
        if child.dep_ not in OBJECT_DEPENDENCIES:
            continue

        tokens = collect_compounds(child) + [child]
        tokens.sort(key=lambda token: token.i)
        resource = " ".join(token.text for token in tokens).strip()
        if resource:
            return resource

    # Handle constructions such as "reply to the email".
    for child in action_token.children:
        if child.dep_ not in {"prep", "dative"}:
            continue

        for token in child.subtree:
            if token.dep_ != "pobj" or token.pos_ not in {"NOUN", "PROPN"}:
                continue

            tokens = collect_compounds(token) + [token]
            tokens.sort(key=lambda item: item.i)
            resource = " ".join(item.text for item in tokens).strip()
            if resource:
                return resource

    return None


def _get_retrieval_text(action_token: Token) -> str:
    """Build the compact capability-oriented query for one intent."""

    resource = _get_core_resource(action_token)
    if not resource:
        return action_token.lemma_
    return f"{action_token.lemma_} {resource}"


def _get_fallback_target(
    doc: Doc,
    action_token: Token,
    action_tokens: list[Token],
) -> str | None:
    action_index = action_token.i

    later_actions = [token.i for token in action_tokens if token.i > action_index]

    end_index = min(later_actions) if later_actions else len(doc)

    candidates = [
        token
        for token in doc[action_index:end_index]
        if token.pos_ in {"NOUN", "PROPN"}
    ]

    if not candidates:
        return None

    return " ".join(token.text for token in candidates)


def analyze_intents(text: str) -> list[dict]:
    """
    Decompose a user request into operational intents.
    """
    if nlp is None:
        raise RuntimeError(
            "Analyzer is not initialized. "
            "Call initialize() before processing requests."
        )

    original_text = text.strip()

    if not original_text:
        return []

    normalized_text = original_text[0].upper() + original_text[1:]

    doc = nlp(normalized_text)

    try:
        action_tokens = _find_action_tokens(doc)
    except StopIteration:
        # Some short, mathematical, or malformed benchmark strings can be
        # parsed without a ROOT. Preserve the request as one opaque intent so
        # retrieval remains available instead of crashing the service.
        return [
            {
                "action": "request",
                "target": None,
                "text": original_text,
                "retrieval_text": original_text,
                "context": original_text,
            }
        ]

    if not action_tokens:
        return [
            {
                "action": "request",
                "target": None,
                "text": original_text,
                "retrieval_text": original_text,
                "context": original_text,
            }
        ]

    intents = []

    for action_token in action_tokens:
        target = _get_direct_target(
            action_token,
            action_tokens,
        )

        if target is None:
            target = _get_fallback_target(
                doc,
                action_token,
                action_tokens,
            )

        intents.append(
            {
                "action": action_token.lemma_,
                "target": target,
                "text": _get_local_text(
                    action_token,
                    action_tokens,
                ),
                "retrieval_text": _get_retrieval_text(action_token),
                "context": original_text,
            }
        )

    return intents
