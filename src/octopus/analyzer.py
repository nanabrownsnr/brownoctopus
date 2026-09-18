import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token


SPACY_MODEL_NAME = "en_core_web_trf"
OBJECT_DEPENDENCIES = {"dobj", "obj", "attr"}

nlp: Language | None = None


def initialize_analyzer() -> None:
    """Load the spaCy model once."""
    global nlp

    if nlp is None:
        nlp = spacy.load(SPACY_MODEL_NAME)


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

    action_tokens = _find_action_tokens(doc)

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
                "context": original_text,
            }
        )

    return intents
