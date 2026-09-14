import spacy

nlp = spacy.load("en_core_web_trf")


def analyze_intents(text: str) -> list[dict]:
    original_text = text.strip()

    text = original_text

    if text:
        text = text[0].upper() + text[1:]

    doc = nlp(text)

    # for chunk in doc.noun_chunks:
    #     print(
    #         f"NOUN CHUNK -> text={chunk.text!r} "
    #         f"root={chunk.root.text!r} "
    #         f"dep={chunk.root.dep_!r}"
    #     )

    # for token in doc:
    #     print(
    #         f"text={token.text!r:12} "
    #         f"lemma={token.lemma_!r:12} "
    #         f"pos={token.pos_:6} "
    #         f"dep={token.dep_:10} "
    #         f"head={token.head.text!r}"
    #     )

    root = next(token for token in doc if token.dep_ == "ROOT")
    print("root: ", root)

    # The main action plus coordinated actions.
    action_tokens = [root]
    print("action_tokens: ", action_tokens)

    action_tokens.extend(
        token for token in doc if token.dep_ == "conj" and token.pos_ == "VERB"
    )

    action_tokens.extend(
        token
        for token in doc
        if token.dep_ == "dep"
        and token.pos_ == "VERB"
        and token.head == root
        and any(child.dep_ in {"dobj", "obj", "attr"} for child in token.children)
    )

    intents = []

    for action_token in action_tokens:
        target = None

        # First preference:
        # an object directly attached to this action.
        direct_targets = []

        for child in action_token.children:
            if child.dep_ in {"dobj", "obj", "attr"}:
                target_tokens = [
                    token.text for token in child.subtree if not token.is_punct
                ]

                direct_targets.append(" ".join(target_tokens))

        if direct_targets:
            target = " ".join(direct_targets)

        # Second preference:
        # If the action has no clean direct target,
        # inspect the part of the sentence belonging to this action.

        if target is None:
            action_index = action_token.i

            later_actions = [
                token.i for token in action_tokens if token.i > action_index
            ]

            end_index = min(later_actions) if later_actions else len(doc)

            candidates = [
                token
                for token in doc[action_index:end_index]
                if token.pos_ in {"NOUN", "PROPN"}
            ]

            if candidates:
                target = " ".join(token.text for token in candidates)

        intents.append(
            {
                "action": action_token.lemma_,
                "target": target,
                "context": original_text,
            }
        )

    return intents
