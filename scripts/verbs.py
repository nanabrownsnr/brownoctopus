import spacy


MODEL_NAME = "en_core_web_trf"


TEST_QUERIES = [
    "create a Word document summarizing the sprint",

    "search the web for the latest Nvidia news and send what you find to Tom",

    "check my leave balance, search the web for Nvidia news, and email the results to Tom",

    "get the sprint work items and create a Word document summarizing them",

    "email Tom a summary of the latest Nvidia news",

    "send my manager my remaining leave balance",

    "check my leave balance and if I have enough days submit a leave request",
]


def main():
    print(f"Loading spaCy model: {MODEL_NAME}")

    nlp = spacy.load(MODEL_NAME)

    for query in TEST_QUERIES:
        print("\n" + "=" * 100)
        print(f"QUERY: {query}")
        print("=" * 100)

        doc = nlp(query)

        print("\nALL TOKENS:")

        for token in doc:
            print(
                f"{token.text:<15} "
                f"lemma={token.lemma_:<15} "
                f"pos={token.pos_:<8} "
                f"dep={token.dep_:<12} "
                f"head={token.head.text}"
            )

        print("\nVERBS:")

        verbs = [
            token
            for token in doc
            if token.pos_ == "VERB"
        ]

        for number, verb in enumerate(verbs, start=1):
            children = [
                f"{child.text}({child.dep_})"
                for child in verb.children
            ]

            subtree = " ".join(
                token.text
                for token in verb.subtree
            )

            print(f"\nVERB {number}")
            print(f"  text:     {verb.text}")
            print(f"  lemma:    {verb.lemma_}")
            print(f"  dep:      {verb.dep_}")
            print(f"  head:     {verb.head.text}")
            print(f"  children: {children}")
            print(f"  subtree:  {subtree}")

        print("\nPOTENTIAL ACTIONS:")

        for verb in verbs:
            print(
                f"  {verb.text:<15} "
                f"→ {verb.lemma_}"
            )


if __name__ == "__main__":
    main()