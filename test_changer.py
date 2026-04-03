from text_composer import TextChanger


SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "All systems are operating within normal parameters.",
    "User authentication failed after three attempts.",
    "The database connection was successfully established.",
    "Please review the attached report before Friday.",
]

LANGUAGES = [
    "python",
    "javascript",
    "typescript",
    "java",
    "csharp",
    "go",
    "ruby",
    "rust",
    "php",
    "sql",
    "xml",
    "bash",
    "json",
]


def main():
    changer = TextChanger()

    for sentence in SENTENCES:
        lang = __import__("random").choice(LANGUAGES)
        print(f"=== [{lang.upper()}] ===")
        print(f"Original : {sentence}")
        rendered = changer.compose(
            sentence,
            mutation_profile=[
                {
                    "method": "accent_mutation",
                    "chance": 1.0,
                    "params": {
                        "token_prob": 0.6,
                        "char_prob": 0.4,
                        "max_char_mutation_ratio": 0.33,
                    },
                }
            ],
            encryption_method="rot13",
            code_method=lang,
            seed=42,
        )
        print("Wrapped  :")
        print(rendered["text"])
        print(f"Span     : {rendered['span']}")
        start, end = rendered["span"] # type: ignore
        print(f"Payload  : {rendered['text'][start:end]}") # type: ignore
        print()


if __name__ == "__main__":
    main()
