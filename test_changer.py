from text_changer import TextChanger


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
        rendered = changer.render_with_span(lang, sentence)
        print("Wrapped  :")
        print(rendered["text"])
        print(f"Span     : {rendered['span']}")
        print()


if __name__ == "__main__":
    main()
