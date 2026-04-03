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
    "code"
]


def main():
    changer = TextChanger()

    for sentence in SENTENCES:
        lang = __import__("random").choice(LANGUAGES)
        print(f"=== [{lang.upper()}] ===")
        print(f"Original : {sentence}")
        print("Wrapped  :")
        # Run the language wrapper directly by name
        wrapper_fn = getattr(changer, f"_{lang}_adv")
        print(wrapper_fn(sentence))
        print()


if __name__ == "__main__":
    main()
