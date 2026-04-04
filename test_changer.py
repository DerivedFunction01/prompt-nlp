from text_composer import TextChanger


SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "All systems are operating within normal parameters.",
    "User authentication failed after three attempts.",
    "The database connection was successfully established.",
    "Please review the attached report before Friday.",
]

def main():
    changer = TextChanger()

    for sentence in SENTENCES:
        print("=== [RANDOM PLAN] ===")
        print(f"Original : {sentence}")
        rendered = changer.compose(sentence, seed=42)
        print("Rendered :")
        print(rendered["text"])
        print(f"Span     : {rendered['span']}")
        print(f"Method   : {rendered['method']}")
        start, end = rendered["span"] # type: ignore
        print(f"Payload  : {rendered['text'][start:end]}") # type: ignore
        print()


if __name__ == "__main__":
    main()
