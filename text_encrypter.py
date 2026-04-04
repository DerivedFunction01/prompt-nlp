import base64
import codecs
import random
import string


class TextEncrypter:
    METHODS = ("base64", "rot13", "hex", "binary", "morse", "caesar")
    ENCODING_METHODS = ("base64", "hex", "binary")
    TEXT_METHODS = ("rot13", "morse", "caesar")

    MORSE_CODE_DICT = {
        "A": ".-",
        "B": "-...",
        "C": "-.-.",
        "D": "-..",
        "E": ".",
        "F": "..-.",
        "G": "--.",
        "H": "....",
        "I": "..",
        "J": ".---",
        "K": "-.-",
        "L": ".-..",
        "M": "--",
        "N": "-.",
        "O": "---",
        "P": ".--.",
        "Q": "--.-",
        "R": ".-.",
        "S": "...",
        "T": "-",
        "U": "..-",
        "V": "...-",
        "W": ".--",
        "X": "-..-",
        "Y": "-.--",
        "Z": "--..",
        "1": ".----",
        "2": "..---",
        "3": "...--",
        "4": "....-",
        "5": ".....",
        "6": "-....",
        "7": "--...",
        "8": "---..",
        "9": "----.",
        "0": "-----",
        ",": "--..--",
        ".": ".-.-.-",
        "?": "..--..",
        "/": "-..-.",
        "-": "-....-",
        "(": "-.--.",
        ")": "-.--.-",
        " ": "/",
    }

    @staticmethod
    def to_base64(text: str) -> str:
        return base64.b64encode(text.encode()).decode()

    @staticmethod
    def to_rot13(text: str) -> str:
        return codecs.encode(text, "rot_13")

    @staticmethod
    def to_hex(text: str) -> str:
        return text.encode().hex()

    @staticmethod
    def to_binary(text: str) -> str:
        return " ".join(format(ord(x), "08b") for x in text)

    @staticmethod
    def to_morse(text: str) -> str:
        return " ".join(
            TextEncrypter.MORSE_CODE_DICT.get(c.upper(), "") for c in text
        ).strip()

    @staticmethod
    def to_caesar(text: str, shift: int | None = None) -> str:
        # If no shift is provided, pick a random one that isn't 0 (mod 26)
        if shift is None:
            shift = random.randint(1, 25)

        result = ""
        for char in text:
            if char in string.ascii_letters:
                start = ord("a") if char.islower() else ord("A")
                result += chr((ord(char) - start + shift) % 26 + start)
            else:
                result += char
        return result

    @staticmethod
    def _count_character_changes(original: str, transformed: str) -> int:
        """
        Count how many characters were actually altered.

        We compare positionally up to the shared length and then count any extra
        transformed characters as changes. This works well for same-length
        transforms like rot13/caesar and for expanded encodings such as morse.
        """
        shared = sum(1 for a, b in zip(original, transformed) if a != b)
        return shared + abs(len(transformed) - len(original))

    def _count_morse_compatible_chars(self, text: str) -> int:
        return sum(
            1
            for char in text
            if char.strip() and char.upper() in self.MORSE_CODE_DICT
        )

    @staticmethod
    def _dedupe_methods(methods: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for method in methods:
            if method in seen:
                continue
            seen.add(method)
            ordered.append(method)
        return ordered

    def _should_prefer_encoding(self, text: str) -> bool:
        """
        Prefer encoding when ASCII letters are a minority of the visible text.

        This catches prompts that are mostly foreign-language characters,
        digits, or symbols, where rot13/caesar/morse are usually weak choices.
        """
        visible = [char for char in text if not char.isspace()]
        if not visible:
            return False

        ascii_letters = sum(1 for char in visible if char in string.ascii_letters)
        if ascii_letters == 0:
            return True

        return ascii_letters / len(visible) < 0.4

    def _attempt_encrypt(
        self,
        text: str,
        method: str,
        max_chars: int,
        min_changed_chars: int,
    ) -> str | None:
        methods = {
            "base64": self.to_base64,
            "rot13": self.to_rot13,
            "hex": self.to_hex,
            "binary": self.to_binary,
            "morse": self.to_morse,
            "caesar": self.to_caesar,
        }

        try:
            result = methods.get(method, lambda x: x)(text)
        except Exception:
            return None

        changed_chars = self._count_character_changes(text, result)
        if method == "morse":
            changed_chars = min(changed_chars, self._count_morse_compatible_chars(text))
        elif method in {"rot13", "caesar"}:
            changed_chars = min(
                changed_chars,
                sum(1 for char in text if char in string.ascii_letters),
            )

        if changed_chars < min_changed_chars:
            return None

        if len(result) > max_chars:
            result = result[:max_chars].rstrip()
        return result

    def encrypt_span(
        self,
        text: str,
        method: str | None = None,
        max_chars: int = 128,
        min_changed_chars: int = 1,
        return_method: bool = False,
    ) -> str | tuple[str, str]:
        """
        Encrypts and then truncates the result to max_chars.
        Returns the processed string.
        """
        prefer_encoding = self._should_prefer_encoding(text)
        if method is None:
            method = random.choice(self.ENCODING_METHODS if prefer_encoding else self.METHODS)

        if prefer_encoding and method not in self.ENCODING_METHODS:
            candidates = self._dedupe_methods(
                [
                    *self.ENCODING_METHODS,
                    *[name for name in self.METHODS if name not in self.ENCODING_METHODS],
                ]
            )
        else:
            candidates = [method, *[name for name in self.METHODS if name != method]]

        for candidate in candidates:
            result = self._attempt_encrypt(
                text,
                candidate,
                max_chars=max_chars,
                min_changed_chars=min_changed_chars,
            )
            if result is not None:
                return (result, candidate) if return_method else result

        return (text, method) if return_method else text
