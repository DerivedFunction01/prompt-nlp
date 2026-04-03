import base64
import codecs
import random


class TextEncrypter:

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
    def to_caesar(text: str, shift: int = 3) -> str:
        result = ""
        for char in text:
            if char.isalpha():
                start = ord("a") if char.islower() else ord("A")
                result += chr((ord(char) - start + shift) % 26 + start)
            else:
                result += char
        return result

    def encrypt_span(self, text: str, method: str, max_chars: int = 128) -> str:
        """
        Encrypts and then truncates the result to max_chars.
        Returns the processed string.
        """
        methods = {
            "base64": self.to_base64,
            "rot13": self.to_rot13,
            "hex": self.to_hex,
            "binary": self.to_binary,
            "morse": self.to_morse,
            "caesar": self.to_caesar,
        }

        # Execute selected method
        try:
            result = methods.get(method, lambda x: x)(text)
        except Exception:
            result = text  # Fallback to original on error

        # Apply strict truncation
        if len(result) > max_chars:
            result = result[:max_chars].rstrip()

        return result
