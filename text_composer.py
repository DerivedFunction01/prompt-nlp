from __future__ import annotations

import random

from text_formatter import TextFormatter
from text_encrypter import TextEncrypter
from text_mutator import MutationOrchestrator


class TextChanger:
    def __init__(self) -> None:
        self.formatter = TextFormatter()
        self.mutator = MutationOrchestrator()
        self.encrypter = TextEncrypter()

    def mutate(
        self,
        text: str,
        profile: list[dict] | None = None,
        seed: int | None = None,
    ) -> str:
        return self.mutator.mutate(text, profile=profile, seed=seed)

    def encrypt(
        self,
        text: str,
        method: str | None = None,
        max_chars: int = 128,
    ) -> str:
        return self.encrypter.encrypt_span(text, method=method, max_chars=max_chars)

    def code_format(self, method: str, text: str) -> dict[str, object]:
        return self.formatter.code_format(method, text)

    def compose(
        self,
        text: str,
        *,
        mutation_profile: list[dict] | None = None,
        encryption_method: str | None = None,
        code_method: str | None = None,
        seed: int | None = None,
        max_chars: int = 128,
    ) -> dict[str, object]:
        """
        Apply mutation, optional encryption, and optional code formatting.

        The returned span refers to the final wrapped payload when a code method
        is supplied. Without a code wrapper, the span is the whole payload.
        """
        payload = text

        if seed is not None:
            random.seed(seed)

        explicit_plan = any(
            value is not None
            for value in (mutation_profile, encryption_method, code_method)
        )

        if explicit_plan:
            if mutation_profile is not None and encryption_method is None:
                payload = self.mutate(payload, profile=mutation_profile, seed=seed)

            if encryption_method is not None:
                payload = self.encrypt(
                    payload,
                    method=encryption_method,
                    max_chars=max_chars,
                )

            if code_method is not None:
                return self.formatter.code_format(code_method, payload)

            return {"text": payload, "span": (0, len(payload)), "method": None}

        selected_ops = [
            op
            for op in ("mutation", "encryption", "formatting")
            if random.random() < 0.5
        ]
        if not selected_ops:
            selected_ops = [random.choice(("mutation", "encryption", "formatting"))]

        if "encryption" in selected_ops and "mutation" in selected_ops:
            selected_ops = [op for op in selected_ops if op != "mutation"]

        if "mutation" in selected_ops:
            payload = self.mutate(payload, profile=None, seed=seed)

        if "encryption" in selected_ops:
            payload = self.encrypt(payload, method=None, max_chars=max_chars)

        if "formatting" in selected_ops:
            return self.formatter.random_code_format(payload)

        return {"text": payload, "span": (0, len(payload)), "method": None}
