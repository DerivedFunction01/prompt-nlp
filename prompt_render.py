from __future__ import annotations

import random
import re
from collections.abc import Sequence

from labels import PromptEntity
from faker import Faker
import randomname




class PromptEditor:
    """
    Higher-level text editor that applies category-aware transformations.

    """

    DEFAULT_MODEL_NAMES = [
        "ChatGPT",
        "Claude",
        "Gemini",
        "Grok",
        "LLaMA",
        "Mistral",
        "Falcon",
        "Cohere",
        "HuggingChat",
        "Phi",
        "Gemma",
        "DeepSeek",
        "Qwen",
        "Nemotron",
        "MiniMax",
        "oLMO",
        "GLM",
        "Pangu",
        "Command-R"
    ]

    DEFAULT_MODEL_ORGS = {
        "chatgpt": "OpenAI",
        "claude": "Anthropic",
        "gemini": "Google",
        "grok": "xAI",
        "llama": "Meta",
        "mistral": "Mistral AI",
        "falcon": "TII",
        "cohere": "Cohere",
        "huggingchat": "Hugging Face",
        "phi": "Microsoft",
        "gemma": "Google",
        "deepseek": "DeepSeek",
        "qwen": "Alibaba",
        "nemotron": "NVIDIA",
        "minimax": "MiniMax",
        "olmo": "oLMO",
        "glm": "GLM",
        "pangu": "Pangu",
        "command-r": "Cohere"
    }

    ORGANIZATION_SEEDS = [
        "OpenAI",
        "Anthropic",
        "Google",
        "Meta",
        "Microsoft",
        "xAI",
        "Alibaba",
        "NVIDIA",
        "Mistral AI",
        "Cohere",
        "Hugging Face",
        "DeepMind",
        "TII",
        "AI2",
        "AI21 Labs",
        "Databricks",
        "Nomic AI",
        "SambaNova",
        "Reka AI",
        "Adept AI",
        "Inflection AI",
        "EleutherAI",
        "LAION",
        "Stanford CRFM",
        "Berkeley AI Research",
        "MIT CSAIL",
        "Carnegie Mellon LTI",
        "Tsinghua KEG Lab",
        "AMD",
        "Intel",
        "Qualcomm",
        "Cerebras",
        "Graphcore",
        "Oracle",
        "IBM",
        "Snowflake",
        "Salesforce",
        "OpenRouter",
        "Red Hat",
        "Apache",
        "Linux",
        "Huawei",
        "iFlyTek",
        "SenseTime",
        "Zhipu AI",
        "ByteDance",
        "JD.com",
        "360 AI",
        "Kunlun Xin",
    ]

    ORGANIZATION_SUFFIXES = [
        "AI",
        "Labs",
        "Research",
        "Systems",
        "Studio",
        "Works",
        "Foundation",
        "Technologies",
        "Cloud",
        "Compute",
        "Intelligence",
        "Research Lab",
    ]

    NSHOT_HUMAN_ROLE_POOL = [
        "Human",
        "User",
        "Input",
        "Text",
        "Question",
        "Q",
        "Prompt",
        "Instruction",
        "Query",
    ]

    NSHOT_ASSISTANT_ROLE_POOL = [
        "AI",
        "Assistant",
        "Output",
        "Answer",
        "Solution",
        "Model",
        "Bot",
        "RAesponder",
    ]

    HUMAN_MARKERS = [
        r"\[HUMAN\]:?",
    ]

    ASSISTANT_MARKERS = [
        r"\[ASSISTANT\]:?",
    ]

    PERSONA_PREFIX_POOL = [
        "You are",
        "Act as",
        "Assume the role of",
        "Take on the role of",
        "Take on the persona of",
        "Respond as",
        "Reply as",
        "Present yourself as",
        "Imagine you are",
        "From now on, you are",
        "For this conversation, you are",
        "In this scenario, you are",
        "You are acting as",
        "You will behave as",
        "You are speaking as",
    ]

    PERSONA_NEGATED_PREFIX_POOL = [
        "You are not",
        "Do not act as",
        "Do not assume the role of",
        "Do not take on the role of",
        "Do not take on the persona of",
        "Do not respond as",
        "Do not reply as",
        "Do not present yourself as",
        "Do not imagine you are",
        "From now on, you are not",
        "For this conversation, you are not",
        "In this scenario, you are not",
        "You are not acting as",
        "You will not behave as",
        "You are not speaking as",
    ]

    
    PERSONA_MODEL_STYLE_POOL = [
        "prefix",
        "is_a",
        "as_you_are",
    ]

    PERSONA_COMPACT_CUE_POOL = [
        "Also",
        "Next",
        "Then",
        "Additionally",
        "Furthermore",
        "Moreover",
        "Plus",
    ]

    MODEL_IDENTITY_FORMAT_POOL = [
        "name_only",
        "from_org",
        "org_possessive",
    ]

    MODEL_VARIANT_SUFFIXES = ["mini", "small", "large", "plus", "pro", "lite", "thinking"]
    MODEL_FICTIONAL_SUFFIXES = ["Chat", "AI", "ML", "LM", "LLM", "Bot", "GPT"]
    MODEL_VARIANT_SIZE_UNITS = ["M", "B", "T"]
    MODEL_VARIANT_ORG_CONNECTORS = ["from", "at", "by"]
    MODEL_COMPACT_DROP_TOKENS = {
        "mini",
        "small",
        "large",
        "plus",
        "pro",
        "lite",
        "thinking",
        "preview",
        "beta",
        "alpha",
        "rc",
        "v",
    }

    def __init__(
        self,
        texts: Sequence[str] | None = None,
        categories: Sequence[PromptEntity | str] | None = None,
        model_name: str | None = None,
        model_org: str | None = None,
        allow_empty_model_name: bool = False,
        use_random_model_name_for_empty_model_name: bool = False,
        random_model_name_chance: float = 0.5,
        use_random_model_org_for_empty_model_name: bool = False,
        random_model_org_chance: float = 0.5,
        use_model_org_variants_for_empty_model_name: bool = False,
        model_org_variant_chance: float = 0.5,
        use_model_name_variants_for_empty_model_name: bool = False,
        model_name_variant_chance: float = 0.5,
        use_faker_for_empty_model_name: bool = False,
        faker_name_chance: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.texts = list(texts) if texts is not None else []
        self.categories = list(categories) if categories is not None else []
        self.seed = seed
        self.rng = random.Random(seed)
        self.allow_empty_model_name = allow_empty_model_name
        self.use_random_model_name_for_empty_model_name = (
            use_random_model_name_for_empty_model_name
        )
        self.random_model_name_chance = random_model_name_chance
        self.use_random_model_org_for_empty_model_name = (
            use_random_model_org_for_empty_model_name
        )
        self.random_model_org_chance = random_model_org_chance
        self.use_model_org_variants_for_empty_model_name = (
            use_model_org_variants_for_empty_model_name
        )
        self.model_org_variant_chance = model_org_variant_chance
        self.use_model_name_variants_for_empty_model_name = (
            use_model_name_variants_for_empty_model_name
        )
        self.model_name_variant_chance = model_name_variant_chance
        self.use_faker_for_empty_model_name = use_faker_for_empty_model_name
        self.faker_name_chance = faker_name_chance
        self._faker = Faker() if Faker is not None else None
        if self._faker is not None and seed is not None:
            self._faker.seed_instance(seed)
        if model_name is not None:
            self.model_name = model_name
        elif allow_empty_model_name:
            self.model_name = ""
        else:
            self.model_name = self._dummy_model_name()
        self.model_org = model_org or self._infer_model_org(self.model_name)

    def _dummy_model_name(self) -> str:
        return self.rng.choice(self.DEFAULT_MODEL_NAMES)

    def _dummy_model_org(self) -> str | None:
        return self.rng.choice(self.ORGANIZATION_SEEDS) if self.ORGANIZATION_SEEDS else None

    def _randomname_fragment(self) -> str | None:
        if randomname is None:
            return None
        raw = randomname.get_name()
        parts = [part for part in raw.split("-") if part]
        if not parts:
            return None
        return self.rng.choice(parts).title()

    def _faker_name_fragment(self) -> str | None:
        if self._faker is None:
            return None
        raw = self._faker.first_name()
        if not raw:
            return None
        return re.sub(r"[^A-Za-z]", "", raw).title() or None

    def _fictional_model_stem(self) -> str | None:
        candidates = [
            self._faker_name_fragment(),
            self._randomname_fragment(),
        ]
        candidates = [candidate for candidate in candidates if candidate]
        if candidates:
            return self.rng.choice(candidates)
        return None

    def _fictional_model_name(self) -> str:
        stem = self._fictional_model_stem()
        if not stem:
            stem = self._dummy_model_name()
        suffix = self.rng.choice(self.MODEL_FICTIONAL_SUFFIXES)
        return f"{stem}{suffix}"

    def _generate_version_tag(self) -> str:
        """
        Build a compact version tag such as v3, 5.2, -4q, or -5.1.
        """
        pattern = self.rng.choice(
            ["v_major", "v_major_minor", "major_alpha", "dash_major_minor", "dash_major_alpha"]
        )
        major = self.rng.randint(2, 9)
        minor = self.rng.randint(0, 9)
        alpha = self.rng.choice("abcdefghijklmnopqrstuvwxyz")

        if pattern == "v_major":
            return f"v{major}"
        if pattern == "v_major_minor":
            return f"v{major}.{minor}"
        if pattern == "major_alpha":
            return f"{major}{alpha}"
        if pattern == "dash_major_minor":
            return f"-{major}.{minor}"
        return f"-{major}{alpha}"

    def _variantize_model_name(self, base_name: str) -> str:
        base_name = base_name.strip()
        if not base_name:
            return base_name

        mode = self.rng.choice(["version", "size", "suffix", "random_word"])
        separator = self.rng.choice(["-", " "])

        if mode == "version":
            tag = self._generate_version_tag()
            if tag.startswith("-"):
                return f"{base_name}{tag}"
            return f"{base_name}{separator}{tag}"
        if mode == "size":
            magnitude = self.rng.choice([1, 3, 7, 10, 13, 22, 34, 70, 110])
            if self.rng.random() < 0.35:
                size = f"{self.rng.randint(1, 99)}.{self.rng.randint(0, 9)}{self.rng.choice(self.MODEL_VARIANT_SIZE_UNITS)}"
            else:
                size = f"{magnitude}{self.rng.choice(self.MODEL_VARIANT_SIZE_UNITS)}"
            return f"{base_name}{separator}{size}"
        if mode == "suffix":
            return f"{base_name}{separator}{self.rng.choice(self.MODEL_VARIANT_SUFFIXES)}"

        word = self._fictional_model_stem()
        if word:
            return f"{base_name}{separator}{word}"
        return f"{base_name}{separator}{self.rng.choice(self.MODEL_VARIANT_SUFFIXES)}"

    def _variantize_model_org(self, base_org: str | None = None) -> str | None:
        org = (base_org or self._dummy_model_org() or "").strip()
        if not org:
            return None

        mode = self.rng.choice(["seed", "suffix", "word", "possessive"])
        if mode == "seed":
            return org
        if mode == "suffix":
            return f"{org} {self.rng.choice(self.ORGANIZATION_SUFFIXES)}"
        if mode == "word":
            fragment = self._randomname_fragment()
            if fragment:
                return f"{org} {fragment}"
            return f"{org} {self.rng.choice(self.ORGANIZATION_SUFFIXES)}"

        owner = self.rng.choice(["AI", "Labs", "Research", "Systems", "Network"])
        return f"{org}'s {owner}"

    def _fake_persona_name(self) -> str:
        if self._faker is None:
            raise RuntimeError("Faker is not installed.")
        return self._faker.name()

    @staticmethod
    def _strip_leading_article(text: str) -> str:
        return re.sub(r"^(?:a|an)\s+", "", text, flags=re.IGNORECASE).strip()

    @staticmethod
    def _capitalize_leading_article(text: str) -> str:
        if not text:
            return text
        return re.sub(
            r"^(a|an)\b",
            lambda m: m.group(1).capitalize(),
            text,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _split_leading_article(text: str) -> tuple[str, str]:
        cleaned = text.strip()
        match = re.match(r"^(a|an)\b\s*(.*)$", cleaned, flags=re.IGNORECASE)
        if not match:
            return "", cleaned
        return match.group(1).lower(), match.group(2).strip()

    @classmethod
    def _compact_model_name(cls, model_name: str | None) -> str:
        if not model_name:
            return ""

        name = re.sub(r"[\s\-_]+", " ", str(model_name).strip())
        parts = [part for part in name.split() if part]
        if not parts:
            return ""

        kept: list[str] = []
        for part in parts:
            lower = part.casefold().strip(".")
            if re.fullmatch(r"(?:v)?\d+(?:\.\d+)*", lower):
                break
            if lower in cls.MODEL_COMPACT_DROP_TOKENS:
                break
            if re.fullmatch(r"\d+[a-z]?", lower):
                break
            kept.append(part)

        if not kept:
            kept = [parts[0]]
        return " ".join(kept).strip()

    @staticmethod
    def _guess_indefinite_article(text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "a"
        return "an" if re.match(r"^[aeiou]", cleaned, flags=re.IGNORECASE) else "a"

    def _format_persona_clause(
        self,
        text: str,
        *,
        compact: bool = False,
        negated: bool = False,
        drop_model_version: bool = False,
    ) -> str:
        if compact:
            cue = self.rng.choice(self.PERSONA_COMPACT_CUE_POOL)
            article, persona_text = self._split_leading_article(text)
            if not persona_text:
                persona_text = text.strip()
            if self.model_name:
                identity, _ = self._format_model_identity()
                if drop_model_version:
                    identity = self._compact_model_name(identity)
                article_out = article or self._guess_indefinite_article(persona_text)
                copula = "is not" if negated else "is"
                return f"{cue}, {identity} {copula} {article_out} {persona_text}"
            article_out = article or self._guess_indefinite_article(persona_text)
            copula = "are not" if negated else "are"
            return f"{cue}, you {copula} {article_out} {persona_text}"

        return self._capitalize_leading_article(text.strip())

    def _format_model_identity(
        self,
        model_name: str | None = None,
        model_org: str | None = None,
        style: str | None = None,
    ) -> tuple[str, str]:
        name = (model_name or self.model_name or "").strip()
        org = (model_org or self.model_org or "").strip()
        chosen_style = style or self.rng.choice(self.MODEL_IDENTITY_FORMAT_POOL)
        org_redundant = bool(org) and org.casefold() == name.casefold()

        if not name:
            return "", "empty"

        if chosen_style == "org_possessive" and org and not org_redundant:
            return f"{org}'s {name}", chosen_style
        if chosen_style == "from_org" and org and not org_redundant:
            return f"{name} from {org}", chosen_style
        return name, "name_only"

    def _infer_model_org(self, model_name: str | None) -> str | None:
        if not model_name:
            return None
        key = self._model_key(model_name)
        return self.DEFAULT_MODEL_ORGS.get(key) if key else ""

    def _model_key(self, model_name: str | None, model_org: str | None = None) -> str | None:
        haystack = " ".join(
            part.lower() for part in (model_name or "", model_org or "") if part
        )
        for key in self.DEFAULT_MODEL_ORGS:
            if key in haystack:
                return key
        return None

    @staticmethod
    def _normalize_category(category: PromptEntity | str) -> str:
        if isinstance(category, PromptEntity):
            return category.value
        return str(category).upper()

    @staticmethod
    def _decorate_role(role: str, style: str = "colon") -> str:
        if style == "brackets":
            return f"[{role}]"
        if style == "arrow":
            return f"{role} >"
        if style == "speaker":
            return f"{role}:"
        if style == "plain":
            return role
        return f"{role}:"

    def _pick_nshot_variants(self) -> tuple[str, str]:
        human_role = self.rng.choice(self.NSHOT_HUMAN_ROLE_POOL)
        assistant_role = self.rng.choice(self.NSHOT_ASSISTANT_ROLE_POOL)
        return human_role, assistant_role

    @classmethod
    def _replace_markers(cls, text: str, patterns: list[str], replacement: str) -> str:
        result = text

        for pattern in patterns:
            regex = re.compile(pattern, flags=re.IGNORECASE)
            result = regex.sub(replacement, result)
        return result

    def _compose_nshot(self, text: str) -> tuple[str, str]:
        human_role, assistant_role = self._pick_nshot_variants()
        human_style = self.rng.choice(["colon", "brackets", "speaker", "plain", "arrow"])
        assistant_style = self.rng.choice(["colon", "brackets", "speaker", "plain", "arrow"])
        human_variant = self._decorate_role(human_role, style=human_style)
        assistant_variant = self._decorate_role(assistant_role, style=assistant_style)

        if assistant_role == "Model":
            assistant_variant = self._decorate_role(self.model_name, style=assistant_style)

        transformed = text
        transformed = self._replace_markers(transformed, self.HUMAN_MARKERS, human_variant)
        transformed = self._replace_markers(
            transformed,
            self.ASSISTANT_MARKERS,
            assistant_variant,
        )
        return transformed, f"{human_role}->{assistant_role} ({human_style}/{assistant_style})"

    def _persona_prefix(self, *, negated: bool = False) -> str:
        pool = self.PERSONA_NEGATED_PREFIX_POOL if negated else self.PERSONA_PREFIX_POOL
        return self.rng.choice(pool)

    def _compose_persona(
        self,
        text: str,
        *,
        compact: bool = False,
        negated: bool = False,
        drop_model_version: bool = False,
    ) -> str:
        if compact:
            return self._format_persona_clause(
                text,
                compact=True,
                negated=negated,
                drop_model_version=drop_model_version,
            )

        prefix = self._persona_prefix(negated=negated)

        if self.model_name:
            if negated:
                identity, _ = self._format_model_identity()
                if drop_model_version:
                    identity = self._compact_model_name(identity)
                article, persona_text = self._split_leading_article(text)
                if not persona_text:
                    persona_text = text.strip()
                article_out = article or self._guess_indefinite_article(persona_text)
                if self.rng.random() < 0.5:
                    return f"{identity} is not {article_out} {persona_text}"
                return f"As {identity}, you are not {article_out} {persona_text}"

            style = self.rng.choice(self.PERSONA_MODEL_STYLE_POOL)
            identity, _ = self._format_model_identity()
            if drop_model_version:
                identity = self._compact_model_name(identity)
            if style == "is_a":
                article, persona_text = self._split_leading_article(text)
                if not persona_text:
                    persona_text = text.strip()
                article_out = article or self._guess_indefinite_article(persona_text)
                if negated:
                    return f"{identity} is not {article_out} {persona_text}"
                return f"{identity} is {article_out} {persona_text}"
            if style == "as_you_are":
                article, persona_text = self._split_leading_article(text)
                if not persona_text:
                    persona_text = text.strip()
                article_out = article or self._guess_indefinite_article(persona_text)
                copula = "are not" if negated else "are"
                return f"As {identity}, you {copula} {article_out} {persona_text}"
            copula = "are not" if negated else "are"
            return f"{prefix} {identity}, {self._capitalize_leading_article(text.strip())}"

        if negated:
            article, persona_text = self._split_leading_article(text)
            if not persona_text:
                persona_text = text.strip()
            article_out = article or self._guess_indefinite_article(persona_text)
            return f"{prefix} {article_out} {persona_text}"

        return f"{prefix} {self._capitalize_leading_article(text.strip())}"

    def _resolve_empty_model_identity(self) -> None:
        if self.model_name:
            return
        if not self.allow_empty_model_name:
            self.model_name = self._dummy_model_name()
            self.model_org = self.model_org or self._infer_model_org(self.model_name)
            return

        if self.use_random_model_name_for_empty_model_name and (
            self.rng.random() < self.random_model_name_chance
        ):
            base_name = self._dummy_model_name()
            if (
                self.use_model_name_variants_for_empty_model_name
                and self.rng.random() < self.model_name_variant_chance
            ):
                self.model_name = self._variantize_model_name(base_name)
            else:
                self.model_name = base_name
            if self.use_random_model_org_for_empty_model_name and (
                self.rng.random() < self.random_model_org_chance
            ):
                base_org = self._infer_model_org(self.model_name) or self._dummy_model_org()
                if (
                    self.use_model_org_variants_for_empty_model_name
                    and self.rng.random() < self.model_org_variant_chance
                ):
                    self.model_org = self._variantize_model_org(base_org)
                else:
                    self.model_org = base_org
            else:
                self.model_org = self.model_org or self._infer_model_org(self.model_name)
            return

        if (
            self.use_faker_for_empty_model_name
            and self._faker is not None
            and self.rng.random() < self.faker_name_chance
        ):
            self.model_name = self._fictional_model_name()
            if self.use_random_model_org_for_empty_model_name and (
                self.rng.random() < self.random_model_org_chance
            ):
                base_org = self._dummy_model_org()
                if (
                    self.use_model_org_variants_for_empty_model_name
                    and self.rng.random() < self.model_org_variant_chance
                ):
                    self.model_org = self._variantize_model_org(base_org)
                else:
                    self.model_org = base_org
            return

    def _broadcast_categories(
        self,
        texts: Sequence[str],
        categories: Sequence[PromptEntity | str],
    ) -> list[PromptEntity | str]:
        if len(categories) == len(texts):
            return list(categories)
        if len(categories) == 1 and len(texts) > 1:
            return list(categories) * len(texts)
        raise ValueError(
            "categories must either match texts one-to-one or contain a single shared category"
        )

    def compose(
        self,
        texts: Sequence[str] | None = None,
        categories: Sequence[PromptEntity | str] | None = None,
        model_name: str | None = None,
        model_org: str | None = None,
        persona_compact: bool = False,
        persona_negated: bool = False,
        persona_compact_drop_model_version: bool = False,
    ) -> list[dict[str, object]]:
        """
        Transform each text according to its category.

        Returns a list of dictionaries with the original text, transformed text,
        selected branch, and model metadata.
        """
        working_texts = list(texts) if texts is not None else list(self.texts)
        working_categories = list(categories) if categories is not None else list(self.categories)

        if not working_texts:
            return []
        if not working_categories:
            raise ValueError("categories must be provided")

        if model_name is not None:
            self.model_name = model_name
        if model_org is not None:
            self.model_org = model_org
        self._resolve_empty_model_identity()
        if self.model_org is None:
            self.model_org = self._infer_model_org(self.model_name)

        resolved_categories = self._broadcast_categories(working_texts, working_categories)
        results: list[dict[str, object]] = []

        for text, category in zip(working_texts, resolved_categories, strict=True):
            normalized = self._normalize_category(category)

            if normalized == PromptEntity.NSHOT.value:
                transformed, variant = self._compose_nshot(text)
                branch = "nshot"
            elif normalized == PromptEntity.PERSONA.value:
                transformed = self._compose_persona(
                    text,
                    compact=persona_compact,
                    negated=persona_negated,
                    drop_model_version=persona_compact_drop_model_version,
                )
                variant = "persona_prefix"
                branch = "persona"
            else:
                transformed = text
                variant = "passthrough"
                branch = "passthrough"

            results.append(
                {
                    "original_text": text,
                    "text": transformed,
                    "category": normalized,
                    "branch": branch,
                    "variant": variant,
                    "model_name": self.model_name,
                    "model_org": self.model_org,
                }
            )

        return results

    def compose_texts(
        self,
        texts: Sequence[str] | None = None,
        categories: Sequence[PromptEntity | str] | None = None,
        model_name: str | None = None,
        model_org: str | None = None,
    ) -> list[str]:
        """Convenience wrapper that returns only transformed text values."""
        return [item["text"] for item in self.compose(texts, categories, model_name, model_org) if isinstance(item["text"], str)]
