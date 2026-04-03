import logging
import random
import string
from functools import lru_cache

log = logging.getLogger(__name__)

import nltk
from nltk.corpus import words as nltk_words

# ---------------------------------------------------------------------------
# Dictionary word set — short words only (len <= 6) to catch collision risk.
# Long words are unlikely to accidentally swap/dup/delete into another word.
# ---------------------------------------------------------------------------


def _build_dict() -> frozenset[str]:
    try:
        word_list = nltk_words.words()
    except LookupError:
        nltk.download("words", quiet=True)
        word_list = nltk_words.words()
    return frozenset(w.lower() for w in word_list if len(w) <= 6)


_DICT: frozenset[str] = _build_dict()

_STRIP_CHARS = string.punctuation + string.whitespace


@lru_cache(maxsize=8192)
def _is_dict_word(token: str) -> bool:
    return token.strip(_STRIP_CHARS).lower() in _DICT


@lru_cache(maxsize=8192)
def _spelling_eligible(token: str) -> bool:
    """Token must be purely alphabetic (after stripping punctuation) and meet
    the minimum length for at least one strategy."""
    stem = token.strip(_STRIP_CHARS)
    return stem.isalpha() and len(stem) >= 3  # 3 is the lowest gate (duplicate)


# ---------------------------------------------------------------------------
# Lookup tables (replaces NLPAug OCR augmenter)
# ---------------------------------------------------------------------------

OCR_MAP: dict[str, set[str]] = {
    "0": {"o", "O"},
    "1": {"l", "I", "i"},
    "3": {"e", "E"},
    "4": {"a", "A"},
    "5": {"s", "S"},
    "6": {"b", "G"},
    "7": {"t", "T"},
    "8": {"B"},
    "@": {"a", "A"},
    "$": {"s", "S"},
    "\"": {"'"},
    "9": {"g", "q"},
    
}
# Inverted: char -> replacement digit/symbol choices
_OCR_CHAR_MAP: dict[str, list[str]] = {}
for replacement, originals in OCR_MAP.items():
    for ch in originals:
        _OCR_CHAR_MAP.setdefault(ch, []).append(replacement)

UNICODE_ACCENT_VARIANTS: dict[str, list[str]] = {
    "a": ["á", "à", "â", "ä", "ã", "å", "ā", "ă", "ą"],
    "b": ["ḃ", "ƀ", "ɓ"],
    "c": ["ç", "ć", "č"],
    "d": ["ď", "đ", "ḋ", "ḍ"],
    "e": ["é", "è", "ê", "ë", "ē", "ė", "ę"],
    "f": ["ƒ"],
    "g": ["ğ", "ĝ", "ġ", "ģ"],
    "h": ["ħ", "ḥ"],
    "i": ["í", "ì", "î", "ï", "ī", "į"],
    "j": ["ĵ"],
    "k": ["ķ", "ḱ"],
    "l": ["ĺ", "ļ", "ľ", "ł"],
    "m": ["ṃ"],
    "n": ["ñ", "ń", "ņ", "ň"],
    "o": ["ó", "ò", "ô", "ö", "õ", "ō", "ő"],
    "p": ["ṕ"],
    "r": ["ŕ", "ř", "ṛ"],
    "s": ["ś", "š", "ş", "ș"],
    "t": ["ť", "ţ", "ṭ", "ŧ"],
    "u": ["ú", "ù", "û", "ü", "ū", "ű", "ŭ", "ũ"],
    "v": ["ṽ"],
    "w": ["ŵ", "ẁ", "ẃ"],
    "y": ["ý", "ÿ", "ŷ"],
    "z": ["ź", "ž", "ż", "ẓ"],
}

UNICODE_SIMPLE_VARIANTS: dict[str, list[str]] = {
    "a": ["а"],
    "c": ["с"],
    "e": ["е"],
    "i": ["і"],
    "j": ["ј"],
    "o": ["о"],
    "p": ["р"],
    "s": ["ѕ"],
    "x": ["х"],
    "y": ["у"],
}

UNICODE_STYLIZED_VARIANTS: dict[str, list[str]] = {
    "a": ["ɑ", "α"],
    "b": ["Ь", "в"],
    "c": ["ᴄ", "ϲ"],
    "d": ["ԁ"],
    "e": ["ε", "℮"],
    "f": ["ғ"],
    "g": ["ɡ", "г"],
    "h": ["һ"],
    "i": ["ı", "ι"],
    "k": ["к", "κ"],
    "l": ["ⅼ", "ӏ"],
    "m": ["м"],
    "n": ["п", "η"],
    "o": ["օ", "ο"],
    "q": ["զ"],
    "r": ["г", "ɾ", "ʀ"],
    "t": ["т", "τ"],
    "u": ["ս", "υ"],
    "v": ["ѵ", "ν"],
    "w": ["ш", "ω"],
    "z": ["զ"],
}

UNICODE_VARIANT_MAP: dict[str, dict[str, list[str]]] = {}
for base in set(UNICODE_ACCENT_VARIANTS) | set(UNICODE_SIMPLE_VARIANTS) | set(UNICODE_STYLIZED_VARIANTS):
    accent_variants = UNICODE_ACCENT_VARIANTS.get(base, [])
    simple_variants = UNICODE_SIMPLE_VARIANTS.get(base, [])
    stylized_variants = UNICODE_STYLIZED_VARIANTS.get(base, [])
    broad_variants = list(
        dict.fromkeys([*accent_variants, *simple_variants, *stylized_variants])
    )
    levels: dict[str, list[str]] = {}
    if accent_variants:
        levels["accent"] = accent_variants
    if simple_variants:
        levels["simple"] = simple_variants
    if broad_variants:
        levels["broad"] = broad_variants
    UNICODE_VARIANT_MAP[base] = levels

KEYBOARD_MAP: dict[str, list[str]] = {
    "q": ["w", "a"],
    "w": ["q", "e", "s"],
    "e": ["w", "r", "d"],
    "r": ["e", "t", "f"],
    "t": ["r", "y", "g"],
    "y": ["t", "u", "h"],
    "u": ["y", "i", "j"],
    "i": ["u", "o", "k"],
    "o": ["i", "p", "l"],
    "p": ["o", "l"],
    "a": ["q", "s", "z"],
    "s": ["a", "d", "w", "x"],
    "d": ["s", "f", "e", "c"],
    "f": ["d", "g", "r", "v"],
    "g": ["f", "h", "t", "b"],
    "h": ["g", "j", "y", "n"],
    "j": ["h", "k", "u", "m"],
    "k": ["j", "l", "i"],
    "l": ["k", "o", "p"],
    "z": ["a", "x"],
    "x": ["z", "c", "s"],
    "c": ["x", "v", "d"],
    "v": ["c", "b", "f"],
    "b": ["v", "n", "g"],
    "n": ["b", "m", "h"],
    "m": ["n", "j"],
}


# ---------------------------------------------------------------------------
# Index selectors (pure functions, no class state needed)
# ---------------------------------------------------------------------------


def _select_token_indices(
    tokens: list[str],
    token_prob: float,
    min_mutations: int = 0,
    predicate=None,
) -> set[int]:
    predicate = predicate or (lambda t: any(c.isalpha() for c in t))
    eligible = [i for i, t in enumerate(tokens) if predicate(t)]
    if not eligible:
        return set()
    forced_count = min(min_mutations, len(eligible))
    forced = set(random.sample(eligible, k=forced_count)) if forced_count else set()
    stochastic = {
        i for i in eligible if i not in forced and random.random() < token_prob
    }
    return forced | stochastic


def _select_char_indices(
    token: str,
    char_prob: float,
    min_mutations: int = 0,
    predicate=None,
) -> set[int]:
    predicate = predicate or (lambda c: c.isalpha())
    eligible = [i for i, c in enumerate(token) if predicate(c)]
    if not eligible:
        return set()
    forced_count = min(min_mutations, len(eligible))
    forced = set(random.sample(eligible, k=forced_count)) if forced_count else set()
    stochastic = {
        i for i in eligible if i not in forced and random.random() < char_prob
    }
    return forced | stochastic


# ---------------------------------------------------------------------------
# Pool allocation — Hamilton's method
# ---------------------------------------------------------------------------


def _hamilton_allocate(
    requests: list[tuple[str, int]],  # [(mutator_name, min_mutations), ...]
    pool: list[int],  # available token indices
) -> dict[str, int]:
    """
    Fairly allocates guaranteed slots from a finite pool using Hamilton's method
    (largest-remainder apportionment). Emits logging.warning for any mutator
    that receives fewer slots than requested.

    Returns {mutator_name: allocated_count}.
    """
    total_requested = sum(m for _, m in requests)
    available = len(pool)

    if total_requested == 0:
        return {name: 0 for name, _ in requests}

    allocated: dict[str, int] = {}

    if total_requested <= available:
        # No conflict — everyone gets exactly what they asked for
        for name, min_m in requests:
            allocated[name] = min_m
        return allocated

    # Proportional quotas
    quotas = {name: (min_m / total_requested) * available for name, min_m in requests}
    floors = {name: int(q) for name, q in quotas.items()}
    remainders = sorted(quotas.keys(), key=lambda name: -(quotas[name] - floors[name]))

    # Distribute leftover seats by largest remainder
    leftover = available - sum(floors.values())
    result = dict(floors)
    for name in remainders[:leftover]:
        result[name] += 1

    # Warn any mutator that was shorted
    for name, min_m in requests:
        got = result[name]
        if got < min_m:
            log.warning(
                "Mutator %r requested min_mutations=%d but was allocated %d "
                "(pool of %d exhausted by %d total requests).",
                name,
                min_m,
                got,
                available,
                total_requested,
            )

    return result


# ---------------------------------------------------------------------------
# Per-token transform functions (str -> str, stateless)
# ---------------------------------------------------------------------------


def _apply_ocr(
    token: str,
    char_prob: float,
    max_char_mutation_ratio: float = 0.33,
) -> str:
    chars = list(token)
    eligible = [i for i, c in enumerate(chars) if c in _OCR_CHAR_MAP]
    if not eligible:
        return token

    max_char_mutations = max(1, int(len(eligible) * max_char_mutation_ratio))
    indices = {i for i in eligible if random.random() < char_prob}

    # Keep OCR noise readable by limiting the number of substitutions per token.
    if len(indices) > max_char_mutations:
        indices = set(random.sample(sorted(indices), k=max_char_mutations))

    for i in indices:
        chars[i] = random.choice(_OCR_CHAR_MAP[chars[i]])
    return "".join(chars)


def _unicode_variants_for(
    char: str, level: str, fallback_level: str | None = None
) -> list[str]:
    base = char.lower()
    levels = UNICODE_VARIANT_MAP.get(base, {})
    variants = levels.get(level, [])
    if not variants and fallback_level is not None:
        variants = levels.get(fallback_level, [])
    if char.isupper():
        return [v.upper() if len(v) == 1 else v for v in variants]
    return variants


def _apply_accent_mutation(
    token: str,
    char_prob: float,
    max_char_mutation_ratio: float = 0.33,
    max_char_mutations_cap: int = 2,
) -> str:
    chars = list(token)
    eligible = [
        i
        for i, c in enumerate(chars)
        if _unicode_variants_for(c, "accent")
        or _unicode_variants_for(c, "simple")
    ]
    if not eligible:
        return token

    max_char_mutations = min(
        max(1, int(len(eligible) * max_char_mutation_ratio)),
        max_char_mutations_cap,
    )
    indices = {i for i in eligible if random.random() < char_prob}

    # Keep accent mutation readable by limiting substitutions per token.
    if len(indices) > max_char_mutations:
        indices = set(random.sample(sorted(indices), k=max_char_mutations))

    for i in indices:
        variants = _unicode_variants_for(chars[i], "accent") or _unicode_variants_for(
            chars[i], "simple"
        )
        chars[i] = random.choice(variants)
    return "".join(chars)


def _apply_unicode_variation(text: str, level: str = "simple") -> str:
    chars = list(text)
    changed = False
    for i, c in enumerate(chars):
        variants = _unicode_variants_for(c, level)
        if not variants:
            continue
        chars[i] = random.choice(variants)
        changed = True
    return "".join(chars) if changed else text


def _apply_keyboard(
    token: str,
    char_prob: float,
    max_char_mutation_ratio: float = 0.33,
) -> str:
    chars = list(token)
    eligible = [i for i, c in enumerate(chars) if c.lower() in KEYBOARD_MAP]
    if not eligible:
        return token

    max_char_mutations = max(1, int(len(eligible) * max_char_mutation_ratio))
    indices = {i for i in eligible if random.random() < char_prob}

    # Keep the token readable by avoiding excessive mutations in a single word.
    if len(indices) > max_char_mutations:
        indices = set(random.sample(sorted(indices), k=max_char_mutations))

    for i in indices:
        c = chars[i]
        neighbors = KEYBOARD_MAP[c.lower()]
        sub = random.choice(neighbors)
        chars[i] = sub.upper() if c.isupper() else sub
    return "".join(chars)


def _apply_spelling(token: str) -> str:
    """
    Mutates a token using one of three primitive char operations:
      - swap:      swap two adjacent characters         (len >= 4)
      - duplicate: insert a copy of a char after itself (len >= 3)
      - delete:    remove an interior character         (len >= 6)

    Strips leading/trailing punctuation before operating, reattaches after.
    Picks one strategy at random, generates up to 3 candidate positions,
    returns the first result that is not a dictionary word.
    If all candidates collide or the token is too short, returns unchanged.
    """
    prefix = token[: len(token) - len(token.lstrip(_STRIP_CHARS))]
    suffix = token[len(token.rstrip(_STRIP_CHARS)) :]
    stem = (
        token[len(prefix) : len(token) - len(suffix)]
        if suffix
        else token[len(prefix) :]
    )

    if not stem.isalpha():
        return token

    slen = len(stem)

    available = []
    if slen >= 4:
        available.append("swap")
    if slen >= 3:
        available.append("duplicate")
    if slen >= 6:
        available.append("delete")

    if not available:
        return token

    strategy = random.choice(available)
    chars = list(stem)

    if strategy == "swap":
        positions = random.sample(range(slen - 1), k=min(3, slen - 1))
        for pos in positions:
            candidate = chars[:]
            candidate[pos], candidate[pos + 1] = candidate[pos + 1], candidate[pos]
            result = "".join(candidate)
            if not _is_dict_word(result):
                return prefix + result + suffix
        return token

    elif strategy == "duplicate":
        positions = random.sample(range(slen), k=min(3, slen))
        for pos in positions:
            candidate = chars[: pos + 1] + [chars[pos]] + chars[pos + 1 :]
            result = "".join(candidate)
            if not _is_dict_word(result):
                return prefix + result + suffix
        return token

    else:  # delete — interior chars only
        interior = list(range(1, slen - 1))
        if not interior:
            return token
        positions = random.sample(interior, k=min(3, len(interior)))
        for pos in positions:
            candidate = chars[:pos] + chars[pos + 1 :]
            result = "".join(candidate)
            if not _is_dict_word(result):
                return prefix + result + suffix
        return token


def _apply_casing_swapcase(token: str) -> str:
    return token.swapcase()


def _apply_casing_uppercase(token: str) -> str:
    return token.upper()


def _apply_interval_flip_char(token: str, char_prob: float) -> str:
    indices = _select_char_indices(token, char_prob=char_prob)
    if not indices:
        return token
    chars = list(token)
    for i in indices:
        chars[i] = chars[i].swapcase()
    return "".join(chars)


def _apply_sparse_stuffing(token: str, sep: str, char_prob: float) -> str:
    if len(token) <= 1:
        return token
    out = []
    last = len(token) - 1
    for j, c in enumerate(token):
        out.append(c)
        if j < last and random.random() < char_prob:
            out.append(sep)
    return "".join(out)


def _apply_pig_latin(token: str) -> str:
    """
    Convert a token to a simple Pig Latin variant.

    Rules:
      - vowel-starting words: append "yay"
      - consonant-starting words: move the leading consonant cluster to the end
        and append "ay"

    Leading/trailing punctuation is preserved.
    """
    prefix = token[: len(token) - len(token.lstrip(_STRIP_CHARS))]
    suffix = token[len(token.rstrip(_STRIP_CHARS)) :]
    stem = token[len(prefix) : len(token) - len(suffix)] if suffix else token[len(prefix) :]

    if not stem or not stem.isalpha():
        return token

    lower = stem.lower()
    vowels = "aeiou"

    if lower[0] in vowels:
        result = lower + "yay"
    else:
        cluster_end = 0
        for i, ch in enumerate(lower):
            if ch in vowels:
                cluster_end = i
                break
        else:
            cluster_end = len(lower)

        if cluster_end == 0:
            cluster_end = 1

        result = lower[cluster_end:] + lower[:cluster_end] + "ay"

    if stem[0].isupper():
        result = result.capitalize()

    return prefix + result + suffix


# ---------------------------------------------------------------------------
# MutationOrchestrator
# ---------------------------------------------------------------------------

_STOP_WORDS = ["the", "and", "is", "of", "at", "with", "it", "or", "to", "a", "an"]


def _stop_word_caller(params: dict | None) -> str:
    return random.choice(_STOP_WORDS)


# ---------------------------------------------------------------------------
# RegisteredMutation — registry contract for token-level and post-process ops
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class RegisteredMutation:
    """
    A named registered mutation.

    Fields:
        name     : registry key — must match the 'method' string in a profile step
        kind     : "token", "post_process", or "text"
        transform : token-level or post-process callable
        chance   : probability the post-processor fires per mutate() call
        count    : number of injection points to generate for post-process ops
        position : "random" | "start" | "end" for post-process ops
        params   : arbitrary extra data forwarded to the callable
        randomizable : whether blank-profile generation may pick this mutation
    """

    name: str
    kind: str
    transform: Callable[..., str]
    chance: float = 1.0
    count: int = 1
    position: str = "random"
    params: dict = field(default_factory=dict)
    randomizable: bool = True

    def _resolve_positions(self, n: int) -> list[int]:
        if self.position == "start":
            return [0] * self.count
        if self.position == "end":
            return [n] * self.count
        return [random.randint(0, n) for _ in range(self.count)]

    def build_injection_map(self, n: int) -> dict[int, list[str]]:
        """Returns {token_index: [injected_strings]} for splicing at join time."""
        if random.random() >= self.chance:
            return {}
        injection: dict[int, list[str]] = {}
        for pos in self._resolve_positions(n):
            injection.setdefault(pos, []).append(self.transform(self.params or None))
        return injection


class MutationOrchestrator:
    """
    Single split → single pass → single join pipeline.

    Built-in profile methods:
        "invert_base"         params: {}
        "nth_strategy_word"   params: {interval: int}
        "nth_strategy_char"   params: {interval: int}
        "interval_flip_token" params: {token_prob: float}
        "interval_flip_char"  params: {token_prob: float, char_prob: float}
        "sparse_stuffing"     params: {word_prob: float, char_prob: float, symbols: str}
        "accent_mutation"     params: {token_prob: float, char_prob: float, max_char_mutation_ratio: float}
        "pig_latin"           params: {token_prob: float}
        "word_chunk_swap"     params: {}
        "reverse"             params: {}
        "stop_word_injection" params: {count: int, position: str}
        "ocr"                 params: {token_prob: float, char_prob: float, max_char_mutation_ratio: float}
        "keyboard"            params: {token_prob: float, char_prob: float, max_char_mutation_ratio: float}
        "spelling"            params: {token_prob: float}
        "unicode_variation"   params: {level: "accent" | "simple" | "broad"}

    Registered mutations are referenced by their name as 'method' in a
    profile step, identical to built-ins. Use register() to add them.
    """

    # Default restriction map: restrictive mutator -> set of permitted co-mutators.
    # Any mutator not in the permitted set is blocked from sharing a token index
    # with the restrictor. Extend via register_restriction().
    _DEFAULT_RESTRICTIONS: dict[str, set[str]] = {
        "sparse_stuffing": {"casing_swap", "casing_upper"},
        "ocr": {"casing_swap", "casing_upper"},
        "accent_mutation": {"casing_swap", "casing_upper"},
        "pig_latin": {"casing_swap", "casing_upper"},
    }

    def __init__(self) -> None:
        # stop_word_injection is pre-registered so it is dispatched identically
        # to any user-registered post-process mutation at join time.
        self._registry: dict[str, RegisteredMutation] = {}
        self._restrictions: dict[str, set[str]] = dict(self._DEFAULT_RESTRICTIONS)
        self._randomizable_overrides: dict[str, bool] = {}
        self.register(
            name="stop_word_injection",
            transform=_stop_word_caller,
            chance=1.0,
            count=1,
            position="random",
            kind="post_process",
        )

    def register(
        self,
        name: str,
        caller: Callable[..., str] | None = None,
        transform: Callable[..., str] | None = None,
        chance: float = 1.0,
        count: int = 1,
        position: str = "random",
        params: dict | None = None,
        kind: str = "post_process",
        randomizable: bool = True,
    ) -> None:
        """
        Register a mutation.

        Args:
            name     : profile 'method' key that activates this mutation.
            caller   : backward-compatible callable alias for transform.
            transform : callable for the registered mutation.
                       - kind="token": callable(token: str, params: dict | None) -> str
                       - kind="post_process": callable(params: dict | None) -> str
                       - kind="text": callable(text: str) -> str
            chance   : probability [0, 1] the mutation runs per mutate() call.
            count    : number of injection points to produce for post-process ops.
            position : "random" | "start" | "end"
            params   : forwarded verbatim to the callable on every invocation.
            kind     : "token", "post_process", or "text"
            randomizable : if False, blank-profile generation will skip it.

        Raises:
            ValueError: if kind or position are not one of the accepted values.
        """
        if kind not in ("token", "post_process", "text"):
            raise ValueError(
                f"kind must be 'token', 'post_process', or 'text'; got {kind!r}"
            )
        if position not in ("random", "start", "end"):
            raise ValueError(
                f"position must be 'random', 'start', or 'end'; got {position!r}"
            )
        fn = transform or caller
        if fn is None:
            raise ValueError("register() requires either caller or transform")
        self._registry[name] = RegisteredMutation(
            name=name,
            kind=kind,
            transform=fn,
            chance=chance,
            count=count,
            position=position,
            params=params or {},
            randomizable=randomizable,
        )

    def set_randomizable(self, name: str, enabled: bool) -> None:
        """
        Enable or disable whether a method may be chosen by blank-profile
        generation.

        This applies to both built-ins and registered mutations.
        """
        self._randomizable_overrides[name] = enabled

    def _is_randomizable(self, name: str) -> bool:
        if name in self._randomizable_overrides:
            return self._randomizable_overrides[name]
        reg = self._registry.get(name)
        if reg is not None:
            return reg.randomizable
        return True

    def register_restriction(
        self,
        name: str,
        permitted: set[str],
    ) -> None:
        """
        Register or replace a restriction for a mutator.

        Args:
            name      : the restrictive mutator name (e.g. "sparse_stuffing").
            permitted : set of mutator names allowed to co-apply on the same
                        token. All others are blocked.

        Example:
            orchestrator.register_restriction("keyboard", {"casing_swap"})
        """
        self._restrictions[name] = permitted

    def _random_profile(self) -> list[dict]:
        """
        Build a small random profile with 1-3 methods.

        The reorder pair (word_chunk_swap / reverse) is treated as mutually
        exclusive so the fallback does not create a no-op combination.
        """
        builtins = [
            "invert_base",
            "nth_strategy_word",
            "nth_strategy_char",
            "interval_flip_token",
            "interval_flip_char",
            "sparse_stuffing",
            "accent_mutation",
            "pig_latin",
            "word_chunk_swap",
            "reverse",
            "ocr",
            "keyboard",
            "spelling",
            "unicode_variation",
        ]
        registry_names = list(self._registry)
        randomizable_registry_names = [
            name for name, reg in self._registry.items() if reg.randomizable
        ]
        methods: list[str] = []

        if random.random() < 0.5:
            methods.append(random.choice(["word_chunk_swap", "reverse"]))

        pool = [m for m in builtins if m not in methods and self._is_randomizable(m)]
        pool.extend(
            name
            for name in randomizable_registry_names
            if name not in methods and self._is_randomizable(name)
        )

        target_size = random.randint(1, min(3, len(pool) + len(methods)))
        while len(methods) < target_size and pool:
            choice = random.choice(pool)
            pool.remove(choice)
            if choice in {"word_chunk_swap", "reverse"} and any(
                m in {"word_chunk_swap", "reverse"} for m in methods
            ):
                continue
            methods.append(choice)

        def _profile_for(method: str) -> dict:
            if method == "invert_base":
                return {"method": method, "chance": 1.0}
            if method == "nth_strategy_word":
                return {
                    "method": method,
                    "chance": 1.0,
                    "params": {"interval": random.randint(2, 4)},
                }
            if method == "nth_strategy_char":
                return {
                    "method": method,
                    "chance": 1.0,
                    "params": {"interval": random.randint(2, 4)},
                }
            if method == "interval_flip_token":
                return {
                    "method": method,
                    "chance": 1.0,
                    "params": {"token_prob": round(random.uniform(0.3, 0.8), 2)},
                }
            if method == "interval_flip_char":
                return {
                    "method": method,
                    "chance": 1.0,
                    "params": {
                        "token_prob": round(random.uniform(0.3, 0.8), 2),
                        "char_prob": round(random.uniform(0.2, 0.6), 2),
                    },
                }
            if method == "sparse_stuffing":
                return {
                    "method": method,
                    "chance": 1.0,
                    "params": {
                        "word_prob": round(random.uniform(0.3, 0.8), 2),
                        "char_prob": round(random.uniform(0.2, 0.5), 2),
                    },
                }
            if method == "accent_mutation":
                return {
                    "method": method,
                    "chance": 1.0,
                    "params": {
                        "token_prob": round(random.uniform(0.4, 0.9), 2),
                        "char_prob": round(random.uniform(0.2, 0.6), 2),
                        "max_char_mutation_ratio": 0.33,
                    },
                }
            if method == "pig_latin":
                return {
                    "method": method,
                    "chance": 1.0,
                    "params": {"token_prob": round(random.uniform(0.4, 0.9), 2)},
                }
            if method == "word_chunk_swap":
                return {"method": method, "chance": 1.0}
            if method == "reverse":
                return {"method": method, "chance": 1.0}
            if method == "ocr":
                return {
                    "method": method,
                    "chance": 1.0,
                    "params": {
                        "token_prob": round(random.uniform(0.4, 0.9), 2),
                        "char_prob": round(random.uniform(0.2, 0.6), 2),
                        "max_char_mutation_ratio": 0.33,
                    },
                }
            if method == "keyboard":
                return {
                    "method": method,
                    "chance": 1.0,
                    "params": {
                        "token_prob": round(random.uniform(0.4, 0.9), 2),
                        "char_prob": round(random.uniform(0.2, 0.6), 2),
                        "max_char_mutation_ratio": 0.33,
                    },
                }
            if method == "spelling":
                return {
                    "method": method,
                    "chance": 1.0,
                    "params": {"token_prob": round(random.uniform(0.4, 0.9), 2)},
                }
            if method == "unicode_variation":
                return {
                    "method": method,
                    "chance": 1.0,
                    "params": {"level": random.choice(["simple", "accent", "broad"])},
                }
            return {"method": method, "chance": 1.0}

        return [_profile_for(method) for method in methods]

    def mutate(
        self,
        text: str,
        profile: list[dict] | None = None,
        seed: int | None = None,
    ) -> str:
        if seed is not None:
            random.seed(seed)

        if not text:
            return text

        if not profile:
            profile = self._random_profile()

        # ------------------------------------------------------------------ #
        # Stage 0 — invert_base: whole-string swapcase before split           #
        # ------------------------------------------------------------------ #
        for step in profile:
            if step["method"] == "invert_base" and random.random() < step.get(
                "chance", 1.0
            ):
                text = text.swapcase()
                break  # only one invert_base makes sense

        # ------------------------------------------------------------------ #
        # Stage 1 — split once                                                #
        # ------------------------------------------------------------------ #
        tokens: list[str] = text.split()
        n = len(tokens)
        if n == 0:
            return text

        # ------------------------------------------------------------------ #
        # Stage 2 — word_chunk_swap: determine two half-ranges                #
        # ------------------------------------------------------------------ #
        do_swap = any(
            step["method"] == "word_chunk_swap"
            and random.random() < step.get("chance", 1.0)
            for step in profile
        )
        if do_swap and n >= 2:
            mid = n // 2
            range_a = (mid, n)  # second half iterated first
            range_b = (0, mid)
        else:
            mid = n
            range_a = (0, n)
            range_b = None  # no second range

        # ------------------------------------------------------------------ #
        # Stage 3 — reverse flag                                              #
        # ------------------------------------------------------------------ #
        do_reverse = any(
            step["method"] == "reverse" and random.random() < step.get("chance", 1.0)
            for step in profile
        )
        if do_swap and do_reverse:
            log.warning(
                "word_chunk_swap and reverse were both selected; skipping both to avoid equivalent reorderings."
            )
            do_swap = False
            do_reverse = False

        # ------------------------------------------------------------------ #
        # Stage 4 — precompute index sets for in-pass mutators                #
        #                                                                      #
        # Allocation order:                                                    #
        #   4a. Parse active steps, separate restrictive vs unrestricted       #
        #   4b. Hamilton-allocate min_mutations for restrictive mutators first #
        #       from the full eligible pool → restrictive_indices              #
        #   4c. free_pool = eligible - restrictive_indices                     #
        #   4d. Hamilton-allocate min_mutations for unrestricted mutators      #
        #       from free_pool only                                            #
        #   4e. Strip any unrestricted index that landed on a restrictive      #
        #       token and is not in that token's permitted co-mutator set      #
        # ------------------------------------------------------------------ #

        _STRUCTURAL = {"invert_base", "word_chunk_swap", "reverse"} | set(
            name
            for name, reg in self._registry.items()
            if reg.kind == "post_process"
        )
        _TOKEN_CUSTOM = {
            name for name, reg in self._registry.items() if reg.kind == "token"
        }
        _TEXT_CUSTOM = {
            name for name, reg in self._registry.items() if reg.kind == "text"
        }
        _TEXT_FINAL = {"unicode_variation"}

        # Per-mutator config harvested from active profile steps
        _cfg: dict[str, dict] = {}  # method -> {p, min_m, chance_passed}
        for step in profile:
            method = step["method"]
            if method in _STRUCTURAL or method in _TEXT_CUSTOM or method in _TEXT_FINAL:
                continue
            if random.random() >= step.get("chance", 1.0):
                continue
            p = step.get("params", {})
            min_m = step.get("min_mutations", p.get("min_mutations", 0))
            _cfg[method] = {"p": p, "min_m": min_m}

        # Shared eligibility pools
        _all_eligible = [i for i, t in enumerate(tokens) if any(c.isalpha() for c in t)]

        # --- 4b: Restrictive mutators first ---
        _restrictive_names = [m for m in _cfg if m in self._restrictions]
        _restrict_requests = [(m, _cfg[m]["min_m"]) for m in _restrictive_names]
        _restrict_alloc = _hamilton_allocate(_restrict_requests, _all_eligible)

        ocr_indices: set[int] = set()
        accent_indices: set[int] = set()
        stuffing_indices: set[int] = set()
        ocr_char_prob: float = 0.3
        ocr_max_char_mutation_ratio: float = 0.33
        accent_char_prob: float = 0.3
        accent_max_char_mutation_ratio: float = 0.33
        stuffing_char_prob: float = 0.3
        stuffing_sep: str = "."

        _restrictive_indices: set[int] = set()

        for method in _restrictive_names:
            cfg = _cfg[method]
            p, min_m = cfg["p"], _restrict_alloc[method]
            # Select from the portion of the pool not yet claimed by other restrictors
            local_pool = [i for i in _all_eligible if i not in _restrictive_indices]

            if method == "sparse_stuffing":
                stuffing_char_prob = p.get("char_prob", 0.3)
                stuffing_sep = random.choice(p.get("symbols", "._*- "))
                stuffing_indices |= _select_token_indices(
                    tokens,
                    p.get("word_prob", 0.4),
                    min_m,
                    predicate=lambda t: len(t) > 1,
                )
                _restrictive_indices |= stuffing_indices

            elif method == "ocr":
                ocr_char_prob = p.get("char_prob", 0.3)
                ocr_max_char_mutation_ratio = p.get("max_char_mutation_ratio", 0.33)
                ocr_indices |= _select_token_indices(
                    tokens,
                    p.get("token_prob", 0.3),
                    min_m,
                )
                _restrictive_indices |= ocr_indices

            elif method == "accent_mutation":
                accent_char_prob = p.get("char_prob", 0.3)
                accent_max_char_mutation_ratio = p.get("max_char_mutation_ratio", 0.33)
                accent_indices |= _select_token_indices(
                    tokens,
                    p.get("token_prob", 0.3),
                    min_m,
                )
                _restrictive_indices |= accent_indices

        # --- 4c: Free pool ---
        _free_pool = [i for i in _all_eligible if i not in _restrictive_indices]

        # --- 4d: Unrestricted mutators from free_pool only ---
        _unrestricted_names = [m for m in _cfg if m not in self._restrictions]
        _free_requests = [(m, _cfg[m]["min_m"]) for m in _unrestricted_names]
        _free_alloc = _hamilton_allocate(_free_requests, _free_pool)
        _custom_token_methods = [m for m in _unrestricted_names if m in _TOKEN_CUSTOM]

        casing_swap_indices: set[int] = set()
        casing_upper_indices: set[int] = set()
        interval_char_indices: set[int] = set()
        keyboard_indices: set[int] = set()
        spelling_indices: set[int] = set()
        pig_latin_indices: set[int] = set()
        custom_token_indices: dict[str, set[int]] = {}
        nth_char_interval: int | None = None
        interval_char_prob: float = 0.3
        keyboard_char_prob: float = 0.3
        keyboard_max_char_mutation_ratio: float = 0.33

        # Wrap _select_token_indices to draw only from free_pool
        def _select_free(min_m: int, token_prob: float, predicate=None) -> set[int]:
            eligible = (
                [i for i in _free_pool if predicate(tokens[i])]
                if predicate
                else [i for i in _free_pool if any(c.isalpha() for c in tokens[i])]
            )
            if not eligible:
                return set()
            forced_count = min(min_m, len(eligible))
            forced = (
                set(random.sample(eligible, k=forced_count)) if forced_count else set()
            )
            stochastic = {
                i for i in eligible if i not in forced and random.random() < token_prob
            }
            return forced | stochastic

        for method in _unrestricted_names:
            cfg = _cfg[method]
            p, min_m = cfg["p"], _free_alloc[method]

            if method == "nth_strategy_word":
                # Deterministic — all nth indices, unrestricted by pool
                casing_upper_indices |= {i for i in range(0, n, p.get("interval", 2))}

            elif method == "nth_strategy_char":
                nth_char_interval = p.get("interval", 2)

            elif method == "interval_flip_token":
                casing_swap_indices |= _select_free(min_m, p.get("token_prob", 0.3))

            elif method == "interval_flip_char":
                interval_char_prob = p.get("char_prob", 0.3)
                interval_char_indices |= _select_free(min_m, p.get("token_prob", 0.3))

            elif method == "keyboard":
                keyboard_char_prob = p.get("char_prob", 0.3)
                keyboard_max_char_mutation_ratio = p.get("max_char_mutation_ratio", 0.33)
                keyboard_indices |= _select_free(min_m, p.get("token_prob", 0.3))

            elif method == "spelling":
                spelling_indices |= _select_free(
                    min_m,
                    p.get("token_prob", 0.3),
                    predicate=_spelling_eligible,
                )

            elif method == "pig_latin":
                pig_latin_indices |= _select_free(
                    min_m,
                    p.get("token_prob", 0.3),
                    predicate=lambda t: t.strip(_STRIP_CHARS).isalpha(),
                )

            elif method in _TOKEN_CUSTOM:
                custom_token_indices[method] = _select_free(
                    min_m,
                    p.get("token_prob", 0.3),
                    predicate=lambda t: bool(t.strip()),
                )

        # --- 4e: Strip unrestricted indices that co-land on restricted tokens ---
        # For each restrictive mutator, remove co-landing unrestricted indices
        # that are not in its permitted co-mutator set.
        _name_to_index_set: dict[str, set[int]] = {
            "casing_swap": casing_swap_indices,
            "casing_upper": casing_upper_indices,
            "interval_char": interval_char_indices,
            "keyboard": keyboard_indices,
            "spelling": spelling_indices,
            "accent_mutation": accent_indices,
            "pig_latin": pig_latin_indices,
        }
        _name_to_index_set.update(custom_token_indices)
        for restrictor, permitted in self._restrictions.items():
            restrictor_indices = (
                stuffing_indices if restrictor == "sparse_stuffing" else ocr_indices
            )
            for co_name, co_set in _name_to_index_set.items():
                if co_name not in permitted:
                    co_set -= restrictor_indices

        # ------------------------------------------------------------------ #
        # Stage 5 — single pass: apply transforms in order per token          #
        # OCR -> casing -> interval_flip -> sparse_stuffing                   #
        # ------------------------------------------------------------------ #
        def _process_token(i: int) -> str:
            tok = tokens[i]

            # 1. OCR
            if i in ocr_indices:
                tok = _apply_ocr(tok, ocr_char_prob, ocr_max_char_mutation_ratio)

            # 2. Accent mutation
            if i in accent_indices:
                tok = _apply_accent_mutation(
                    tok,
                    accent_char_prob,
                    accent_max_char_mutation_ratio,
                )

            # 3. Keyboard
            if i in keyboard_indices:
                tok = _apply_keyboard(tok, keyboard_char_prob, keyboard_max_char_mutation_ratio)

            # 4. Spelling
            if i in spelling_indices:
                tok = _apply_spelling(tok)

            # 5. Pig Latin
            if i in pig_latin_indices:
                tok = _apply_pig_latin(tok)

            # 6. Casing: nth_strategy (word-level uppercase)
            if i in casing_upper_indices:
                tok = _apply_casing_uppercase(tok)

            # 7. Casing: interval_flip token-level swapcase
            if i in casing_swap_indices:
                tok = _apply_casing_swapcase(tok)

            # 8. nth_strategy char-level
            if nth_char_interval is not None:
                chars = list(tok)
                for j in range(0, len(chars), nth_char_interval):
                    if chars[j].isalpha():
                        chars[j] = chars[j].upper()
                tok = "".join(chars)

            # 9. interval_flip char-level
            if i in interval_char_indices:
                tok = _apply_interval_flip_char(tok, interval_char_prob)

            # 10. sparse_stuffing
            if i in stuffing_indices:
                tok = _apply_sparse_stuffing(tok, stuffing_sep, stuffing_char_prob)

            # 11. registered token-level custom mutations
            for method in _custom_token_methods:
                if i in custom_token_indices.get(method, set()):
                    reg = self._registry[method]
                    tok = reg.transform(tok, _cfg[method]["p"])

            return tok

        # ------------------------------------------------------------------ #
        # Stage 6 — build output with optional reverse, then post-processors  #
        # ------------------------------------------------------------------ #

        # Dispatch all registered post-processors referenced in the profile.
        # chance/count/position on the profile step override the registration
        # defaults, so callers can tune per-use without re-registering.
        injection_map: dict[int, list[str]] = {}
        for step in profile:
            pp = self._registry.get(step["method"])
            if pp is None or pp.kind != "post_process":
                continue
            p = step.get("params", {})
            # Allow profile step to override registration defaults
            override = RegisteredMutation(
                name=pp.name,
                kind=pp.kind,
                transform=pp.transform,
                chance=step.get("chance", pp.chance),
                count=p.get("count", pp.count),
                position=p.get("position", pp.position),
                params={**pp.params, **p},
            )
            for pos, words in override.build_injection_map(n).items():
                injection_map.setdefault(pos, []).extend(words)

        def _iter_range(start: int, stop: int, reverse: bool):
            r = range(stop - 1, start - 1, -1) if reverse else range(start, stop)
            for i in r:
                # injections keyed by original index (pre-reverse), inserted before token
                for w in injection_map.get(i, []):
                    yield w
                yield _process_token(i)

        parts: list[str] = []
        parts.extend(_iter_range(*range_a, do_reverse))
        if range_b is not None:
            parts.extend(_iter_range(*range_b, do_reverse))

        # Trailing injections (position == n) only if no swap (original end)
        if not do_swap:
            for w in injection_map.get(n, []):
                parts.append(w)

        result = " ".join(parts)

        # Final text-level processors run after the full output has been built.
        for step in profile:
            pp = self._registry.get(step["method"])
            if pp is None or pp.kind != "text":
                continue
            if random.random() >= step.get("chance", pp.chance):
                continue
            p = step.get("params", {})
            override = RegisteredMutation(
                name=pp.name,
                kind=pp.kind,
                transform=pp.transform,
                chance=step.get("chance", pp.chance),
                count=pp.count,
                position=pp.position,
                params={**pp.params, **p},
            )
            result = override.transform(result)

        for step in profile:
            if step["method"] != "unicode_variation":
                continue
            if random.random() >= step.get("chance", 1.0):
                continue
            level = step.get("params", {}).get("level", "simple")
            result = _apply_unicode_variation(result, level=level)

        return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_texts = [
        "The service was excellent.",
        "Please send the report by 5:30 PM on Tuesday.",
        "The quick brown fox jumps over the lazy dog.",
        "Accents and unicode variants should be easy to spot.",
    ]

    orchestrator = MutationOrchestrator()

    # --- Example: register a custom emoji injector ---
    _EMOJIS = ["😀", "💯", "🔥", "✨", "😤", "👀", "💀", "🫡"]
    orchestrator.register(
        name="emoji_injection",
        transform=lambda params: random.choice(params["pool"]) if params else "😀",
        chance=1.0,
        count=2,
        position="random",
        params={"pool": _EMOJIS},
        kind="post_process",
    )

    profiles = [
        (
            "invert_base",
            [
                {"method": "invert_base", "chance": 1.0},
            ],
        ),
        (
            "nth_strategy_word",
            [
                {
                    "method": "nth_strategy_word",
                    "chance": 1.0,
                    "params": {"interval": 2},
                },
            ],
        ),
        (
            "nth_strategy_char",
            [
                {
                    "method": "nth_strategy_char",
                    "chance": 1.0,
                    "params": {"interval": 2},
                },
            ],
        ),
        (
            "interval_flip_token",
            [
                {
                    "method": "interval_flip_token",
                    "chance": 1.0,
                    "params": {"token_prob": 0.6},
                },
            ],
        ),
        (
            "accent_mutation",
            [
                {
                    "method": "accent_mutation",
                    "chance": 1.0,
                    "params": {
                        "token_prob": 1.0,
                        "char_prob": 0.6,
                        "max_char_mutation_ratio": 0.33,
                    },
                },
            ],
        ),
        (
            "unicode_simple",
            [
                {
                    "method": "unicode_variation",
                    "chance": 1.0,
                    "params": {"level": "simple"},
                },
            ],
        ),
        (
            "unicode_broad",
            [
                {
                    "method": "unicode_variation",
                    "chance": 1.0,
                    "params": {"level": "broad"},
                },
            ],
        ),
        (
            "accent + unicode",
            [
                {
                    "method": "accent_mutation",
                    "chance": 1.0,
                    "params": {
                        "token_prob": 1.0,
                        "char_prob": 0.6,
                        "max_char_mutation_ratio": 0.33,
                    },
                },
                {
                    "method": "unicode_variation",
                    "chance": 1.0,
                    "params": {"level": "broad"},
                },
            ],
        ),
        (
            "stop_word_injection",
            [
                {
                    "method": "stop_word_injection",
                    "chance": 1.0,
                    "params": {"count": 2},
                },
            ],
        ),
        (
            "ocr",
            [
                {
                    "method": "ocr",
                    "chance": 1.0,
                    "params": {"token_prob": 0.8, "char_prob": 0.6, "max_char_mutation_ratio": 0.33},
                },
            ],
        ),
        (
            "keyboard",
            [
                {
                    "method": "keyboard",
                    "chance": 1.0,
                    "params": {"token_prob": 0.5, "char_prob": 0.4, "max_char_mutation_ratio": 0.33},
                },
            ],
        ),
        (
            "spelling",
            [
                {"method": "spelling", "chance": 1.0, "params": {"token_prob": 0.8}},
            ],
        ),
        (
            "pig_latin",
            [
                {
                    "method": "pig_latin",
                    "chance": 1.0,
                    "params": {"token_prob": 0.8},
                },
            ],
        ),
        (
            "emoji_injection",
            [
                {
                    "method": "emoji_injection",
                    "chance": 1.0,
                    "params": {"count": 3, "position": "random"},
                },
            ],
        ),
        (
            "emoji_injection (start)",
            [
                {
                    "method": "emoji_injection",
                    "chance": 1.0,
                    "params": {"count": 1, "position": "start"},
                },
            ],
        ),
        (
            "emoji_injection (end)",
            [
                {
                    "method": "emoji_injection",
                    "chance": 1.0,
                    "params": {"count": 2, "position": "end"},
                },
            ],
        ),
        (
            "text processor",
            [
                {
                    "method": "suffix_note",
                    "chance": 1.0,
                }
            ],
        ),
        (
            "orchestrated",
            [
                {"method": "invert_base", "chance": 0.3},
                {
                    "method": "ocr",
                    "chance": 0.5,
                    "params": {
                        "token_prob": 0.4,
                        "char_prob": 0.5,
                        "max_char_mutation_ratio": 0.33,
                    },
                },
                {
                    "method": "accent_mutation",
                    "chance": 0.5,
                    "params": {
                        "token_prob": 0.4,
                        "char_prob": 0.5,
                        "max_char_mutation_ratio": 0.33,
                    },
                },
                {
                    "method": "interval_flip_token",
                    "chance": 0.7,
                    "params": {"token_prob": 0.4},
                },
                {
                    "method": "unicode_variation",
                    "chance": 0.5,
                    "params": {"level": "simple"},
                },
            ],
        ),
    ]

    orchestrator.register(
        name="suffix_note",
        kind="text",
        transform=lambda text: text + " [finalized]",
        chance=1.0,
    )

    for text in demo_texts:
        print(f"\nINPUT: {text}")
        for name, profile in profiles:
            result = orchestrator.mutate(text, profile, seed=42)
            print(f"  {name:<25}: {result}")
