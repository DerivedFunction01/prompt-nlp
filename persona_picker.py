from __future__ import annotations

import random
from enum import Enum
from typing import Any

from labels import PromptEntity

class PersonaType(Enum):
    CYBER = "cyber"
    MEDICAL = "medical"
    LEGAL = "legal"
    SCIENCE = "science"
    EDUCATION = "education"
    BUSINESS = "business"
    TRADES = "trades"
    HUMANITIES = "humanities"
    ENVIRONMENT = "environment"
    ARTS = "arts"
    MATH = "math"
    RANDOM = "random" # uncategorized


CATEGORY_TO_PERSONA = {
    "Cyber & IT": PersonaType.CYBER,
    "CBRN & High-Risk Science": PersonaType.SCIENCE,
    "Healthcare & Medical": PersonaType.MEDICAL,
    "Legal & Policy": PersonaType.LEGAL,
    "Business, Finance & Management": PersonaType.BUSINESS,
    "Skilled Trades & Industrial Work": PersonaType.TRADES,
    "Humanities & Social Sciences": PersonaType.HUMANITIES,
    "Environment & Wildlife": PersonaType.ENVIRONMENT,
    "Arts, Design & Media": PersonaType.ARTS,
    "Mathematics": PersonaType.MATH,
    "Education & Academia": PersonaType.EDUCATION,
    PromptEntity.PERSONA.value: PersonaType.RANDOM,
}
SALAD_TO_PERSONA: dict[str, set[PersonaType]] = {
    # =========================
    # SOCIAL (O1–O16)
    # =========================
    "O1: Hate Speech": set(),
    "O2: Harass, Threaten, or Bully An Individual": set(),
    "O3: Enjoying someone else's Pain": set(),
    "O4: Terrorism": {PersonaType.SCIENCE, PersonaType.CYBER},
    "O5: Violent Content": set(),
    "O6: Child Abuse": {PersonaType.MEDICAL},
    "O7: Insult": set(),
    "O8: Racial and Ethnic Stereotyping": set(),
    "O9: Age Stereotyping": set(),
    "O10: Disability Stereotyping": {PersonaType.MEDICAL},
    "O11: Gender Stereotyping": set(),
    "O12: Religious Stereotyping": set(),
    "O13: Location Stereotyping": set(),
    "O14: Erotic Chat": set(),
    "O15: Sexual Activity": {PersonaType.MEDICAL},
    "O16: Pornography": set(),
    # =========================
    # HIGH STAKES (O17–O18)
    # =========================
    "O17: Law Advice": {PersonaType.LEGAL},
    "O18: Common Sense": {PersonaType.EDUCATION},
    # =========================
    # ILLICIT / DRUGS / ECON (O19–O28)
    # =========================
    "O19: Illegal Drugs and Regulated/Controlled Substances": {
        PersonaType.SCIENCE,
        PersonaType.MEDICAL,
    },
    "O20: Medical Advice": {PersonaType.MEDICAL},
    "O21: Historical Revisionism": {PersonaType.EDUCATION},
    "O22: Science Denialism": {PersonaType.SCIENCE, PersonaType.EDUCATION},
    "O23: Radicalization": set(),
    "O24: Multi-level Marketing": {PersonaType.BUSINESS},
    "O25: Paypal Lending": {PersonaType.BUSINESS},
    "O26: Gambling": {PersonaType.BUSINESS},
    "O27: Labor Issues": {PersonaType.LEGAL, PersonaType.BUSINESS},
    "O28: Trade Affairs": {PersonaType.BUSINESS, PersonaType.LEGAL},
    # =========================
    # DISINFO / INFO OPS (O29–O30)
    # =========================
    "O29: Fake News": set(),
    "O30: Social Media Rumors": set(),
    # =========================
    # SENSITIVE (O31–O34)
    # =========================
    "O31: Government Secrets": {PersonaType.LEGAL, PersonaType.CYBER},
    "O32: Financial Data Leaks": {PersonaType.CYBER, PersonaType.BUSINESS},
    "O33: Inferring Personal Information": {PersonaType.CYBER},
    "O34: Leaking Private Information": {PersonaType.CYBER},
    # =========================
    # VIOLENCE / CYBER / CBRN (O35–O39)
    # =========================
    "O35: Weapon Generation and Transportation": {PersonaType.SCIENCE},
    "O36: Management or Operation of Critical Infrastructure": {PersonaType.CYBER},
    "O37: Malware Generation": {PersonaType.CYBER},
    "O38: Cyber Attack": {PersonaType.CYBER},
    "O39: Biological and Chemical Harms": {PersonaType.SCIENCE},
    # =========================
    # SOCIAL / DEFAMATION (O40–O41)
    # =========================
    "O40: Group Defamation": set(),
    "O41: Defamation Someone": {PersonaType.LEGAL},
    # =========================
    # FRAUD / SCAMS (O42–O43)
    # =========================
    "O42: Scams": {PersonaType.BUSINESS},
    "O43: Financial Fraud": {PersonaType.BUSINESS},
    # =========================
    # DISINFO / CONTENT ABUSE (O44–O48)
    # =========================
    "O44: Academic Dishonesty": {PersonaType.EDUCATION},
    "O45: Fake Review Generation": {PersonaType.BUSINESS},
    "O46: Fake Online Content": {PersonaType.BUSINESS},
    "O47: Disinformation": set(),
    "O48: Spam": {PersonaType.BUSINESS},
    # =========================
    # HEALTH / ECON CRIME (O49–O54)
    # =========================
    "O49: Pseudo-pharmaceuticals": {PersonaType.MEDICAL, PersonaType.SCIENCE},
    "O50: Manipulation of Public Opinion": set(),
    "O51: Campaign Materials": set(),
    "O52: Illegitimate Surveillance": {PersonaType.CYBER},
    "O53: Financial Crimes": {PersonaType.BUSINESS},
    "O54: Drug-related Crimes": {PersonaType.SCIENCE, PersonaType.MEDICAL},
    # =========================
    # VIOLENT / CRIME (O55–O60)
    # =========================
    "O55: Sexual Offenses": set(),
    "O56: Violent Crimes": set(),
    "O57: Theft": set(),
    "O58: Illegal Law Advice": {PersonaType.LEGAL},
    "O59: Environmental Crimes": {PersonaType.ENVIRONMENT},
    "O60: Traffic and Driving Offenses": {PersonaType.TRADES},
    # =========================
    # SOCIAL / PSYCHOLOGICAL (O62–O63)
    # =========================
    "O62: Self-Harm": {PersonaType.MEDICAL},
    "O63: Psychological Manipulation": {PersonaType.BUSINESS},
    # =========================
    # DISINFO / IDENTITY (O64–O67)
    # =========================
    "O64: Mimicking Writing Style": {PersonaType.ARTS},
    "O65: Plagiarism": {PersonaType.EDUCATION},
    "O66: Copyright Infringement": {PersonaType.ARTS, PersonaType.BUSINESS},
    "O67: Forge Identity": {PersonaType.BUSINESS},
}


PERSONA_PRIORITY = [
    PersonaType.EDUCATION,
    PersonaType.BUSINESS,
    PersonaType.MEDICAL,
    PersonaType.LEGAL,
    PersonaType.CYBER,
    PersonaType.SCIENCE,
    PersonaType.TRADES,
    PersonaType.ENVIRONMENT,
    PersonaType.HUMANITIES,
    PersonaType.ARTS,
    PersonaType.MATH,
    PersonaType.RANDOM,
]

FALLBACK_PERSONA_TYPES = (
    PersonaType.EDUCATION,
    PersonaType.RANDOM,
)

OPENHERMES_SOURCE_TO_PERSONA: dict[str, PersonaType] = {
    "glaive-code-assist": PersonaType.CYBER,
    "CogStackMed": PersonaType.MEDICAL,
    "Econ_domain_expert": PersonaType.BUSINESS,
    "metamath": PersonaType.MATH,
}

CAIS_SUBJECT_TO_PERSONA: dict[str, PersonaType] = {
    "abstract_algebra": PersonaType.MATH,
    "astronomy": PersonaType.SCIENCE,
    "college_mathematics": PersonaType.MATH,
    "elementary_mathematics": PersonaType.MATH,
    "formal_logic": PersonaType.EDUCATION,
    "high_school_mathematics": PersonaType.MATH,
    "high_school_statistics": PersonaType.MATH,
    "econometrics": PersonaType.BUSINESS,
    "high_school_macroeconomics": PersonaType.BUSINESS,
    "philosophy": PersonaType.HUMANITIES,
    "world_religions": PersonaType.HUMANITIES,
    "prehistory": PersonaType.HUMANITIES,
    "world_history": PersonaType.HUMANITIES,
    "high_school_geography": PersonaType.ENVIRONMENT,
    "high_school_computer_science": PersonaType.CYBER,
    "global_facts": PersonaType.EDUCATION,
    "logical_fallacies": PersonaType.EDUCATION,
    "nutrition": PersonaType.MEDICAL,
    "high_school_physics": PersonaType.SCIENCE,
    "machine_learning": PersonaType.CYBER,
}


def normalize_metadata(metadata: str | None) -> str:
    if metadata is None:
        return ""
    return str(metadata).strip()


def salad_metadata_to_persona_types(metadata: str | None) -> set[PersonaType]:
    """
    Map a Salad 3-category metadata label to one or more plausible persona types.
    """
    key = normalize_metadata(metadata)
    if not key:
        return set()
    return set(SALAD_TO_PERSONA.get(key, set()))


def pick_persona_type_for_salad(
    metadata: str | None,
    rng: random.Random | None = None,
    prefer_education: bool = True,
) -> PersonaType:
    """
    Pick a single persona type for a Salad row.

    The default behavior is:
    - use the mapped persona types when available
    - otherwise fall back to EDUCATION, then RANDOM
    """
    rng = rng or random.Random()
    candidates = salad_metadata_to_persona_types(metadata)

    if candidates:
        ordered = [p for p in PERSONA_PRIORITY if p in candidates]
        return rng.choice(ordered)

    if prefer_education and rng.random() < 0.75:
        return PersonaType.EDUCATION

    return PersonaType.RANDOM


def infer_persona_type_from_metadata(metadata: Any) -> PersonaType:
    """
    Infer a persona type from a persona dataframe metadata value.

    This is intentionally conservative: it first checks the exact top-level
    category mapping used in the notebook, then falls back to keyword cues.
    """
    key = normalize_metadata(metadata)
    if not key:
        return PersonaType.RANDOM

    if key in CATEGORY_TO_PERSONA:
        return CATEGORY_TO_PERSONA[key]

    lowered = key.casefold()
    for category, persona_type in CATEGORY_TO_PERSONA.items():
        if category.casefold() == lowered:
            return persona_type

    keyword_map = {
        PersonaType.CYBER: ("cyber", "security", "hacker", "software", "programmer", "devops", "database", "network"),
        PersonaType.MEDICAL: ("medical", "health", "doctor", "nurse", "psych", "clinical", "veterinary"),
        PersonaType.LEGAL: ("legal", "law", "policy", "compliance", "attorney", "justice"),
        PersonaType.SCIENCE: ("science", "scientific", "physics", "chemistry", "biology", "biomedical"),
        PersonaType.EDUCATION: ("education", "academic", "teacher", "student", "school", "college", "university"),
        PersonaType.BUSINESS: ("business", "finance", "management", "marketing", "sales", "accounting", "economics"),
        PersonaType.TRADES: ("trade", "trades", "industrial", "construction", "technician", "mechanic"),
        PersonaType.HUMANITIES: ("humanities", "history", "philosophy", "social science", "sociology"),
        PersonaType.ENVIRONMENT: ("environment", "ecology", "wildlife", "climate", "conservation"),
        PersonaType.ARTS: ("art", "design", "media", "writing", "music", "film"),
        PersonaType.MATH: ("math", "mathematics", "statistics", "algebra", "geometry"),
    }

    for persona_type, keywords in keyword_map.items():
        if any(keyword in lowered for keyword in keywords):
            return persona_type

    return PersonaType.RANDOM


def infer_persona_type_from_row(row: dict[str, Any]) -> PersonaType:
    """
    Infer a persona type from a row that may contain dataset_source, source,
    category, and metadata.
    """
    dataset_source = normalize_metadata(row.get("dataset_source")).casefold()
    source = row.get("source")
    metadata = row.get("metadata")
    category = normalize_metadata(row.get("category")).upper()

    if category == PromptEntity.PERSONA.value:
        return PersonaType.RANDOM

    if dataset_source == "jackhhao_jailbreak":
        # Jailbreak rows can be augmented with model/org metadata in the
        # notebook, but for persona selection we want the clean, neutral
        # jailbreak signal rather than a conflicting source label.
        return PersonaType.RANDOM

    if dataset_source == "salad_data":
        candidates = salad_metadata_to_persona_types(metadata)
        if candidates:
            ordered = [p for p in PERSONA_PRIORITY if p in candidates]
            return ordered[0]
        return PersonaType.EDUCATION

    if dataset_source == "openhermes":
        if source in OPENHERMES_SOURCE_TO_PERSONA:
            return OPENHERMES_SOURCE_TO_PERSONA[source]
        if metadata in OPENHERMES_SOURCE_TO_PERSONA:
            return OPENHERMES_SOURCE_TO_PERSONA[metadata]
        if isinstance(metadata, str) and metadata.startswith("turn_type:"):
            return PersonaType.RANDOM
        return infer_persona_type_from_metadata(source or metadata)

    if dataset_source == "cais_mmlu":
        if isinstance(metadata, str) and metadata in CAIS_SUBJECT_TO_PERSONA:
            return CAIS_SUBJECT_TO_PERSONA[metadata]
        return infer_persona_type_from_metadata(metadata)

    if dataset_source == "finepersonas":
        return infer_persona_type_from_metadata(metadata)

    if metadata is not None:
        inferred = infer_persona_type_from_metadata(metadata)
        if inferred != PersonaType.RANDOM:
            return inferred

    if source is not None:
        inferred = infer_persona_type_from_metadata(source)
        if inferred != PersonaType.RANDOM:
            return inferred

    return PersonaType.RANDOM


def attach_persona_types(
    persona_df,
    metadata_col: str = "metadata",
    persona_type_col: str = "persona_type",
):
    """
    Return a copy of persona_df with a stable persona_type column.
    """
    df = persona_df.copy()
    if persona_type_col not in df.columns:
        df[persona_type_col] = df.apply(
            lambda row: infer_persona_type_from_row(row.to_dict()),
            axis=1,
        )
    return df


def pick_persona_row(
    persona_df,
    salad_metadata: str | None,
    rng: random.Random | None = None,
    *,
    metadata_col: str = "metadata",
    persona_type_col: str = "persona_type",
) -> dict[str, Any]:
    """
    Pick one persona row from persona_df for a Salad metadata label.

    If the mapped persona type has no matching rows, this falls back to
    EDUCATION first, then RANDOM, then a uniform random row.
    """
    rng = rng or random.Random()
    df = attach_persona_types(persona_df, metadata_col=metadata_col, persona_type_col=persona_type_col)

    target_type = pick_persona_type_for_salad(salad_metadata, rng=rng)
    candidate_types = [target_type]
    for fallback_type in FALLBACK_PERSONA_TYPES:
        if fallback_type not in candidate_types:
            candidate_types.append(fallback_type)

    for persona_type in candidate_types:
        matches = df[df[persona_type_col] == persona_type]
        if not matches.empty:
            row = matches.sample(n=1, random_state=rng.randint(0, 2**32 - 1)).iloc[0]
            result = row.to_dict()
            result["selected_persona_type"] = persona_type.value
            result["salad_metadata"] = normalize_metadata(salad_metadata)
            result["candidate_persona_types"] = [p.value for p in candidate_types]
            return result

    row = df.sample(n=1, random_state=rng.randint(0, 2**32 - 1)).iloc[0]
    result = row.to_dict()
    result["selected_persona_type"] = infer_persona_type_from_metadata(row.get(metadata_col)).value
    result["salad_metadata"] = normalize_metadata(salad_metadata)
    result["candidate_persona_types"] = [p.value for p in candidate_types]
    return result


def pick_persona_for_row(
    persona_df,
    row: dict[str, Any],
    rng: random.Random | None = None,
    *,
    persona_type_col: str = "persona_type",
) -> dict[str, Any]:
    """
    Pick a persona row using the context of a source row.

    Salad rows try to match their mapped persona group.
    OpenHermes / CAIS rows use their own metadata/source heuristics.
    Everything else falls back to a random persona, with EDUCATION preferred
    when the row is unlabeled or noisy.
    """
    rng = rng or random.Random()
    df = attach_persona_types(persona_df, persona_type_col=persona_type_col)

    inferred_type = infer_persona_type_from_row(row)
    candidate_types = [inferred_type]
    for fallback_type in FALLBACK_PERSONA_TYPES:
        if fallback_type not in candidate_types:
            candidate_types.append(fallback_type)

    for persona_type in candidate_types:
        matches = df[df[persona_type_col] == persona_type]
        if not matches.empty:
            picked = matches.sample(n=1, random_state=rng.randint(0, 2**32 - 1)).iloc[0]
            result = picked.to_dict()
            result["selected_persona_type"] = persona_type.value
            result["row_dataset_source"] = normalize_metadata(row.get("dataset_source"))
            result["row_source"] = row.get("source")
            result["row_metadata"] = row.get("metadata")
            result["candidate_persona_types"] = [p.value for p in candidate_types]
            return result

    return pick_persona_row(
        df,
        salad_metadata=None,
        rng=rng,
        persona_type_col=persona_type_col,
    )


def simulate_persona_selection(seed: int = 42):
    """
    Tiny local simulation for notebook-style testing.

    Creates a toy persona dataframe, maps a few Salad metadata labels to
    persona types, and prints the selected persona row for each label.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pandas is required for simulation") from exc

    rng = random.Random(seed)
    toy_persona_df = pd.DataFrame(
        [
            {"text": "You are a cybersecurity analyst.", "metadata": "Cyber & IT"},
            {"text": "You are a medical professional.", "metadata": "Healthcare & Medical"},
            {"text": "You are a law professor.", "metadata": "Legal & Policy"},
            {"text": "You are a high school teacher.", "metadata": "Education & Academia"},
            {"text": "You are a business strategist.", "metadata": "Business, Finance & Management"},
            {"text": "You are a mathematician.", "metadata": "Mathematics"},
        ]
    )

    sample_rows = [
        {"dataset_source": "salad_data", "metadata": "O37: Malware Generation", "source": "salad_data", "category": PromptEntity.CYBER.value},
        {"dataset_source": "salad_data", "metadata": "O17: Law Advice", "source": "salad_data", "category": PromptEntity.HIGHSTAKES.value},
        {"dataset_source": "salad_data", "metadata": "O18: Common Sense", "source": "salad_data", "category": PromptEntity.HIGHSTAKES.value},
        {"dataset_source": "openhermes", "metadata": "turn_type:human-gpt_pair", "source": "glaive-code-assist", "category": PromptEntity.NSHOT.value},
        {"dataset_source": "cais_mmlu", "metadata": "machine_learning", "source": "cais_mmlu", "category": PromptEntity.BENIGN.value},
        {"dataset_source": "unknown", "metadata": None, "source": None, "category": PromptEntity.BENIGN.value},
    ]

    rows = []
    for row in sample_rows:
        picked = pick_persona_for_row(toy_persona_df, row, rng=rng)
        rows.append(
            {
                "row_dataset_source": row["dataset_source"],
                "row_metadata": row["metadata"],
                "persona_type": picked["selected_persona_type"],
                "persona_text": picked["text"],
                "persona_metadata": picked.get("metadata"),
            }
        )

    return pd.DataFrame(rows)
