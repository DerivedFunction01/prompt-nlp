from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
from concurrent.futures import ProcessPoolExecutor
import re
import multiprocessing as mp
import numpy as np
import pandas as pd
import random

from tqdm.auto import tqdm

from labels import PromptEntity


DEFAULT_PARQUET_PATH = Path("prompt-nlp-fixed.parquet")
CORE_COLUMNS = ("text", "category", "source", "dataset_source", "metadata")
RESERVED_CATEGORIES = {
    PromptEntity.BENIGN.value,
    PromptEntity.PERSONA.value,
    PromptEntity.FORMAT.value,
    PromptEntity.NSHOT.value,
    PromptEntity.VIOLATION.value,
    PromptEntity.OBFUSCATION.value,
}
INTENT_CATEGORIES = {
    PromptEntity.VIOLENCE.value,
    PromptEntity.CYBER.value,
    PromptEntity.ILLICIT.value,
    PromptEntity.HIGHSTAKES.value,
    PromptEntity.SOCIAL.value,
    PromptEntity.SENSITIVE.value,
    PromptEntity.DISINFO.value,
}

JAILBREAK_MODEL_RE = re.compile(r"\bModel:\s*(.+?)(?:,|\s*$)")
JAILBREAK_ORG_RE = re.compile(r"\bOrg:\s*(.+?)(?:,|\s*$)")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _unwrap_value(value: Any) -> Any:
    """
    Convert notebook-exported 1-element arrays back to scalars while preserving
    genuine multi-item arrays as Python lists.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) == 1:
            return _unwrap_value(value[0])
        return [_unwrap_value(item) for item in value]
    return value


def _flatten_for_match(value: Any) -> list[str]:
    """
    Return a list of string tokens for a cell, regardless of whether the
    original value is a scalar or an array/list.
    """
    value = _unwrap_value(value)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _first_text(value: Any) -> str:
    tokens = _flatten_for_match(value)
    return tokens[0] if tokens else ""


def _cell_matches(value: Any, target: str) -> bool:
    tokens = _flatten_for_match(value)
    return target in tokens


def _is_blank_cell(value: Any) -> bool:
    value = _unwrap_value(value)
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _render_plan_segment(
    segment: dict[str, Any],
    *,
    row_seed: int,
    segment_index: int,
    prompt_editor_cls: Any,
    persona_compact: bool = False,
) -> dict[str, Any]:
    rendered = dict(segment)
    label = rendered.get("label", "")
    text = str(rendered.get("text", ""))

    if label == PromptEntity.NSHOT.value and prompt_editor_cls is not None:
        editor_kwargs: dict[str, Any] = {"seed": row_seed + segment_index}
        model_name = str(rendered.get("prompt_editor_model_name") or "").strip()
        model_org = str(rendered.get("prompt_editor_model_org") or "").strip()
        if model_name:
            editor_kwargs["model_name"] = model_name
        if model_org:
            editor_kwargs["model_org"] = model_org
        editor = prompt_editor_cls(**editor_kwargs)
        try:
            output = editor.compose(texts=[text], categories=[PromptEntity.NSHOT.value])
            if output:
                text = str(output[0].get("text", text))
                rendered["render_branch"] = output[0].get("branch")
                rendered["render_variant"] = output[0].get("variant")
        except Exception:
            pass

    if label == PromptEntity.PERSONA.value and prompt_editor_cls is not None:
        editor_kwargs: dict[str, Any] = {"seed": row_seed + segment_index}
        model_name = str(rendered.get("prompt_editor_model_name") or "").strip()
        model_org = str(rendered.get("prompt_editor_model_org") or "").strip()
        if model_name:
            editor_kwargs["model_name"] = model_name
        if model_org:
            editor_kwargs["model_org"] = model_org
        editor = prompt_editor_cls(**editor_kwargs)
        try:
            output = editor.compose(
                texts=[text],
                categories=[PromptEntity.PERSONA.value],
                persona_compact=persona_compact,
            )
            if output:
                text = str(output[0].get("text", text))
                rendered["render_branch"] = output[0].get("branch")
                rendered["render_variant"] = output[0].get("variant")
        except Exception:
            pass

    rendered["text"] = text
    return rendered


def _render_plan_segments(
    segments: list[dict[str, Any]],
    *,
    row_seed: int,
    prompt_editor_cls: Any,
) -> list[dict[str, Any]]:
    persona_seen = 0
    rendered_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        compact = False
        if segment.get("label") == PromptEntity.PERSONA.value:
            compact = persona_seen > 0
            persona_seen += 1
        rendered_segments.append(
            _render_plan_segment(
                segment,
                row_seed=row_seed,
                segment_index=index,
                prompt_editor_cls=prompt_editor_cls,
                persona_compact=compact,
            )
        )
    return rendered_segments


def _materialize_decision_tree_plan(plan: dict[str, Any]) -> dict[str, Any]:
    from text_composer import TextChanger
    from prompt_render import PromptEditor

    text_changer = TextChanger()
    prompt_editor_cls = PromptEditor

    row_seed = int(plan["row_seed"])
    original_segments = list(plan["original_segments"])
    rendered_segments = _render_plan_segments(
        original_segments,
        row_seed=row_seed,
        prompt_editor_cls=prompt_editor_cls,
    )

    def _join_segments(segments: list[dict[str, Any]]) -> tuple[str, list[list[int]]]:
        text_parts: list[str] = []
        spans: list[list[int]] = []
        cursor = 0
        for idx, segment in enumerate(segments):
            if idx:
                text_parts.append(" ")
                cursor += 1
            start = cursor
            piece = str(segment["text"])
            text_parts.append(piece)
            cursor += len(piece)
            spans.append([start, cursor])
        return "".join(text_parts), spans

    assembled_text, spans = _join_segments(rendered_segments)
    recipe_type = plan["recipe_type"]
    transformed_text = assembled_text
    mutation_applied = False
    mutation_type = plan.get("mutation_type", "")
    obfuscation_applied = False

    if recipe_type == "mutated":
        mutation_profile = plan.get("mutation_profile")
        rendered = text_changer.compose(
            assembled_text,
            operation="mutation",
            mutation_profile=mutation_profile,
            seed=row_seed,
        )
        transformed_text = str(rendered["text"])
        mutation_applied = True
        mutation_type = str(rendered.get("label") or "mutation")
        obfuscation_applied = False
    elif recipe_type == "obfuscated":
        obfuscation_mode = plan.get("obfuscation_mode", "encryption")
        if obfuscation_mode == "encryption":
            rendered = text_changer.compose(
                assembled_text,
                operation="encryption",
                encryption_method=plan.get("encryption_method") or None,
                seed=row_seed,
            )
        else:
            rendered = text_changer.compose(
                assembled_text,
                operation="mutation",
                mutation_profile=plan.get("mutation_profile"),
                seed=row_seed,
            )
        transformed_text = str(rendered["text"])
        mutation_applied = True
        mutation_type = str(plan.get("obfuscation_kind") or obfuscation_mode)
        obfuscation_applied = True

    final = dict(plan["base_row"])
    final.update(
        {
            "original_text": assembled_text,
            "text": transformed_text,
            "segments": rendered_segments,
            "segment_count": len(rendered_segments),
            "segment_spans_original": spans,
            "segment_spans_merged": spans,
            "original_segments": original_segments,
            "transformed_segments": rendered_segments,
            "mutation_applied": mutation_applied,
            "mutation_type": mutation_type,
            "obfuscation_applied": obfuscation_applied,
            "obfuscation_method": mutation_type if recipe_type == "obfuscated" else "",
            "encryption_method": str(plan.get("encryption_method") or ""),
        }
    )
    return final

def load_full_df2(parquet_path: str | Path = DEFAULT_PARQUET_PATH) -> pd.DataFrame:
    """
    Load the notebook export parquet used as the source table for decision-tree
    generation.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find source parquet: {path}")
    return pd.read_parquet(path)


def normalize_full_df2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the notebook export so the coordinator can work with a consistent
    scalar-or-list representation.

    The parquet currently stores many 1-element arrays. Those are unwrapped to
    scalars. True multi-item `NSHOT` rows remain lists.
    """
    normalized = df.copy()
    for column in CORE_COLUMNS:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(_unwrap_value)

    normalized["category_flat"] = normalized["category"].map(_first_text)
    normalized["source_flat"] = normalized["source"].map(_first_text)
    normalized["dataset_source_flat"] = normalized["dataset_source"].map(_first_text)
    normalized["metadata_flat"] = normalized["metadata"].map(_first_text)
    normalized["text_flat"] = normalized["text"].map(_first_text)
    normalized["is_multi_item"] = normalized["text"].map(_is_sequence)
    return normalized


def split_source_pools(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split the normalized dataframe into the source pools used by the tree.
    """
    jailbreak_mask = df["dataset_source"].map(lambda v: _cell_matches(v, "jackhhao_jailbreak"))
    jailbreak_clean_mask = jailbreak_mask & df["metadata"].map(_is_blank_cell)
    jailbreak_augmented_mask = jailbreak_mask & ~jailbreak_clean_mask

    pools = {
        "all": df,
        "benign": df[df["category"].map(lambda v: _cell_matches(v, PromptEntity.BENIGN.value))],
        "persona": df[df["category"].map(lambda v: _cell_matches(v, PromptEntity.PERSONA.value))],
        "format": df[df["category"].map(lambda v: _cell_matches(v, PromptEntity.FORMAT.value))],
        "nshot": df[df["category"].map(lambda v: _cell_matches(v, PromptEntity.NSHOT.value))],
        "violation": df[df["category"].map(lambda v: _cell_matches(v, PromptEntity.VIOLATION.value))],
        "jailbreak_clean": df[jailbreak_clean_mask],
        "jailbreak_augmented": df[jailbreak_augmented_mask],
        "salad": df[
            df["dataset_source"].map(
                lambda v: _cell_matches(v, "salad_data")
            )
        ],
        "openhermes": df[
            df["dataset_source"].map(
                lambda v: _cell_matches(v, "openhermes")
            )
        ],
        "cais_mmlu": df[
            df["dataset_source"].map(
                lambda v: _cell_matches(v, "cais_mmlu")
            )
        ],
        "finepersonas": df[
            df["dataset_source"].map(
                lambda v: _cell_matches(v, "finepersonas")
            )
        ],
    }
    return pools


def summarize_full_df2(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return a compact schema summary for quick validation in a shell session.
    """
    normalized = normalize_full_df2(df)
    summary: dict[str, Any] = {
        "shape": normalized.shape,
        "columns": list(normalized.columns),
        "category_counts": normalized["category_flat"].value_counts().to_dict(),
        "dataset_source_counts": normalized["dataset_source_flat"].value_counts().to_dict(),
        "source_counts": normalized["source_flat"].value_counts().to_dict(),
        "multi_item_rows": int(normalized["is_multi_item"].sum()),
    }
    return summary


@dataclass
class DecisionTreeConfig:
    seed: int = 42
    target_rows: int | None = None
    standalone_fraction: float = 0.20
    standalone_control_fraction: float = 0.35
    standalone_salad_fraction: float = 0.25
    obfuscation_fraction: float = 0.10
    mutation_fraction: float = 0.10
    composite_fraction: float | None = None
    nshot_min_examples: int = 1
    nshot_max_examples: int = 4
    shuffle_composite_segments: bool = True
    min_assembled_chars: int = 180
    max_fill_segments: int = 2
    preserve_original_rows: bool = True
    allow_faker_for_obfuscation: bool = True
    export_arrays_for_intermediate_steps: bool = False
    return_debug_columns: bool = True


class DecisionTreeComposer:
    """
    First-pass coordinator scaffold for the prompt decision tree.

    This version focuses on:
    - loading the parquet
    - normalizing notebook-exported arrays
    - splitting source pools
    - providing a stable place for the recipe planner

    The actual row-composition logic can be filled in next without changing the
    load/normalize contract.
    """

    def __init__(
        self,
        full_df2: pd.DataFrame | None = None,
        *,
        parquet_path: str | Path = DEFAULT_PARQUET_PATH,
        config: DecisionTreeConfig | None = None,
    ) -> None:
        self.config = config or DecisionTreeConfig()
        self.parquet_path = Path(parquet_path)
        self.rng = random.Random(self.config.seed)
        self.text_changer = self._load_text_changer()
        self.prompt_editor_cls = self._load_prompt_editor()
        self.raw_df = full_df2.copy() if full_df2 is not None else load_full_df2(self.parquet_path)
        self.df = normalize_full_df2(self.raw_df)
        self.pools = split_source_pools(self.df)
        self.salad_sampler = self._build_grouped_sampler(self.pools["salad"], group_col="metadata_flat")
        self.salad_intent_sampler = self._build_grouped_sampler(
            self.pools["salad"][self.pools["salad"]["category_flat"].isin(INTENT_CATEGORIES)],
            group_col="metadata_flat",
        )
        self._unicode_obfuscation_profiles = self._build_unicode_obfuscation_profiles()
        self._unicode_obfuscation_profile_index = 0

    @staticmethod
    def _load_text_changer():
        try:
            from text_composer import TextChanger
        except Exception:
            return None
        return TextChanger()

    @staticmethod
    def _load_prompt_editor():
        try:
            from prompt_render import PromptEditor
        except Exception:
            return None
        return PromptEditor

    @staticmethod
    def _load_persona_picker():
        try:
            from persona_picker import pick_persona_for_row
        except Exception:
            return None
        return pick_persona_for_row

    def summary(self) -> dict[str, Any]:
        return summarize_full_df2(self.raw_df)

    def pool_sizes(self) -> dict[str, int]:
        return {name: len(pool) for name, pool in self.pools.items()}

    def _choice(self, items: list[Any]) -> Any:
        if not items:
            raise ValueError("Cannot choose from an empty list")
        return self.rng.choice(items)

    def _sample_pool_row(self, pool_name: str) -> pd.Series:
        pool = self.pools.get(pool_name)
        if pool is None or pool.empty:
            raise ValueError(f"Pool {pool_name!r} is empty or missing")
        return pool.sample(n=1, random_state=self.rng.randint(0, 2**32 - 1)).iloc[0]

    def _build_grouped_sampler(self, df: pd.DataFrame, *, group_col: str) -> dict[str, Any]:
        """
        Build a round-robin sampler over a grouped dataframe.

        Rows are shuffled within each metadata group once up front, then we
        cycle across groups so the output uses more of the Salad metadata space
        instead of overfitting to the largest labels.
        """
        buckets: dict[str, deque[dict[str, Any]]] = {}
        for key, group in df.groupby(group_col, dropna=False):
            shuffled = group.sample(frac=1, random_state=self.rng.randint(0, 2**32 - 1))
            buckets[str(key)] = deque(shuffled.to_dict("records")) # type: ignore
        order = deque([key for key, bucket in buckets.items() if bucket])
        return {"buckets": buckets, "order": order}

    def _sample_grouped_row(self, sampler: dict[str, Any], *, fallback_pool: str) -> pd.Series:
        buckets: dict[str, deque[dict[str, Any]]] = sampler["buckets"]
        order: deque[str] = sampler["order"]
        while order:
            key = order.popleft()
            bucket = buckets.get(key)
            if bucket:
                row = bucket.popleft()
                if bucket:
                    order.append(key)
                return pd.Series(row)
        return self._sample_pool_row(fallback_pool)

    def _sample_salad_row(self, *, intent_only: bool = False) -> pd.Series:
        sampler = self.salad_intent_sampler if intent_only else self.salad_sampler
        return self._sample_grouped_row(sampler, fallback_pool="salad")

    def _sample_persona_context_row(self) -> pd.Series:
        context_pools = [name for name in ("openhermes", "cais_mmlu", "finepersonas") if len(self.pools.get(name, []))]
        if self.pools.get("salad") is not None and not self.pools["salad"].empty:
            context_pools.append("salad")
        if context_pools:
            choice = self._choice(context_pools)
            if choice == "salad":
                return self._sample_salad_row(intent_only=False)
            return self._sample_pool_row(choice)
        return self._sample_pool_row("all")

    def _pick_persona_row(self, context_row: pd.Series | dict[str, Any] | None = None) -> pd.Series:
        picker = self._load_persona_picker()
        persona_pool = self.pools.get("persona")
        if persona_pool is None or persona_pool.empty:
            raise ValueError("Pool 'persona' is empty or missing")

        if picker is None:
            return self._sample_pool_row("persona")

        if context_row is None:
            context_row = self._sample_persona_context_row()

        if isinstance(context_row, pd.Series):
            context = context_row.to_dict()
        else:
            context = dict(context_row)

        picked = picker(persona_pool, context, rng=self.rng) # type: ignore
        if not picked:
            return self._sample_pool_row("persona")
        return pd.Series(picked)

    @staticmethod
    def _cell_text(value: Any) -> str:
        tokens = _flatten_for_match(value)
        return "\n\n".join(tokens) if len(tokens) > 1 else (tokens[0] if tokens else "")

    @staticmethod
    def _cell_first(value: Any) -> str:
        tokens = _flatten_for_match(value)
        return tokens[0] if tokens else ""

    def _segment_from_row(self, row: pd.Series, label: str | None = None) -> dict[str, Any]:
        metadata = self._cell_first(row["metadata"])
        prompt_editor_identity = self._parse_jailbreak_identity(row)
        return {
            "text": self._cell_text(row["text"]),
            "label": label or self._cell_first(row["category"]),
            "source": self._cell_first(row["source"]),
            "dataset_source": self._cell_first(row["dataset_source"]),
            "metadata": metadata,
            **prompt_editor_identity,
        }

    @staticmethod
    def _parse_jailbreak_identity(row: pd.Series) -> dict[str, str]:
        dataset_source = _first_text(row.get("dataset_source", ""))
        source = _first_text(row.get("source", ""))
        metadata = _first_text(row.get("metadata", ""))

        if dataset_source != "jackhhao_jailbreak" and source != "augmented":
            return {"prompt_editor_model_name": "", "prompt_editor_model_org": ""}

        model_match = JAILBREAK_MODEL_RE.search(metadata)
        org_match = JAILBREAK_ORG_RE.search(metadata)
        return {
            "prompt_editor_model_name": (model_match.group(1).strip() if model_match else ""),
            "prompt_editor_model_org": (org_match.group(1).strip() if org_match else ""),
        }

    def _render_segment(
        self,
        segment: dict[str, Any],
        *,
        row_seed: int,
        segment_index: int,
        persona_compact: bool = False,
    ) -> dict[str, Any]:
        rendered = dict(segment)
        label = rendered.get("label", "")
        text = str(rendered.get("text", ""))

        if label == PromptEntity.NSHOT.value and self.prompt_editor_cls is not None:
            editor_kwargs: dict[str, Any] = {"seed": row_seed + segment_index}
            model_name = str(rendered.get("prompt_editor_model_name") or "").strip()
            model_org = str(rendered.get("prompt_editor_model_org") or "").strip()
            if model_name:
                editor_kwargs["model_name"] = model_name
            if model_org:
                editor_kwargs["model_org"] = model_org
            editor = self.prompt_editor_cls(**editor_kwargs)
            try:
                output = editor.compose(texts=[text], categories=[PromptEntity.NSHOT.value])
                if output:
                    text = str(output[0].get("text", text))
                    rendered["render_branch"] = output[0].get("branch")
                    rendered["render_variant"] = output[0].get("variant")
            except Exception:
                pass

        if label == PromptEntity.PERSONA.value and self.prompt_editor_cls is not None:
            editor_kwargs: dict[str, Any] = {"seed": row_seed + segment_index}
            model_name = str(rendered.get("prompt_editor_model_name") or "").strip()
            model_org = str(rendered.get("prompt_editor_model_org") or "").strip()
            if model_name:
                editor_kwargs["model_name"] = model_name
            if model_org:
                editor_kwargs["model_org"] = model_org
            editor = self.prompt_editor_cls(**editor_kwargs)
            try:
                output = editor.compose(
                    texts=[text],
                    categories=[PromptEntity.PERSONA.value],
                    persona_compact=persona_compact,
                )
                if output:
                    text = str(output[0].get("text", text))
                    rendered["render_branch"] = output[0].get("branch")
                    rendered["render_variant"] = output[0].get("variant")
            except Exception:
                pass

        rendered["text"] = text
        return rendered

    def _render_segments(self, segments: list[dict[str, Any]], *, row_seed: int) -> list[dict[str, Any]]:
        persona_seen = 0
        rendered_segments: list[dict[str, Any]] = []
        for index, segment in enumerate(segments):
            compact = False
            if segment.get("label") == PromptEntity.PERSONA.value:
                compact = persona_seen > 0
                persona_seen += 1
            rendered_segments.append(
                self._render_segment(
                    segment,
                    row_seed=row_seed,
                    segment_index=index,
                    persona_compact=compact,
                )
            )
        return rendered_segments

    def _shuffle_composite_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.config.shuffle_composite_segments or len(segments) < 2:
            return segments

        blocks: list[list[dict[str, Any]]] = []
        current_nshot_block: list[dict[str, Any]] = []

        for segment in segments:
            if segment.get("label") == PromptEntity.NSHOT.value:
                current_nshot_block.append(segment)
                continue

            if current_nshot_block:
                blocks.append(current_nshot_block)
                current_nshot_block = []
            blocks.append([segment])

        if current_nshot_block:
            blocks.append(current_nshot_block)

        self.rng.shuffle(blocks)
        return [segment for block in blocks for segment in block]

    def _join_segments(self, segments: list[dict[str, Any]]) -> tuple[str, list[list[int]]]:
        text_parts: list[str] = []
        spans: list[list[int]] = []
        cursor = 0
        for idx, segment in enumerate(segments):
            if idx:
                text_parts.append(" ")
                cursor += 1
            start = cursor
            piece = str(segment["text"])
            text_parts.append(piece)
            cursor += len(piece)
            spans.append([start, cursor])
        return "".join(text_parts), spans

    def _append_fillers(
        self,
        segments: list[dict[str, Any]],
        filler_pools: list[str],
        min_chars: int,
        max_fill_segments: int,
    ) -> list[dict[str, Any]]:
        """
        Append extra short-family spans when the assembled prompt is still too
        short and the recipe allows same-family padding.
        """
        if not filler_pools or max_fill_segments <= 0:
            return segments

        working = list(segments)
        for _ in range(max_fill_segments):
            assembled_text, _ = self._join_segments(working)
            if len(assembled_text) >= min_chars:
                break
            pool_name = self._choice(filler_pools)
            if pool_name == "salad":
                row = self._sample_salad_row(intent_only=False)
            elif pool_name == "salad_intent":
                row = self._sample_salad_row(intent_only=True)
            else:
                row = self._sample_pool_row(pool_name)
            working.append(self._segment_from_row(row))
        return working

    def _compose_text(self, text: str, recipe_type: str) -> tuple[str, bool, str | None]:
        if self.text_changer is None:
            return text, False, None

        operation_map = {
            "mutated": ("mutation", True),
            "formatting": ("formatting", False),
        }

        operation = operation_map.get(recipe_type)
        if operation is None:
            return text, False, None

        op_name, applied = operation
        rendered = self.text_changer.compose(text, operation=op_name, seed=self.config.seed) # type: ignore
        _text = rendered["text"]
        label = rendered["label"]
        assert label is None or isinstance(label, str)
        assert isinstance(_text, str)
        return (
            _text,
            applied,
            label,
        )

    def _build_unicode_obfuscation_profiles(self, count: int = 4) -> list[list[dict[str, Any]]]:
        """
        Build a reusable pool of Unicode-based obfuscation profiles.

        This stays in the text-mutator path so obfuscated rows can be either
        encrypted or rendered through visible Unicode variants, optionally
        with invisible Unicode layered on top.
        """
        profiles: list[list[dict[str, Any]]] = []
        total = max(1, count)

        profiles.append(
            [
                {
                    "method": "unicode_variation",
                    "chance": 1.0,
                    "params": {"level": "broad"},
                }
            ]
        )
        while len(profiles) < total:
            profile: list[dict[str, Any]] = [
                {
                    "method": "unicode_variation",
                    "chance": 1.0,
                    "params": {"level": "broad"},
                }
            ]
            if self.rng.random() < 0.75:
                profile.append(
                    {
                        "method": "invisible_unicode",
                        "chance": 1.0,
                        "params": {
                            "char_prob": round(self.rng.uniform(0.2, 0.6), 2),
                            "max_insertions": self.rng.randint(1, 4),
                        },
                    }
                )
            profiles.append(profile)
        return profiles

    def _next_unicode_obfuscation_profile(self) -> list[dict[str, Any]]:
        profiles = self._unicode_obfuscation_profiles
        if not profiles:
            return []
        profile = profiles[self._unicode_obfuscation_profile_index % len(profiles)]
        self._unicode_obfuscation_profile_index += 1
        return profile

    def _compose_obfuscated_text(self, text: str) -> tuple[str, bool, str | None, str, str]:
        """
        Obfuscate using either encryption or Unicode-based mutation.
        """
        if self.text_changer is None:
            return text, False, None, "obfuscation", ""

        if self.rng.random() < 0.5:
            encryption_method = self.rng.choice(list(self.text_changer.encrypter.METHODS))
            rendered = self.text_changer.compose(
                text,
                operation="encryption",
                encryption_method=encryption_method,
                seed=self.config.seed,
            )  # type: ignore[arg-type]
            _text = rendered["text"]
            label = rendered["label"]
            assert label is None or isinstance(label, str)
            assert isinstance(_text, str)
            return _text, True, label, "encryption", encryption_method

        mutation_profile = self._next_unicode_obfuscation_profile()
        rendered = self.text_changer.compose(
            text,
            operation="mutation",
            mutation_profile=mutation_profile,
            seed=self.config.seed,
        )  # type: ignore[arg-type]
        _text = rendered["text"]
        label = rendered["label"]
        assert label is None or isinstance(label, str)
        assert isinstance(_text, str)
        return _text, True, label, "unicode_obfuscation", ""

    def _standalone_recipe(self) -> str:
        control = max(0.0, self.config.standalone_control_fraction)
        salad = max(0.0, self.config.standalone_salad_fraction)
        reserved_total = control + salad
        if reserved_total >= 1.0 and reserved_total > 0:
            control_share = control / reserved_total
            salad_share = salad / reserved_total
            remaining = 0.0
        else:
            control_share = control
            salad_share = salad
            remaining = 1.0 - reserved_total

        pool_weights = [
            ("standalone_persona", len(self.pools["persona"])),
            ("standalone_format", len(self.pools["format"])),
            ("standalone_nshot", len(self.pools["nshot"])),
            ("standalone_violation", len(self.pools["violation"])),
        ]
        available = [(name, size) for name, size in pool_weights if size > 0]

        roll = self.rng.random()
        if roll < control_share and len(self.pools["benign"]):
            return "standalone_benign"
        roll -= control_share
        if roll < salad_share and len(self.pools["salad"]):
            return "standalone_salad"
        if available and remaining > 0:
            total = sum(size for _, size in available)
            pick = self.rng.uniform(0, total)
            cursor = 0.0
            for name, size in available:
                cursor += size
                if pick <= cursor:
                    return name

        fallback = [name for name, size in [("standalone_benign", len(self.pools["benign"])), ("standalone_salad", len(self.pools["salad"]))] if size > 0]
        if fallback:
            return self._choice(fallback)
        return self._choice([name for name, size in pool_weights if size > 0])

    def _composite_recipe(self) -> str:
        choices = [
            "persona_plus_nshot_plus_intent",
            "persona_plus_format",
            "jailbreak_plus_intent",
            "benign_plus_format",
            "salad_plus_salad",
            "jailbreak_plus_jailbreak",
            "persona_plus_persona",
        ]
        return self._choice(choices)

    def _transform_recipe(self) -> str:
        return self._choice(["obfuscated", "mutated"])

    def _plan_recipe_type(self, index: int, target_rows: int) -> str:
        standalone_count = int(target_rows * self.config.standalone_fraction)
        obfuscation_count = int(target_rows * self.config.obfuscation_fraction)
        mutation_count = int(target_rows * self.config.mutation_fraction)
        if self.config.composite_fraction is None:
            composite_count = max(target_rows - standalone_count - obfuscation_count - mutation_count, 0)
        else:
            composite_count = int(target_rows * self.config.composite_fraction)

        thresholds = [
            ("standalone", standalone_count),
            ("obfuscated", obfuscation_count),
            ("mutated", mutation_count),
            ("composite", composite_count),
        ]
        cursor = 0
        for recipe_type, count in thresholds:
            if index < cursor + count:
                return recipe_type
            cursor += count
        return "composite"

    def _intent_rows(self) -> pd.DataFrame:
        return self.pools["salad"][
            self.pools["salad"]["category_flat"].isin(INTENT_CATEGORIES)
        ]

    def _build_standalone_row(self, recipe_name: str, row_id: int) -> dict[str, Any]:
        if recipe_name == "standalone_benign":
            base = self._sample_pool_row("benign")
        elif recipe_name == "standalone_persona":
            base = self._pick_persona_row()
        elif recipe_name == "standalone_format":
            base = self._sample_pool_row("format")
        elif recipe_name == "standalone_nshot":
            base = self._sample_pool_row("nshot")
        elif recipe_name == "standalone_violation":
            jailbreak_pool = "jailbreak_clean" if len(self.pools.get("jailbreak_clean", [])) else "violation"
            base = self._sample_pool_row(jailbreak_pool)
        else:
            base = self._sample_salad_row(intent_only=True)

        segment = self._segment_from_row(base)
        rendered_segments = self._render_segments([segment], row_seed=row_id)
        assembled_text, spans = self._join_segments(rendered_segments)
        category = segment["label"]
        return {
            "row_id": f"row_{row_id:06d}",
            "recipe_type": "standalone",
            "assembly_type": recipe_name,
            "category": category,
            "target_category": category,
            "target_intent": category if category in INTENT_CATEGORIES else "",
            "source": segment["source"],
            "dataset_source": segment["dataset_source"],
            "metadata": segment["metadata"],
            "original_text": assembled_text,
            "text": assembled_text,
            "segments": rendered_segments,
            "segment_count": 1,
            "segment_spans_original": spans,
            "segment_spans_merged": spans,
            "original_segments": [segment],
            "transformed_segments": rendered_segments,
            "mutation_applied": False,
            "mutation_type": None,
            "obfuscation_applied": False,
        }

    def _build_composite_row(self, recipe_name: str, row_id: int) -> dict[str, Any]:
        segments: list[dict[str, Any]] = []
        target_category = ""
        target_intent = ""
        filler_pools: list[str] = []

        if recipe_name == "persona_plus_nshot_plus_intent":
            persona = self._segment_from_row(self._pick_persona_row(context_row=self._sample_salad_row(intent_only=False)))
            nshot = self._segment_from_row(self._sample_pool_row("nshot"))
            intent_row = self._sample_salad_row(intent_only=True)
            intent = self._segment_from_row(intent_row)
            segments = [persona, nshot, intent]
            target_category = intent["label"]
            target_intent = intent["label"]
            filler_pools = ["salad_intent"]
        elif recipe_name == "persona_plus_format":
            persona = self._segment_from_row(self._pick_persona_row(context_row=self._sample_pool_row("cais_mmlu")))
            fmt = self._segment_from_row(self._sample_pool_row("format"))
            segments = [persona, fmt]
            target_category = fmt["label"] or persona["label"]
            filler_pools = ["format"]
        elif recipe_name == "jailbreak_plus_intent":
            jailbreak_pool = "jailbreak_clean" if len(self.pools.get("jailbreak_clean", [])) else "violation"
            jailbreak = self._segment_from_row(self._sample_pool_row(jailbreak_pool))
            intent_row = self._sample_salad_row(intent_only=True)
            intent = self._segment_from_row(intent_row)
            segments = [jailbreak, intent]
            target_category = intent["label"]
            target_intent = intent["label"]
            filler_pools = [jailbreak_pool, "salad_intent"]
        elif recipe_name == "benign_plus_format":
            benign = self._segment_from_row(self._sample_pool_row("benign"))
            fmt = self._segment_from_row(self._sample_pool_row("format"))
            segments = [benign, fmt]
            target_category = fmt["label"] or benign["label"]
            filler_pools = ["format"]
        elif recipe_name == "salad_plus_salad":
            first = self._segment_from_row(self._sample_salad_row(intent_only=False))
            second = self._segment_from_row(self._sample_salad_row(intent_only=False))
            segments = [first, second]
            target_category = first["label"]
            target_intent = first["label"] if first["label"] in INTENT_CATEGORIES else ""
            filler_pools = ["salad"]
        elif recipe_name == "jailbreak_plus_jailbreak":
            jailbreak_pool = "jailbreak_clean" if len(self.pools.get("jailbreak_clean", [])) else "violation"
            first = self._segment_from_row(self._sample_pool_row(jailbreak_pool))
            second = self._segment_from_row(self._sample_pool_row(jailbreak_pool))
            segments = [first, second]
            target_category = first["label"]
            filler_pools = [jailbreak_pool]
        elif recipe_name == "persona_plus_persona":
            first = self._segment_from_row(self._pick_persona_row())
            second = self._segment_from_row(self._pick_persona_row())
            segments = [first, second]
            target_category = first["label"]
            filler_pools = ["persona"]
        else:
            return self._build_standalone_row("standalone_benign", row_id)

        segments = self._append_fillers(
            segments,
            filler_pools=filler_pools,
            min_chars=self.config.min_assembled_chars,
            max_fill_segments=self.config.max_fill_segments,
        )
        segments = self._shuffle_composite_segments(segments)
        rendered_segments = self._render_segments(segments, row_seed=row_id)
        assembled_text, spans = self._join_segments(rendered_segments)
        category = target_category or segments[0]["label"]
        if not category:
            category = segments[0]["label"]
        return {
            "row_id": f"row_{row_id:06d}",
            "recipe_type": "composite",
            "assembly_type": recipe_name,
            "category": category,
            "target_category": target_category or category,
            "target_intent": target_intent,
            "source": segments[0]["source"],
            "dataset_source": segments[0]["dataset_source"],
            "metadata": segments[0]["metadata"],
            "original_text": assembled_text,
            "text": assembled_text,
            "segments": rendered_segments,
            "segment_count": len(rendered_segments),
            "segment_spans_original": spans,
            "segment_spans_merged": spans,
            "original_segments": segments,
            "transformed_segments": rendered_segments,
            "mutation_applied": False,
            "mutation_type": None,
            "obfuscation_applied": False,
        }

    def _build_transform_row(self, recipe_type: str, row_id: int) -> dict[str, Any]:
        base_choice = self._choice(["benign", "persona", "format", "nshot", "salad"])
        base_row = self._sample_salad_row(intent_only=False) if base_choice == "salad" else self._sample_pool_row(base_choice)
        segment = self._segment_from_row(base_row)
        rendered_segments = self._render_segments([segment], row_seed=row_id)
        assembled_text, spans = self._join_segments(rendered_segments)
        if recipe_type == "obfuscated":
            transformed_text, mutation_applied, transform_label, obfuscation_kind, encryption_method = self._compose_obfuscated_text(assembled_text)
        else:
            transformed_text, mutation_applied, transform_label = self._compose_text(assembled_text, recipe_type)
            obfuscation_kind = ""
            encryption_method = ""
        final_category = segment["label"]
        if recipe_type == "obfuscated":
            mutation_type = obfuscation_kind
            obfuscation_applied = True
        else:
            mutation_type = "mutation"
            obfuscation_applied = False
        return {
            "row_id": f"row_{row_id:06d}",
            "recipe_type": recipe_type,
            "assembly_type": f"{recipe_type}_standalone",
            "category": final_category,
            "target_category": final_category,
            "target_intent": final_category if final_category in INTENT_CATEGORIES else "",
            "source": segment["source"],
            "dataset_source": segment["dataset_source"],
            "metadata": segment["metadata"],
            "original_text": assembled_text,
            "text": transformed_text,
            "obfuscation_method": mutation_type if recipe_type == "obfuscated" else "",
            "encryption_method": encryption_method,
            "segments": rendered_segments,
            "segment_count": 1,
            "segment_spans_original": spans,
            "segment_spans_merged": spans,
            "original_segments": [segment],
            "transformed_segments": rendered_segments,
            "mutation_applied": True,
            "mutation_type": mutation_type,
            "obfuscation_applied": obfuscation_applied,
        }

    def _plan_standalone_row(self, recipe_name: str, row_id: int) -> dict[str, Any]:
        if recipe_name == "standalone_benign":
            base = self._sample_pool_row("benign")
        elif recipe_name == "standalone_persona":
            base = self._pick_persona_row()
        elif recipe_name == "standalone_format":
            base = self._sample_pool_row("format")
        elif recipe_name == "standalone_nshot":
            base = self._sample_pool_row("nshot")
        elif recipe_name == "standalone_violation":
            jailbreak_pool = "jailbreak_clean" if len(self.pools.get("jailbreak_clean", [])) else "violation"
            base = self._sample_pool_row(jailbreak_pool)
        else:
            base = self._sample_salad_row(intent_only=True)

        segment = self._segment_from_row(base)
        category = segment["label"]
        return {
            "row_id": f"row_{row_id:06d}",
            "row_seed": row_id,
            "recipe_type": "standalone",
            "assembly_type": recipe_name,
            "category": category,
            "target_category": category,
            "target_intent": category if category in INTENT_CATEGORIES else "",
            "source": segment["source"],
            "dataset_source": segment["dataset_source"],
            "metadata": segment["metadata"],
            "original_segments": [segment],
            "base_row": {
                "row_id": f"row_{row_id:06d}",
                "recipe_type": "standalone",
                "assembly_type": recipe_name,
                "category": category,
                "target_category": category,
                "target_intent": category if category in INTENT_CATEGORIES else "",
                "source": segment["source"],
                "dataset_source": segment["dataset_source"],
                "metadata": segment["metadata"],
                "obfuscation_method": "",
                "encryption_method": "",
            },
            "mutation_profile": None,
            "mutation_type": None,
            "obfuscation_mode": "",
            "obfuscation_kind": "",
            "encryption_method": "",
        }

    def _plan_composite_row(self, recipe_name: str, row_id: int) -> dict[str, Any]:
        segments: list[dict[str, Any]] = []
        target_category = ""
        target_intent = ""
        filler_pools: list[str] = []

        if recipe_name == "persona_plus_nshot_plus_intent":
            persona = self._segment_from_row(self._pick_persona_row(context_row=self._sample_salad_row(intent_only=False)))
            nshot = self._segment_from_row(self._sample_pool_row("nshot"))
            intent_row = self._sample_salad_row(intent_only=True)
            intent = self._segment_from_row(intent_row)
            segments = [persona, nshot, intent]
            target_category = intent["label"]
            target_intent = intent["label"]
            filler_pools = ["salad_intent"]
        elif recipe_name == "persona_plus_format":
            persona = self._segment_from_row(self._pick_persona_row(context_row=self._sample_pool_row("cais_mmlu")))
            fmt = self._segment_from_row(self._sample_pool_row("format"))
            segments = [persona, fmt]
            target_category = fmt["label"] or persona["label"]
            filler_pools = ["format"]
        elif recipe_name == "jailbreak_plus_intent":
            jailbreak_pool = "jailbreak_clean" if len(self.pools.get("jailbreak_clean", [])) else "violation"
            jailbreak = self._segment_from_row(self._sample_pool_row(jailbreak_pool))
            intent_row = self._sample_salad_row(intent_only=True)
            intent = self._segment_from_row(intent_row)
            segments = [jailbreak, intent]
            target_category = intent["label"]
            target_intent = intent["label"]
            filler_pools = [jailbreak_pool, "salad_intent"]
        elif recipe_name == "benign_plus_format":
            benign = self._segment_from_row(self._sample_pool_row("benign"))
            fmt = self._segment_from_row(self._sample_pool_row("format"))
            segments = [benign, fmt]
            target_category = fmt["label"] or benign["label"]
            filler_pools = ["format"]
        elif recipe_name == "salad_plus_salad":
            first = self._segment_from_row(self._sample_salad_row(intent_only=False))
            second = self._segment_from_row(self._sample_salad_row(intent_only=False))
            segments = [first, second]
            target_category = first["label"]
            target_intent = first["label"] if first["label"] in INTENT_CATEGORIES else ""
            filler_pools = ["salad"]
        elif recipe_name == "jailbreak_plus_jailbreak":
            jailbreak_pool = "jailbreak_clean" if len(self.pools.get("jailbreak_clean", [])) else "violation"
            first = self._segment_from_row(self._sample_pool_row(jailbreak_pool))
            second = self._segment_from_row(self._sample_pool_row(jailbreak_pool))
            segments = [first, second]
            target_category = first["label"]
            filler_pools = [jailbreak_pool]
        elif recipe_name == "persona_plus_persona":
            first = self._segment_from_row(self._pick_persona_row())
            second = self._segment_from_row(self._pick_persona_row())
            segments = [first, second]
            target_category = first["label"]
            filler_pools = ["persona"]
        else:
            return self._plan_standalone_row("standalone_benign", row_id)

        segments = self._append_fillers(
            segments,
            filler_pools=filler_pools,
            min_chars=self.config.min_assembled_chars,
            max_fill_segments=self.config.max_fill_segments,
        )
        segments = self._shuffle_composite_segments(segments)
        category = target_category or segments[0]["label"]
        if not category:
            category = segments[0]["label"]
        return {
            "row_id": f"row_{row_id:06d}",
            "row_seed": row_id,
            "recipe_type": "composite",
            "assembly_type": recipe_name,
            "category": category,
            "target_category": target_category or category,
            "target_intent": target_intent,
            "source": segments[0]["source"],
            "dataset_source": segments[0]["dataset_source"],
            "metadata": segments[0]["metadata"],
            "original_segments": segments,
            "base_row": {
                "row_id": f"row_{row_id:06d}",
                "recipe_type": "composite",
                "assembly_type": recipe_name,
                "category": category,
                "target_category": target_category or category,
                "target_intent": target_intent,
                "source": segments[0]["source"],
                "dataset_source": segments[0]["dataset_source"],
                "metadata": segments[0]["metadata"],
                "obfuscation_method": "",
                "encryption_method": "",
            },
            "mutation_profile": None,
            "mutation_type": None,
            "obfuscation_mode": "",
            "obfuscation_kind": "",
            "encryption_method": "",
        }

    def _plan_transform_row(self, recipe_type: str, row_id: int) -> dict[str, Any]:
        base_choice = self._choice(["benign", "persona", "format", "nshot", "salad"])
        base_row = self._sample_salad_row(intent_only=False) if base_choice == "salad" else self._sample_pool_row(base_choice)
        segment = self._segment_from_row(base_row)
        original_segments = [segment]
        category = segment["label"]
        mutation_profile = self.text_changer.mutator.random_profile() if self.text_changer is not None and recipe_type == "mutated" else None
        obfuscation_mode = ""
        obfuscation_kind = ""
        encryption_method = ""

        if recipe_type == "obfuscated":
            if self.rng.random() < 0.5:
                obfuscation_mode = "encryption"
                obfuscation_kind = "encryption"
                encryption_method = self.rng.choice(list(self.text_changer.encrypter.METHODS)) if self.text_changer is not None else ""
            else:
                obfuscation_mode = "unicode_obfuscation"
                obfuscation_kind = "unicode_obfuscation"
                mutation_profile = self._next_unicode_obfuscation_profile()

        return {
            "row_id": f"row_{row_id:06d}",
            "row_seed": row_id,
            "recipe_type": recipe_type,
            "assembly_type": f"{recipe_type}_standalone",
            "category": category,
            "target_category": category,
            "target_intent": category if category in INTENT_CATEGORIES else "",
            "source": segment["source"],
            "dataset_source": segment["dataset_source"],
            "metadata": segment["metadata"],
            "original_segments": original_segments,
            "base_row": {
                "row_id": f"row_{row_id:06d}",
                "recipe_type": recipe_type,
                "assembly_type": f"{recipe_type}_standalone",
                "category": category,
                "target_category": category,
                "target_intent": category if category in INTENT_CATEGORIES else "",
                "source": segment["source"],
                "dataset_source": segment["dataset_source"],
                "metadata": segment["metadata"],
                "obfuscation_method": obfuscation_kind if recipe_type == "obfuscated" else "",
                "encryption_method": encryption_method,
            },
            "mutation_profile": mutation_profile,
            "mutation_type": "mutation" if recipe_type == "mutated" else obfuscation_kind or "mutation",
            "obfuscation_mode": obfuscation_mode,
            "obfuscation_kind": obfuscation_kind,
            "encryption_method": encryption_method,
        }

    def _build_plan_one(self, index: int, target_rows: int) -> dict[str, Any]:
        recipe_type = self._plan_recipe_type(index, target_rows)
        if recipe_type == "standalone":
            return self._plan_standalone_row(self._standalone_recipe(), index)
        if recipe_type == "composite":
            return self._plan_composite_row(self._composite_recipe(), index)
        return self._plan_transform_row(recipe_type, index)

    def _build_parallel(
        self,
        target_rows: int,
        max_workers: int | None = None,
        chunk_size: int = 64,
    ) -> pd.DataFrame:
        safe_chunk_size = max(1, chunk_size)
        records: list[dict[str, Any]] = []
        if not max_workers:
            max_workers = mp.cpu_count()
            print(f"Using {max_workers} workers for parallel processing.")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=target_rows, desc="Building decision-tree rows") as progress:
                for start in range(0, target_rows, safe_chunk_size):
                    chunk: list[dict[str, Any]] = [
                        self._build_plan_one(i, target_rows)
                        for i in range(start, min(start + safe_chunk_size, target_rows))
                    ]
                    records.extend(executor.map(_materialize_decision_tree_plan, chunk))
                    progress.update(len(chunk))
        planned_df = pd.DataFrame(records)
        if not self.config.return_debug_columns:
            keep = ["text", "category", "source", "dataset_source", "metadata"]
            return planned_df[keep].copy()
        return planned_df

    def _build_one(self, index: int, target_rows: int) -> dict[str, Any]:
        recipe_type = self._plan_recipe_type(index, target_rows)
        if recipe_type == "standalone":
            return self._build_standalone_row(self._standalone_recipe(), index)
        if recipe_type == "composite":
            return self._build_composite_row(self._composite_recipe(), index)
        return self._build_transform_row(recipe_type, index)

    def build(
        self,
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        chunk_size: int = 64,
    ) -> pd.DataFrame:
        """
        Build a planned training dataframe using the recipe planner.
        """
        target_rows = self.config.target_rows or len(self.df)
        if parallel:
            return self._build_parallel(target_rows, max_workers=max_workers, chunk_size=chunk_size)
        iterator = tqdm(range(target_rows), total=target_rows, desc="Building decision-tree rows")

        records = [self._build_one(i, target_rows) for i in iterator]
        planned_df = pd.DataFrame(records)
        if not self.config.return_debug_columns:
            keep = ["text", "category", "source", "dataset_source", "metadata"]
            return planned_df[keep].copy()
        return planned_df

    def save(
        self,
        output_path: str | Path = "decision_tree_output.parquet",
        *,
        index: bool = False,
        parallel: bool = True,
        max_workers: int | None = None,
        chunk_size: int = 64,
    ) -> Path:
        """
        Build the planned dataframe and save it as a parquet file.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df = self.build(parallel=parallel, max_workers=max_workers, chunk_size=chunk_size)
        df.to_parquet(output, index=index)
        return output

    def build_and_save(
        self,
        output_path: str | Path = "decision_tree_output.parquet",
        *,
        index: bool = False,
        parallel: bool = True,
        max_workers: int | None = None,
        chunk_size: int = 64,
    ) -> pd.DataFrame:
        """
        Convenience helper for callers that want the dataframe in memory and
        persisted to parquet at the same time.
        """
        df = self.build(parallel=parallel, max_workers=max_workers, chunk_size=chunk_size)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=index)
        return df

    def preview(self, n: int = 5) -> pd.DataFrame:
        return self.df.head(n).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save the decision-tree parquet.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("decision_tree_output.parquet"),
        help="Path to write the generated parquet file.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1000,
        help="Number of rows to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for row selection.",
    )
    parser.add_argument(
        "--standalone-fraction",
        type=float,
        default=0.20,
        help="Fraction of total rows that should be standalone recipes.",
    )
    parser.add_argument(
        "--standalone-control-fraction",
        type=float,
        default=0.35,
        help="Within standalone rows, fraction reserved for benign/control examples.",
    )
    parser.add_argument(
        "--standalone-salad-fraction",
        type=float,
        default=0.25,
        help="Within standalone rows, fraction reserved for Salad examples.",
    )
    parser.add_argument(
        "--flat-only",
        action="store_true",
        help="Save only the flat five-column schema without span/debug columns.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the source summary and pool sizes without generating output.",
    )
    args = parser.parse_args()

    composer = DecisionTreeComposer(
        config=DecisionTreeConfig(
            seed=args.seed,
            target_rows=args.rows,
            standalone_fraction=args.standalone_fraction,
            standalone_control_fraction=args.standalone_control_fraction,
            standalone_salad_fraction=args.standalone_salad_fraction,
            return_debug_columns=not args.flat_only,
        )
    )

    summary = composer.summary()
    print("Source parquet summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print("\nPool sizes:")
    for name, size in composer.pool_sizes().items():
        print(f"- {name}: {size}")

    if args.summary_only:
        return

    output = composer.build_and_save(args.output)
    print(f"\nSaved {args.rows} rows to: {output}")


if __name__ == "__main__":
    main()
