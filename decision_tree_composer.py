from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse

import numpy as np
import pandas as pd
import random

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
    obfuscation_fraction: float = 0.10
    mutation_fraction: float = 0.10
    composite_fraction: float | None = None
    nshot_min_examples: int = 1
    nshot_max_examples: int = 4
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
        self.raw_df = full_df2.copy() if full_df2 is not None else load_full_df2(self.parquet_path)
        self.df = normalize_full_df2(self.raw_df)
        self.pools = split_source_pools(self.df)

    @staticmethod
    def _load_text_changer():
        try:
            from text_composer import TextChanger
        except Exception:
            return None
        return TextChanger()

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

    @staticmethod
    def _cell_text(value: Any) -> str:
        tokens = _flatten_for_match(value)
        return "\n\n".join(tokens) if len(tokens) > 1 else (tokens[0] if tokens else "")

    @staticmethod
    def _cell_first(value: Any) -> str:
        tokens = _flatten_for_match(value)
        return tokens[0] if tokens else ""

    def _segment_from_row(self, row: pd.Series, label: str | None = None) -> dict[str, Any]:
        return {
            "text": self._cell_text(row["text"]),
            "label": label or self._cell_first(row["category"]),
            "source": self._cell_first(row["source"]),
            "dataset_source": self._cell_first(row["dataset_source"]),
            "metadata": self._cell_first(row["metadata"]),
        }

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
            row = self._sample_pool_row(pool_name)
            working.append(self._segment_from_row(row))
        return working

    def _compose_text(self, text: str, recipe_type: str) -> tuple[str, bool, str | None]:
        if self.text_changer is None:
            return text, False, None

        operation_map = {
            "mutated": ("mutation", True),
            "obfuscated": ("encryption", True),
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

    def _standalone_recipe(self) -> str:
        weights = [
            ("standalone_benign", len(self.pools["benign"])),
            ("standalone_persona", len(self.pools["persona"])),
            ("standalone_format", len(self.pools["format"])),
            ("standalone_nshot", len(self.pools["nshot"])),
            ("standalone_violation", len(self.pools["violation"])),
            ("standalone_salad", len(self.pools["salad"])),
        ]
        choices = [name for name, size in weights if size > 0]
        return self._choice(choices)

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
            base = self._sample_pool_row("persona")
        elif recipe_name == "standalone_format":
            base = self._sample_pool_row("format")
        elif recipe_name == "standalone_nshot":
            base = self._sample_pool_row("nshot")
        elif recipe_name == "standalone_violation":
            jailbreak_pool = "jailbreak_clean" if len(self.pools.get("jailbreak_clean", [])) else "violation"
            base = self._sample_pool_row(jailbreak_pool)
        else:
            intent_pool = self._intent_rows()
            if intent_pool.empty:
                base = self._sample_pool_row("salad")
            else:
                base = intent_pool.sample(n=1, random_state=self.rng.randint(0, 2**32 - 1)).iloc[0]

        segment = self._segment_from_row(base)
        assembled_text, spans = self._join_segments([segment])
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
            "segments": [segment],
            "segment_count": 1,
            "segment_spans_original": spans,
            "segment_spans_merged": spans,
            "original_segments": [segment],
            "transformed_segments": [segment],
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
            persona = self._segment_from_row(self._sample_pool_row("persona"))
            nshot = self._segment_from_row(self._sample_pool_row("nshot"))
            intent_row = self._sample_pool_row("salad")
            intent = self._segment_from_row(intent_row)
            segments = [persona, nshot, intent]
            target_category = intent["label"]
            target_intent = intent["label"]
            filler_pools = ["salad"]
        elif recipe_name == "persona_plus_format":
            persona = self._segment_from_row(self._sample_pool_row("persona"))
            fmt = self._segment_from_row(self._sample_pool_row("format"))
            segments = [persona, fmt]
            target_category = fmt["label"] or persona["label"]
            filler_pools = ["format"]
        elif recipe_name == "jailbreak_plus_intent":
            jailbreak_pool = "jailbreak_clean" if len(self.pools.get("jailbreak_clean", [])) else "violation"
            jailbreak = self._segment_from_row(self._sample_pool_row(jailbreak_pool))
            intent_row = self._sample_pool_row("salad")
            intent = self._segment_from_row(intent_row)
            segments = [jailbreak, intent]
            target_category = intent["label"]
            target_intent = intent["label"]
            filler_pools = [jailbreak_pool, "salad"]
        elif recipe_name == "benign_plus_format":
            benign = self._segment_from_row(self._sample_pool_row("benign"))
            fmt = self._segment_from_row(self._sample_pool_row("format"))
            segments = [benign, fmt]
            target_category = fmt["label"] or benign["label"]
            filler_pools = ["format"]
        elif recipe_name == "salad_plus_salad":
            first = self._segment_from_row(self._sample_pool_row("salad"))
            second = self._segment_from_row(self._sample_pool_row("salad"))
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
            first = self._segment_from_row(self._sample_pool_row("persona"))
            second = self._segment_from_row(self._sample_pool_row("persona"))
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
        assembled_text, spans = self._join_segments(segments)
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
            "segments": segments,
            "segment_count": len(segments),
            "segment_spans_original": spans,
            "segment_spans_merged": spans,
            "original_segments": segments,
            "transformed_segments": segments,
            "mutation_applied": False,
            "mutation_type": None,
            "obfuscation_applied": False,
        }

    def _build_transform_row(self, recipe_type: str, row_id: int) -> dict[str, Any]:
        base_row = self._sample_pool_row(self._choice(["benign", "persona", "format", "nshot", "salad"]))
        segment = self._segment_from_row(base_row)
        assembled_text, spans = self._join_segments([segment])
        transformed_text, mutation_applied, transform_label = self._compose_text(assembled_text, recipe_type)
        final_category = segment["label"]
        if recipe_type == "obfuscated":
            mutation_type = transform_label or "encryption"
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
            "segments": [segment],
            "segment_count": 1,
            "segment_spans_original": spans,
            "segment_spans_merged": spans,
            "original_segments": [segment],
            "transformed_segments": [segment],
            "mutation_applied": True,
            "mutation_type": mutation_type,
            "obfuscation_applied": obfuscation_applied,
        }

    def _build_one(self, index: int, target_rows: int) -> dict[str, Any]:
        recipe_type = self._plan_recipe_type(index, target_rows)
        if recipe_type == "standalone":
            return self._build_standalone_row(self._standalone_recipe(), index)
        if recipe_type == "composite":
            return self._build_composite_row(self._composite_recipe(), index)
        return self._build_transform_row(recipe_type, index)

    def build(self) -> pd.DataFrame:
        """
        Build a planned training dataframe using the recipe planner.
        """
        target_rows = self.config.target_rows or len(self.df)
        records = [self._build_one(i, target_rows) for i in range(target_rows)]
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
    ) -> Path:
        """
        Build the planned dataframe and save it as a parquet file.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df = self.build()
        df.to_parquet(output, index=index)
        return output

    def build_and_save(
        self,
        output_path: str | Path = "decision_tree_output.parquet",
        *,
        index: bool = False,
    ) -> pd.DataFrame:
        """
        Convenience helper for callers that want the dataframe in memory and
        persisted to parquet at the same time.
        """
        df = self.build()
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

    output = composer.save(args.output)
    print(f"\nSaved {args.rows} rows to: {output}")


if __name__ == "__main__":
    main()
