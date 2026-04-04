# Decision Tree Implementation Plan

This document describes the first concrete implementation layer for the prompt decision tree.

The goal is to take `full_df2` from the notebook and turn it into a new training dataframe by:

- selecting a recipe
- sampling the right source rows
- assembling one final prompt string
- preserving provenance in the five notebook columns

## Input Contract

The implementation should accept `full_df2` as the main source table.

`full_df2` is the unified notebook dataframe that already contains rows from:

- jailbreak
- persona
- persona synthetic variants
- format synthetic variants
- n-shot
- benign
- Salad-Data

### Expected columns

The core columns should be:

- `text`
- `category`
- `source`
- `dataset_source`
- `metadata`

The decision tree should treat these as the canonical export columns.

## Recommended Top-Level API

Use one coordinator entry point rather than passing many separate dataframes around.

```python
def build_decision_tree_dataset(
    full_df2,
    *,
    seed: int = 42,
    target_rows: int | None = None,
    standalone_fraction: float = 0.20,
    obfuscation_fraction: float = 0.10,
    mutation_fraction: float = 0.10,
    composite_fraction: float | None = None,
    nshot_min_examples: int = 1,
    nshot_max_examples: int = 4,
    preserve_original_rows: bool = True,
    allow_faker_for_obfuscation: bool = True,
    export_arrays_for_intermediate_steps: bool = False,
    return_debug_columns: bool = True,
):
    ...
```

### Tree-to-text changer contract

When the coordinator needs a specific transform branch, it should call:

```python
TextChanger.compose(
    text,
    operation="mutation" | "encryption" | "formatting",
    mutation_profile=...,
    encryption_method=...,
    code_method=...,
)
```

Use `operation` when you want an exact branch.

- `mutation`
  - deterministic mutation branch
  - use when the tree assigns a `mutated` recipe
- `encryption`
  - deterministic obfuscation branch
  - use when the tree assigns an `obfuscated` recipe
- `formatting`
  - deterministic wrapper branch
  - use when the tree wants an exact code/comment formatter

If `operation` is omitted, `TextChanger.compose()` may continue to sample random operations.

## What the Function Should Do

The function should produce a final dataframe ready for export, while keeping the intermediate recipe logic deterministic and debuggable.

### 1. Normalize the source table

- Copy `full_df2`
- Standardize string fields
- Normalize `category`, `source`, `dataset_source`, and `metadata`
- Drop or ignore rows missing required fields
- Ensure `text` is always a string or list of strings in the expected source-specific format

### 2. Split the source table into source pools

Build source pools from `full_df2` using existing notebook columns:

- `salad_pool`
- `jailbreak_pool`
- `persona_pool`
- `format_pool`
- `nshot_pool`
- `benign_pool`

The source pools are just filtered views, not separate truth sources.

### 3. Choose a recipe for each output row

The recipe selector should decide one of the following:

- `standalone`
- `composite`
- `obfuscated`
- `mutated`

The selection should be driven by quota, not pure randomness.

### 4. Sample source rows for the recipe

Once the recipe is chosen, the coordinator should pull from the relevant pool(s).

- `standalone`
  - sample one row from a single source family
  - keep the row clean and directly exportable
- `composite`
  - sample 2 to 3 compatible source spans
  - examples: persona + n-shot, persona + format, jailbreak + intent
- `obfuscated`
  - sample a base row
  - apply faker or an obfuscation transform
- `mutated`
  - sample a base row
  - apply mutation after structure is chosen

### 5. Assemble the prompt

The assembler should:

- concatenate the chosen segments
- preserve segment order
- compute local spans
- compute merged spans
- store provenance for each segment
- call `TextChanger.compose()` only when the recipe requires mutation, encryption, or formatting
- pass `operation=` when the coordinator wants an exact branch instead of randomized `selected_ops`

### 6. Emit the final notebook row

The exported record should still use the notebook’s flat schema:

- `text`
- `category`
- `source`
- `dataset_source`
- `metadata`

If debug mode is enabled, the implementation can also keep:

- `recipe_type`
- `assembly_type`
- `segment_count`
- `segment_spans_original`
- `segment_spans_merged`
- `original_text`
- `original_segments`
- `transformed_segments`

## Recipe Selection Rules

### Standalone reserve

Reserve a portion of rows for single-source examples.

This should cover:

- Salad intent rows
- jailbreak rows
- persona rows
- benign rows
- n-shot rows

### Obfuscation reserve

Reserve a portion of rows for obfuscation examples.

This can use either:

- real rows from `full_df2`
- faker-generated pseudo text when you want to preserve real examples

### Mutation reserve

Reserve a portion of rows for mutation examples.

This is useful for:

- OCR-like corruption
- unicode variation
- keyboard-neighbor noise
- token deletion or swap style mutations

### Composite remainder

Use the remaining quota for composite prompts.

These rows should be the most realistic:

- persona + n-shot + intent
- persona + format
- jailbreak + intent
- benign + format

## N-shot Handling

`NSHOT` should be handled as an internal multi-fragment recipe only.

The implementation should:

- sample 1 to 4 example rows when building a n-shot prompt
- keep the intermediate fragments in a list
- flatten them into one exported `text`

This means `NSHOT` is still 1-to-1 at final export time.

## Suggested Helper Functions

The coordinator should call a small set of helpers.

```python
def split_source_pools(full_df2) -> dict[str, object]: ...
def choose_recipe(rng, quotas, row_idx) -> str: ...
def sample_segments_for_recipe(rng, pools, recipe) -> list[dict]: ...
def assemble_segments(segments, joiners) -> dict[str, object]: ...
def flatten_record(assembled_row) -> dict[str, object]: ...
```

## Recommended Class Design

If this grows beyond a few helpers, make it a small class.

```python
class DecisionTreeComposer:
    def __init__(self, full_df2, *, seed=42, config=None):
        ...

    def build(self) -> pd.DataFrame:
        ...
```

### Class responsibilities

- `__init__`
  - store the source dataframe
  - configure quota settings
  - initialize deterministic randomness
- `build`
  - iterate over rows or target samples
  - choose recipes
  - assemble prompts
  - return the final dataframe

## Minimal First Implementation

If you want the smallest useful version, implement only this:

1. split `full_df2` into source pools
2. generate standalone rows
3. generate a small composite subset
4. generate a small obfuscation subset
5. return one final flat dataframe

That gets the tree running before you add more recipe branches.
