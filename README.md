# Pragmatics / Autism — Precision-RSA project

Speaker production data and stimuli for the reversed director-task experiments examining over-specification in autistic vs. non-autistic speakers, framed through a Bayesian precision-weighting account.

Data and design figure sent by Dimitris on 2026-04-25 following our Thursday meeting.

## Repository layout

```
data/
  raw/              # Speaker production data, four CSV datasets
    accuracy_ASD.csv
    accuracy_NT.csv
    minimal_ASD.csv
    minimal_NT.csv
    _original_from_dimitris/
      Dimitris_precision_rsa_2026-04-25.zip   # untouched original
  stimuli/          # Trial design and listener model inputs
    world_states.json
    possible_expressions.json
docs/
  design_conj.png   # Design figure illustrating the conjunctive trial structure
analysis/           # (empty) for behavioral analysis scripts/notebooks
models/             # (empty) for the precision-weighted RSA implementation
```

## Datasets

There are four speaker datasets, organized by experiment × population:

| File | Experiment | Population | Participants | Trials/participant | Total rows |
|------|-----------|-----------|---:|---:|---:|
| `data/raw/minimal_ASD.csv`  | Minimal (terse instructions) | Autistic | 55 | 24 | 1,320 |
| `data/raw/minimal_NT.csv`   | Minimal | Non-autistic | 65 | 24 | 1,560 |
| `data/raw/accuracy_ASD.csv` | Accuracy (no length limit) | Autistic | 37 | 24 | 888 |
| `data/raw/accuracy_NT.csv`  | Accuracy | Non-autistic | 40 | 24 | 960 |

- *Minimal* refers to the experiment in which speakers were asked to type messages using as few words as possible.
- *Accuracy* refers to the experiment with no limit on how many words speakers could use.
- Each speaker contributes one row per trial; the 24 trials are balanced across the 2 (occlusion) × 2 (set_a_dist_feature) design (6 trials per cell).
- The autistic group includes both self-identified and clinically-diagnosed participants (recorded in the `autistic` column), supporting the planned sensitivity analysis.

### Column schema (all four CSVs)

| Column | Description |
|---|---|
| `autistic` | Speaker neurotype: `No`, `Yes (self-identified)`, or `Yes (clinically diagnosed)` |
| `prolific_pid` | Speaker ID (Prolific participant identifier) |
| `expression` | The message the speaker typed on this trial |
| `unique_property` | The single property that distinguishes set A from set B (e.g., `pentagon`, `big`) |
| `occlusion` | Whether the third object in set A was occluded from the listener (`yes` / `no`) |
| `set_a_dist_feature` | The dimension on which the set-A distractor differs from the target: `type` (shape) or `non_type` (color, size, etc.) |
| `set_a` | Properties of the first two (identical) objects in set A: `"count, size, color, shape"` |
| `set_b` | Properties of the third object — the unique element in set A: `"count, size, color, shape"` |

The condition is jointly defined by `occlusion` × `set_a_dist_feature`.

## Stimuli (`data/stimuli/`)

Both files describe the **24 trial world-states** keyed by an integer `id`.

### `world_states.json`

For each trial, a list of three objects, with the contrastive distractor sets:

```json
{
  "id": 1,
  "target":          ["two big blue square", "one small black star"],
  "set_a_type":      ["two big blue triangle", "one small black star"],
  "set_a_non_type":  ["two big green square",  "one small black star"]
}
```

- `target` — the configuration set A actually contains.
- `set_a_type` — the alternative set A would have been if the distractor differed from the target on **shape** (the *type* condition).
- `set_a_non_type` — the alternative if the distractor differed on a **non-shape** feature (color, size; the *non_type* condition).

### `possible_expressions.json`

The candidate utterances over which the precision-RSA listener should be defined for each trial. Currently 8 expressions per trial — incremental prefixes of the full target description:

```json
{
  "id": 1,
  "target": ["two big blue square", "one small black star"],
  "expressions": [
    "two", "two big", "two big blue", "two big blue square",
    "two big blue square and one",
    "two big blue square and one small",
    "two big blue square and one small black",
    "two big blue square and one small black star"
  ]
}
```

## Design figure

`docs/design_conj.png` — diagram of the conjunctive trial structure (target plus contrastive set A under the type / non-type × occlusion manipulations).

## Provenance

Source: zip file from Dimitris, 2026-04-25, attached to email following our meeting. The original archive is preserved unmodified at `data/raw/_original_from_dimitris/Dimitris_precision_rsa_2026-04-25.zip`; everything else in `data/` is a copy with simplified filenames.
