# Jev / GPT nano SBS relation comparison

A local experiment for FERODEV-8283. It compares Jev and GPT nano on **400
directed rows per shop** (200 forward and 200 reverse) from furniture.co.uk and
themeatboys.nl. The default, historical arm gives both providers the same frozen
product evidence and asks the same four questions for every pair:

1. identity: `duplicate`, `variant`, `redundant`, `distinct`, or `uncertain`;
2. co-purchase: `yes`, `no`, or `uncertain`;
3. alternative: `yes`, `no`, or `uncertain`;
4. incompatibility: `yes`, `no`, or `uncertain`.

That arm asks all four questions in one combined request so provider results remain
directly comparable over all 800 pairs. A separate `--jev-arm decomposed` experiment
asks Jev four independent evidence questions applicable to each pair—identity plus
three streams for that shop's objective—and reuses the completed combined-arm GPT
baseline without making GPT calls. Deterministic, versioned resolvers derive SBS
basket labels afterward. Raw heads or streams remain the source of truth in both
arms.

Neither arm is a production pipeline integration. Nothing imports or changes
`embedding-service`; no retrieval, summarization, training, embedding, or schema
change is involved. Historical FERODEV-8257 runs retain their original three-label
report format.

## Setup

Python **3.11+** and `uv`:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r pyproject.toml --extra test
.venv/bin/python -m pytest
```

Copy `.env.sample` to `.env` and populate the provider keys needed for paid runs.
The script loads `.env` only when `--execute` is present. Values already exported in
the process environment take precedence and credentials are never printed.

There is no build/package or configured linter. `git diff --check` checks patch
whitespace. Tests use synthetic catalogs and mocked HTTP; they never call providers.

## Inputs and offline preflight

The transferred files remain local under `data/<shop>/`. Required per shop:

- `stage1/products_processed.json`;
- `stage2/subset_products.json`, `candidates.json`, and `classification_context.json`;
- `stage2/forward_judgments.json` and `reverse_judgments.json` (JSONL despite the
  extensions);
- the saved recommendation, examples, and `stage2_classify_*` prompt files hashed
  by that classification context.

The old recommendation objective and prompts are retained as source provenance and
to validate the frozen population. They are not the v2 provider decision policy.
The universal relation contract is snapshotted from `compare.py` into the manifest.

Run from this repository root:

```sh
.venv/bin/python compare.py
```

This validates source hashes and product references, renders both provider
payloads, and writes `results/relation-manifest.json`. No credentials or network
requests are needed. Repeating preparation requires byte-identical output; a
different workload or contract needs a new `--manifest` path, never an overwrite.

The manifest freezes:

- all 800 pair identities, directions, product evidence, retrieval ranks/sources,
  and source digests;
- GPT `gpt-5-nano-2025-08-07` with medium reasoning and strict structured output;
- Jev `jev-1.13.0` with four Choice questions per candidate;
- the exact GPT policy and the Jev identity/relation criteria and instructions;
- the resolver version and all accepted labels.

Requests contain at most 20 candidates. A full Jev batch therefore has 80 Choice
questions. Concurrency defaults to 8 (`--concurrency`). The supplied snapshots
produce **62 batches per provider**; reverse batches are often smaller than 20.
Preflight prints the actual counts.

### Worktrees

`data/` and `results/` are ignored and therefore do not appear automatically in a
new Git worktree. Point the worktree at the original checkout's frozen catalog:

```sh
.venv/bin/python compare.py \
  --data-dir /absolute/path/to/jev-sbs-experiment/data
```

Outputs still go to the current worktree's ignored `results/` directory unless an
explicit manifest, baseline root, or output directory is supplied. Do not copy or
edit an existing experiment's snapshots merely to make them visible in a worktree.
In a worktree, add the same `--data-dir` flag to every non-report-only command below.
Paid commands first look for `.env` in the current worktree. If it is absent or does
not define a key, they also load `.env` beside the supplied `data/` directory, so the
original checkout's ignored credential file can be reused without copying it.

## Fresh GPT baseline — Boris operates paid completion

The FERODEV-8257 GPT labels are not valid for the revised objectives. Do **not**
import `results/run` or reuse its baseline. First prepare a fresh v2 baseline
offline. Once completed, this exact baseline is shared by the combined and
decomposed Jev arms:

```sh
.venv/bin/python compare.py --prepare-baseline
```

The initial state reports 0 retained judgments and 800 missing judgments. Baseline
data lives under `results/baselines/<gpt-input-fingerprint>/`:

- `spec.json`: exact GPT requests and ordered pair identities, published once;
- `repairs/*.jsonl`: a new durable log for each paid completion invocation, with
  every HTTP attempt flushed separately;
- `baseline.json`: created only when all 800 judgments exist; immutable thereafter.

The completed v2 baseline also snapshots its exact input specification so later
report-only processing does not rerender prompts from live code. The fingerprint
covers the GPT system/user prompts, strict schema, product evidence, candidate
ordering, pair direction, model, and request settings. Jev-only criteria and
pricing are excluded. Only the first valid answer for each pair is retained.

After checking current provider limits and intended spend, define `OPENAI_API_KEY`
in `.env` (or the process environment) and explicitly complete the baseline:

```sh
.venv/bin/python compare.py --prepare-baseline --execute
```

This can make all 62 GPT requests for a fresh baseline. Only incomplete batches are
sent on later invocations. Partial valid candidate judgments are retained; retries
send the original full request and share a maximum of three HTTP attempts per batch
per invocation. A nonzero exit preserves progress. Completed baseline preparation
makes zero requests even if repeated without credentials.

Baseline preparation uses a file lock on macOS/Linux. Immutable specs, response
history, and completed baselines cannot be replaced. An interrupted request that
reached the provider before its checkpoint may be billed again.

`--import-run` remains available only for a run containing the **exact matching v2
GPT requests**. A legacy or differently prompted run is rejected.

## Combined Jev execution — Boris operates the paid run

Plain `--execute` selects the combined arm, never runs GPT, and requires a complete
matching v2 baseline. It only calls Jev. Define `TYPESAFE_API_KEY` in `.env` or the
process environment; `OPENAI_API_KEY` is not needed:

```sh
.venv/bin/python compare.py --execute
```

Every invocation creates a fresh
`results/experiments/<UTC-timestamp>-<unique-id>/` directory and prints its path.
`--output-dir` may select another new directory, but it must not already exist.
There is no automatic Jev resume: every new execution resends the entire workload
and can incur new charges.

Keep credentials out of source, shell history, chat, and logs. `.env` is ignored by
Git and `.env.sample` lists only the variable names. Before execution, check provider
limits, current prices, and intended spend. There is no hard dollar cap.

Each request has a 120-second I/O timeout. Transport errors, 408, 429, and 5xx
responses receive at most two retries. `Retry-After` is honored up to 60 seconds;
longer waits stop the batch. Authentication/validation failures are not retried.
Retries can incur additional charges.

GPT may return valid judgments for only part of a batch; those judgments survive
while omitted candidates remain operational failures. Jev must answer every Choice
question in a batch. Unknown IDs, missing heads, invalid labels, and malformed
responses fail validation rather than becoming `uncertain` or `skip`.

Jev probabilities are preserved exactly. Three-choice relation distributions may
sum within 0.015 of one; the five-choice identity distribution may sum within 0.025
to accommodate two-decimal rounding. A selected choice within 0.01 of the reported
maximum is retained with an explicit warning; larger mismatches fail validation.
Decomposed streams use the same versioned rule: the sum tolerance is the greater of
0.015 and 0.005 per available choice, while selected-choice tolerance remains 0.01.
Reports flag discrepancies and never renormalize provider values.

## Relation contract and basket projection

Identity and each relation are intentionally separate. Providers must answer all
three relation heads even when they classify a pair as a duplicate or variant. This
keeps the raw comparison complete and lets the resolver apply precedence offline.

The resolver is conservative:

| Identity / relations | Basket result |
|---|---|
| `duplicate`, `variant`, or directional `redundant` | `skip` |
| co-purchase `yes`, alternative `no`, incompatible `no` | `positive` |
| co-purchase `no`, alternative `yes`, incompatible `no` | `hard_negative` |
| co-purchase `no`, alternative `no`, incompatible `yes` | `hard_negative` |
| all relation heads `no` | `skip` |
| unresolved uncertainty without a decisive matrix row | `skip` with uncertain status |
| more than one relation head `yes` | `conflict` for review |

`uncertain` identity continues to the relation matrix but remains visible in the
resolution record. No probability threshold or maximum-probability conflict rule is
applied. Raw identity, relation choices, reasons, probabilities, and confidence are
preserved even when identity filters the derived basket label.

Changing the GPT prompt/schema or pair evidence selects a fresh baseline. A
Jev-only criteria experiment may reuse the baseline, but creates a new manifest and
experiment snapshot. Agreement with GPT is comparison evidence, not ground truth.

The post-run Jev criteria require affirmative listing evidence for `yes` or `no`;
missing evidence is `uncertain`. Co-purchase needs a concrete together-use reason,
alternative needs the same immediate purchasing role, and incompatibility needs an
explicit mismatch. Shared category, collection, brand, cuisine, colour, or plausible
joint purchase is insufficient by itself.

The original paid run and its manifest remain immutable. Prepare any later
Jev-criteria experiment under a new manifest path while reusing the matching GPT
baseline:

```sh
.venv/bin/python compare.py \
  --manifest results/relation-manifest-evidence-v2.json
```

Use that same `--manifest` argument, plus the worktree's `--data-dir` argument when
needed, for the later developer-owned `--execute` command. Do not replace the
original `results/relation-manifest.json`.

## Decomposed Jev arm — operator handoff

The decomposed arm keeps the same 800 directed pairs and byte-equivalent GPT input
fingerprint. Jev receives four independent streams for every pair; requests are not
conditionally skipped, so raw evidence remains comparable before the offline
identity gate is applied:

| Objective | Streams |
|---|---|
| both | identity |
| complements | together-use, substitute, incompatibility |
| style compatibility | construction/finish, design language, placement/context |

Each stream preserves the selected choice, complete probability distribution,
confidence, and rounding warnings. Identity must be accepted as `distinct` before
an objective signal can produce an actionable basket label. The complements and
style-compatibility resolvers are separate. Accepted positive and negative evidence
in one objective becomes an explicit conflict rather than silently choosing a side.

The canonical ignored adjudication artifact is
`results/adjudication/jev-sbs-adjudication-v1/`. It freezes 48 development pairs and
96 sealed holdout pairs, balanced by shop and direction. An unordered product-pair
group cannot cross splits. Reviewer-facing evidence contains no
provider outputs; holdout rows must not be compared with Jev output until the
probability policy is immutable.

### 1. Offline preparation

Use a new decomposed manifest path and validate the canonical adjudication artifact:

```sh
.venv/bin/python compare.py --jev-arm decomposed \
  --manifest results/decomposed-manifest-v1.json \
  --adjudication-dir results/adjudication/jev-sbs-adjudication-v1
```

This is offline. For the frozen catalogs it prints 800 pairs and 248 Jev requests.
It does not alter the GPT baseline. In a worktree, also pass the original checkout's
absolute `--data-dir` as described above.

### 2. Complete and publish the human review

Copy `labels-template.jsonl` to a separate working file; never edit the template.
Complete every identity, objective mechanism, final label, ambiguity flag, reviewer
timestamp, and notes field from the frozen provider-blind evidence. Boris is the
approved initial reviewer. Publish the completed file as append-only revision 1:

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import compare

artifact = Path("results/adjudication/jev-sbs-adjudication-v1")
completed = compare.jsonl_records(Path("/absolute/path/to/completed-labels-v1.jsonl"))
print(json.dumps(compare.publish_adjudication_labels(artifact, completed, 1), indent=2))
PY
```

Publication validates all 144 labels, mechanisms, timestamps, split identities, and
the append-only digest chain. A changed revision must use the next revision number;
existing revision files cannot be replaced.

### 3. Developer-owned paid Jev run

After checking current TypeSafe limits, pricing, and intended spend, Boris runs:

```sh
.venv/bin/python compare.py --jev-arm decomposed \
  --manifest results/decomposed-manifest-v1.json \
  --adjudication-dir results/adjudication/jev-sbs-adjudication-v1 \
  --execute
```

This is the only paid command in this handoff. It requires `TYPESAFE_API_KEY`, makes
no GPT calls, reuses the completed matching v2 baseline, and creates a fresh
experiment directory. The frozen workload currently produces 248 Jev requests
before retries. Every pair receives all four applicable streams; identity covers all
800 pairs, while each objective-specific stream covers that objective's 400 pairs.
The experiment does not measure the savings of a production identity cascade. Never
rerun merely to repair derived reports—use report-only processing.

The completed run immediately writes raw stream reports with no selected threshold.
Missing or malformed answers remain failures, not semantic skips or zero usage.

### 4. Generate development curves while the holdout stays sealed

Replace `YOUR-RUN-ID` and the revision number as needed:

```sh
.venv/bin/python compare.py --report-only \
  --output-dir results/experiments/YOUR-RUN-ID \
  --adjudication-dir results/adjudication/jev-sbs-adjudication-v1 \
  --review-revision 1
```

This creates immutable `probability-curves.json` from the 48 development pairs only.
It does not choose thresholds, copy the full review into the run, or reveal holdout
labels in CSV reports. Inspect precision and coverage for the global identity rule
and each objective stream. Do not open or join the holdout labels with provider
output during policy selection.

### 5. Boris selects and freezes the probability policy

Create a JSON file with this exact shape. Every placeholder must be replaced with a
reviewed numeric value from 0 to 1; the script never supplies or optimizes one:

```json
{
  "schema_version": "decomposed-probability-policy-selection-v1",
  "chosen_by": "Boris",
  "chosen_at": "YYYY-MM-DDTHH:MM:SS+00:00",
  "rules": {
    "identity": {"min_probability": "<reviewed>", "min_margin": "<reviewed>"},
    "objectives": {
      "complements": {
        "together_use": {"min_probability": "<reviewed>", "min_margin": "<reviewed>"},
        "substitute": {"min_probability": "<reviewed>", "min_margin": "<reviewed>"},
        "incompatibility": {"min_probability": "<reviewed>", "min_margin": "<reviewed>"}
      },
      "style_compatibility": {
        "construction_finish": {"min_probability": "<reviewed>", "min_margin": "<reviewed>"},
        "design_language": {"min_probability": "<reviewed>", "min_margin": "<reviewed>"},
        "placement_context": {"min_probability": "<reviewed>", "min_margin": "<reviewed>"}
      }
    }
  }
}
```

The quoted placeholders deliberately make the template invalid for publication;
replace them with JSON numbers. Then freeze the policy and generate human-grounded
development and holdout reports:

```sh
.venv/bin/python compare.py --report-only \
  --output-dir results/experiments/YOUR-RUN-ID \
  --adjudication-dir results/adjudication/jev-sbs-adjudication-v1 \
  --review-revision 1 \
  --policy-selection /absolute/path/to/boris-policy-selection.json
```

The command binds the immutable policy to the exact manifest, experiment, request
log, review revision, and already-reviewed curve bytes before copying the full review
snapshot and evaluating the holdout. A different policy cannot replace it. After
publication, deterministic reconstruction needs only the saved run directory:

```sh
.venv/bin/python compare.py --report-only \
  --output-dir results/experiments/YOUR-RUN-ID
```

## Results

All outputs are ignored by Git. Each v2 experiment directory contains:

| File | Contents |
|---|---|
| `manifest.json` | Frozen 800-pair workload and universal relation contract |
| `baseline.json` | Completed GPT raw heads, exact input specification, provenance, and historical accounting |
| `experiment.json` | Exact Jev criteria/instructions, resolver version, dated pricing, digests, and intended request hashes |
| `requests.jsonl` | This experiment's Jev requests/responses, attempt usage/latency/status, retries, model IDs, and group wall times |
| `gpt_classifications.jsonl`, `jev_classifications.jsonl` | One record per pair/provider with status, raw heads, and deterministic resolution |
| `paired_results.csv` | All 800 rows with product/retrieval evidence, every provider head, GPT reasons, Jev probabilities/confidence, derived basket labels, and agreement flags |
| `disagreements.csv` | Successful pairs differing on any raw head or basket result |
| `summary.json` | Per-head/basket matrices, distributions, filtered/conflict counts, failures, and separated historical GPT versus new Jev accounting |

A decomposed experiment uses the same immutable manifest, baseline, experiment, and
request log, plus:

| File | Contents |
|---|---|
| `adjudication.json` | Frozen base adjudication manifest used when execution started |
| `jev_stream_classifications.jsonl` | One record per pair with every raw objective stream and status |
| `probability-curves.json` | Immutable development-only threshold/margin curves; never an automatically selected policy |
| `probability-policy.json` | Boris-selected immutable rules bound to the run, review revision, and curve bytes |
| `review-labels.jsonl`, `review-evidence.jsonl`, `review-labels.manifest.json` | Full reviewed snapshot copied only after policy publication |
| `paired_results.csv` | Raw streams and, once selected, per-stream policy decisions and objective resolution |
| `summary.json` | Stream counts, operational accounting, policy status, human metrics/slices, conflicts, calibration, and secondary GPT agreement |

Before policy selection, decomposed `paired_results.csv` can expose development
labels but not holdout labels. After selection, human metrics report precision,
hard-negative precision, coverage, abstention, confusion, calibration, conflicts,
objective/shop/direction slices, and pair-count-pro-rata cost per accepted correct
label. Group sections retain latency, retries, usage, and total run cost.

The CSV writer prefixes formula-like catalog text with an apostrophe. JSON retains
the original text. Raw usage belongs to requests and is not apportioned across pairs;
the decomposed human metric explicitly labels its pair-count-pro-rata cost estimate.

Regenerate reports without catalog inputs, credentials, or network access:

```sh
.venv/bin/python compare.py --report-only \
  --output-dir results/experiments/YOUR-RUN-ID
```

Reports use only that directory's immutable manifest, GPT baseline, experiment,
raw request log, and any published review/policy snapshots. They verify snapshot
digests and Jev request hashes. Changing live provider prompts, criteria, pricing,
or the curve grid cannot rewrite snapshotted inputs or a published policy. Basket
projection dispatches to the saved resolver version; an unavailable version stops
reporting rather than reinterpreting historical judgments. Derived reports are
replaced deterministically by the report extraction, policy, and resolver versions;
raw snapshots and `requests.jsonl` are not.

For the original FERODEV-8257 combined run, use:

```sh
.venv/bin/python compare.py --report-only --output-dir results/run
```

Its v1 three-label reports remain unchanged. Omitting `--output-dir` in report-only
mode still selects this legacy directory.

Interrupted v2 logs retain completed batches. Failed or missing answers remain
explicit errors and never enter agreement denominators. A malformed final partial
line is flagged and omitted; malformed interior lines or conflicting identities
stop reporting. Saved HTTP-200 responses rejected by an earlier parser may be
re-extracted offline without editing the raw log or making new calls.

## Reading the numbers

- **Agreement is not correctness.** Identity, each relation head, and the basket
  projection have separate matrices. Denominators include only pairs where both
  providers returned valid complete judgments.
- **Failures are not semantic labels.** `uncertain`, `skip`, and `conflict` are valid
  outcomes; provider errors and missing answers are separately counted and excluded.
- **New versus historical:** `groups` and `new_inference` cover only fresh Jev
  inference. GPT baseline timing, usage, and estimated cost remain under
  `gpt_baseline.accounting`; they are not charged to every Jev experiment.
- **Conflicts and filtering:** `resolution_review` exposes raw multi-head conflicts,
  identity-filtered pairs, and uncertain identities for semantic review.
- **Wall time** covers a complete provider/shop group including retries and
  persistence. Incomplete groups have null wall time and throughput.
- **Request latency** is per HTTP attempt and excludes retry sleeps. Batch requests,
  HTTP attempts, and retries are separate counts.
- **Usage and cost:** every recorded attempt contributes. Missing provider usage
  leaves totals null rather than zero. Costs are dated estimates, not bills; verify
  them in provider consoles.
- **Decomposed human metrics:** reviewed labels, not GPT, define correctness.
  Development curves describe candidate thresholds; only the frozen policy is
  evaluated on the holdout. Abstentions remain in coverage denominators, conflicts
  stay explicit, and incomplete reviewed inference makes the metric status
  incomplete rather than fabricating a score.
- **Decomposed cost per accepted correct label:** the selected review split receives
  a pair-count-pro-rata allocation of whole-run Jev cost. It is an experiment-level
  comparison aid, not provider billing or a production serving estimate.

Pricing references: [OpenAI](https://platform.openai.com/docs/pricing) and
[TypeSafe](https://docs.typesafe.ai/models).

## Interpretation limits

Sampling uses stable-hash anchor ordering and retained candidate order, not uniform
sampling of the full catalogs. The **saved reverse population was originally chosen
from historical GPT non-skip forward judgments**, so it is not an independent
sample under the revised objectives. Forward/reverse direction is part of pair
identity; the same product IDs can appear in both directions.

Combined requests maximize provider comparability but do not measure the request or
token savings of a future two-pass identity gate. The decomposed arm also sends all
four applicable streams for every pair: it repeats product evidence, increases
request count, and does not implement a production cascade. Batch context and
different API interfaces can affect answers. The 48-pair development set is small;
its curves are selection evidence, not proof of calibrated probabilities. The
96-pair holdout is used once against the frozen policy and must not be recycled into
tuning.

This experiment is a semantic prompt/extraction diagnostic, not capacity,
recommendation-quality, embedding-quality, or production training validation. It
does not change production retrieval, schemas, prompts, dependencies, or deployment
code. Paid-run results still require human review. GPT agreement remains secondary
comparison evidence and never substitutes for reviewed correctness.
