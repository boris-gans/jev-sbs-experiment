# Jev / GPT nano SBS relation comparison

A local experiment for FERODEV-8283. It compares Jev and GPT nano on **400
directed rows per shop** (200 forward and 200 reverse) from furniture.co.uk and
themeatboys.nl. Both providers receive the same frozen product evidence and answer
the same four questions for every pair:

1. identity: `duplicate`, `variant`, `redundant`, `distinct`, or `uncertain`;
2. co-purchase: `yes`, `no`, or `uncertain`;
3. alternative: `yes`, `no`, or `uncertain`;
4. incompatibility: `yes`, `no`, or `uncertain`.

The experiment asks all four questions in one combined request so provider results
remain directly comparable over all 800 pairs. A deterministic resolver derives an
SBS basket label afterward. The raw heads remain the source of truth.

This is not a production pipeline integration. Nothing imports or changes
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
offline:

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

## Jev execution — Boris operates the paid run

Plain `--execute` never runs GPT. It requires a complete matching v2 baseline and
only calls Jev. Define `TYPESAFE_API_KEY` in `.env` or the process environment;
`OPENAI_API_KEY` is not needed:

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
to accommodate two-decimal rounding. Reports flag discrepancies and never
renormalize provider values.

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

The CSV writer prefixes formula-like catalog text with an apostrophe. JSON retains
the original text. Usage belongs to requests and is not apportioned across pairs.

Regenerate reports without catalog inputs, credentials, or network access:

```sh
.venv/bin/python compare.py --report-only \
  --output-dir results/experiments/YOUR-RUN-ID
```

Reports use only that directory's immutable manifest, GPT baseline, experiment,
and raw request log. They verify snapshot digests and Jev request hashes. Changing
live provider prompts, criteria, or pricing cannot rewrite the snapshotted inputs.
Basket projection dispatches to the saved resolver version; an unavailable version
stops reporting rather than reinterpreting historical judgments. Derived reports
are replaced deterministically by the report extraction and resolver versions; raw
snapshots and `requests.jsonl` are not.

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

Pricing references: [OpenAI](https://platform.openai.com/docs/pricing) and
[TypeSafe](https://docs.typesafe.ai/models).

## Interpretation limits

Sampling uses stable-hash anchor ordering and retained candidate order, not uniform
sampling of the full catalogs. The **saved reverse population was originally chosen
from historical GPT non-skip forward judgments**, so it is not an independent
sample under the revised objectives. Forward/reverse direction is part of pair
identity; the same product IDs can appear in both directions.

Combined requests maximize provider comparability but do not measure the request or
token savings of a future two-pass identity gate. Batch context and different API
interfaces can affect answers. The experiment is a semantic prompt/extraction
diagnostic, not calibrated accuracy, capacity, recommendation-quality, or production
training validation. Paid-run results still require human or agent semantic review.
