# Jev / GPT nano SBS comparison

A small local experiment for FERODEV-8257. Compare **400 directed rows per shop**
(200 forward, 200 reverse) for furniture.co.uk's style-compatibility objective
and themeatboys.nl's complements objective. Both providers receive the same
frozen product evidence, shop policies, examples, and candidate batches. Prepare
GPT once, then compare successive Jev prompt variants against that saved baseline.

This is not a production pipeline integration. Nothing imports or changes
`embedding-service`; no retrieval, summarization, training, or embedding job runs.

## Setup

Python **3.11+** and `uv`:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r pyproject.toml --extra test
.venv/bin/python -m pytest
```

There is no build/package or configured linter. `git diff --check` checks patch
whitespace. Tests use synthetic catalogs and mocked HTTP; they never call providers.

## Inputs and offline preflight

The transferred files remain local under `data/<shop>/`. Required per shop:

- `stage1/products_processed.json`
- `stage2/subset_products.json`, `candidates.json`, `classification_context.json`
- `stage2/forward_judgments.json` and `reverse_judgments.json` (JSONL despite their extensions)
- `stage2/recommendation.yaml`, `classification_examples.txt`, and the objective's
  `stage2_classify_*.system.txt` / `stage2_classify_*.user.txt`

Run from this repository root:

```sh
.venv/bin/python compare.py
```

This checks context hashes and product references, renders both provider payloads,
and writes `results/manifest.json`. No credentials or network requests are needed.
Repeating preparation checks that the manifest is byte-identical; a different
workload/settings combination requires a new `--manifest` path, never an overwrite.

The manifest contains selected evidence, policy, provenance, batching, and models:

- GPT: `gpt-5-nano-2025-08-07`, medium reasoning, strict structured output;
  no temperature or output-token override.
- Jev: `jev-1.13.0`, one three-label Choice question per candidate.
- At most 20 candidates per request; concurrency defaults to 8 (`--concurrency`).

The supplied snapshots produce **62 batches per provider**. Each new experiment
sends only the 62 Jev batches (800 pairs) and reuses GPT's labels. Reverse batches
are often smaller than 20. Preflight prints the actual counts.

## Reusable GPT baseline — Boris operates paid completion

Import the existing GPT responses **offline** (including the 795 valid labels
already returned). This preserves the original run and does not call either API:

```sh
.venv/bin/python compare.py --prepare-baseline --import-run results/run
```

Baseline data lives under `results/baselines/<gpt-input-fingerprint>/`:

- `spec.json`: exact GPT requests and directed pair identities, published once.
- `imported.jsonl`: immutable snapshot of the initial run; only GPT records are used.
- `repairs/*.jsonl`: a new durable log for each paid completion invocation, with
  each HTTP attempt flushed separately. Prior logs are never appended to or replaced.
- `baseline.json`: created only when all expected labels exist; immutable thereafter.

The fingerprint covers GPT prompts/examples, product evidence, candidate grouping,
pair direction, model and request settings. Jev settings/criteria are excluded.
Only the first valid answer for each pair is retained; later answers cannot replace
it. Changing GPT inputs selects a different baseline, never silently reuses old labels.

With `OPENAI_API_KEY` set securely, complete the missing answers explicitly:

```sh
.venv/bin/python compare.py --prepare-baseline --execute
```

Only incomplete batches are requested. The current baseline needs **two batches
to fill five missing answers**, not a rerun of all 62 GPT batches. Retries send the
original full prompt/candidate batch so context stays unchanged; only missing IDs
are merged. Partial-answer and transient-error retries share a limit of **three
HTTP attempts per incomplete batch per invocation**. Completed batches are skipped.
No Jev credential or call is involved. Repeating this command after completion
makes zero requests, even without credentials.

If still incomplete, the command exits nonzero with progress preserved. A later
explicit invocation resumes from the saved answers; it is not an unlimited loop.
An interrupted attempt that reached the provider but was not persisted might be
billed again. Completed attempts are checkpointed before retrying. Historical usage
and wall times remain separate from repair usage and wall times in the baseline.
File locking prevents concurrent baseline preparation on macOS/Linux. Completed
baseline loading is offline-only and rejects absent, incomplete, or changed history.

## Repeatable Jev experiments — Boris operates paid execution

Once the GPT baseline is complete, set only `TYPESAFE_API_KEY` for experiments.
`OPENAI_API_KEY` is not required and GPT is never called by this command. An absent,
incomplete, or mismatched baseline is an error, not permission to rebuild it online.

Keep credentials out of source, shell history, chat, and logs. Use your local
secret-management method. `.env.sample` lists the names; **the script does not load
`.env` files**.

Before execution, check account limits, current prices, and your intended spend
limit in the provider consoles. There is **no hard dollar-budget limiter**; usage
is reported after requests, and GPT has no output-token override.

After reviewing the offline preflight, run a Jev experiment:

```sh
.venv/bin/python compare.py --execute
```

Every invocation creates a fresh `results/experiments/<UTC-timestamp>-<unique-id>/`
directory and prints its path. You may choose a directory with `--output-dir`, but
it must not exist. `results/run/` (the original combined run) remains intact.

The two Jev shop groups run separately; batches within each group run concurrently.
Each HTTP attempt has a 120-second I/O timeout. Transient transport errors, 408,
429, and 5xx get at most two retries (up to 186 Jev attempts for this
workload). `Retry-After` is honored up to 60 seconds; longer waits stop that batch
rather than retrying early. Retries, including requests that timed out after being
processed remotely, can incur additional cost. Authentication/validation failures
and malformed successful responses are not retried. GPT responses missing candidate
answers are retried within the same attempt budget, retaining first-valid answers.

A stopped GPT completion can omit candidate IDs. Valid returned judgments are
retained as a partial batch; omitted candidates remain explicit errors with null
labels, never `skip`. Unknown/duplicate IDs and malformed judgments still fail
validation. Jev's three probabilities may sum within 0.015 of one to accommodate
two-decimal rounding. Their reported values and selected label are preserved, not
renormalized; classification JSONL and summary counts flag rounding discrepancies.

Each new experiment resends the complete Jev workload and incurs new charges;
there is no automatic Jev resume. Use `--report-only` to inspect/regenerate existing
results without new inference. Never rerun `--execute` merely to obtain reports.

## Jev prompt variants

`compare.py` contains separate `JEV_INSTRUCTIONS` and `JEV_CRITERIA` entries for
`style_compatibility` and `complements`, plus the descriptive `JEV_PROMPT_VERSION`.
The original shop policy/examples remain in shared state; each candidate receives
its own Choice with the applicable objective's complete ordered decision procedure.
The prompts explicitly define:

- Furniture: variants and redundancy first, conventional complements next, then
  distinct-model compatibility requiring both a compatible context and concrete
  material, finish, construction, shape, or design-era evidence.
- Complements: variants (including same-brand/same-category line extensions) and
  redundancy first, then a direct functional add-on—not merely another related
  product, extra unit, broad shared activity, or meal variety.
- Hard negative: only after earlier checks, for confident substitution or explicit
  incompatibility/wrong context. Uncertainty defaults to `skip`.
- Each Choice criterion also contains short catalog-grounded boundary examples.
  Furniture distinguishes named family variants, matching room furniture, and
  explicit wrong-size/wrong-context products. Complements distinguishes same-cut
  variants, alternative main proteins, meal variety, and model-matched accessories.
  These teach shared rules and never identify product IDs as exceptions.

For a prompt-only experiment, change the applicable objective's Jev definitions and
run again. Do not edit the frozen GPT templates, input data, or candidate batches.
Prompt changes are snapshotted per experiment and do not invalidate the saved GPT
baseline.

## Results

All outputs are ignored by Git. Each experiment directory contains:

| File | Contents |
|---|---|
| `manifest.json` | Frozen workload snapshot |
| `baseline.json` | Completed GPT labels, provenance, and historical accounting; no copy of raw GPT logs |
| `experiment.json` | Exact Jev criteria/instructions/version, dated pricing, baseline and manifest digests, and intended payload hashes |
| `requests.jsonl` | Only this experiment's Jev requests/responses, attempt usage/latency/status, retries, model IDs, and group wall times; no authorization headers |
| `gpt_classifications.jsonl`, `jev_classifications.jsonl` | One record per expected pair/provider; statuses `ok`, `error`, or `missing`; errors/missing rows have null labels |
| `paired_results.csv` | All 800 directed rows with evidence, labels, GPT reasons, Jev probabilities, statuses, and agreement |
| `disagreements.csv` | Only pairs where both providers succeeded but chose different labels |
| `summary.json` | Agreement matrices, new Jev timing/cost/usage, and separately identified historical GPT baseline measurements |

The CSV writer prefixes formula-like text with an apostrophe for spreadsheet
safety. JSON retains original text. Usage belongs to requests, not individual
pairs; it is not duplicated or arbitrarily apportioned across classification rows.

Reports can be regenerated without inputs, credentials, or network access:

```sh
.venv/bin/python compare.py --report-only --output-dir results/experiments/YOUR-RUN-ID
```

Replace `YOUR-RUN-ID` with the printed directory name. Reports use that directory's
snapshots: no original catalog, baseline directory, credentials, or current Jev
criteria are needed. They verify manifest/baseline/experiment digests and logged
payload hashes, so changing the code's criteria or pricing cannot rewrite history.
An explicit `--manifest` must match the snapshot.

For the original combined run, use `--report-only --output-dir results/run`;
omitting `--output-dir` in report-only mode also selects that legacy run. Its
original report format is retained. Only derived reports
are overwritten; the request log and manifest are never overwritten. Interrupted
runs retain completed batches; an unterminated, malformed final log line is
flagged and omitted. Malformed interior lines or conflicting identities fail
reporting rather than silently mixing results.

Report-only processing also re-extracts saved HTTP-200 validation failures with
the current parser. This recovers valid answers rejected by earlier batch-wide
checks **without new API calls or edits to the original request log**. Reports
record the extraction version and source-log hash. `recorded_failed_attempts`
retains the original failure count; `failed_attempts` reflects current validation.
`partial_batches` have some valid answers; `failed_batches` have none.
Cost, token usage, request timing, and wall time remain the original measurements.

## Reading the numbers

- **Agreement is not correctness.** Its denominator includes only paired valid
  results; failed/missing responses are excluded and separately counted. A valid
  `skip` is a real label, not an operational failure.
- **New versus historical:** experiment summaries contain only fresh Jev inference
  under `groups` and `new_inference`. GPT has zero new calls/cost; its original and
  repair measurements live under `gpt_baseline.accounting`. Cached lookup time is
  not GPT inference latency, and historical GPT cost is not charged to each run.
- **Wall time** covers each provider/shop group including retries and persistence,
  excluding offline preparation. Throughput is successful pairs divided by that
  time. Incomplete groups have null wall time/throughput.
- **Request latency** is measured per HTTP attempt, including failed attempts,
  but excludes retry sleeps. p50 is the median; p95 is nearest-rank. Logical batch
  counts and HTTP-attempt/retry counts are separate.
- **Usage** includes every recorded attempt, not just successful batches. Each
  field has a reported subtotal, missing-attempt count, and a total only when the
  group and field coverage are complete. Null never means zero.
- **Costs are dated estimates**, not bills. Rates as of 2026-09-17, USD per million
  tokens: GPT input $0.05, cached input $0.005, output $0.40; Jev input $0.042,
  output free. GPT completion tokens already include reasoning tokens. A GPT
  attempt needs input, cached-input, and output counts to be priced; a Jev attempt
  needs input counts. Missing counts leave the total unknown and expose only a
  priced-attempt subtotal. In-flight attempts lost on interruption may still be
  billed. Reconcile against the provider consoles.

Pricing references: [OpenAI](https://platform.openai.com/docs/pricing),
[TypeSafe](https://docs.typesafe.ai/models).

## Interpretation limits

Sampling uses stable-hash anchor ordering and retains candidate order, not uniform
sampling of the full catalog. Historical GPT labels do not select sample rows,
but the **saved reverse population was originally chosen from GPT non-skip forward
judgments**. Forward/reverse direction is part of record identity; the same ordered
product IDs may appear in both populations and are not independent observations.
Batch context and different API interfaces can affect answers. These experiments
are prompt/extraction diagnostics, not calibrated accuracy or capacity benchmarks.
