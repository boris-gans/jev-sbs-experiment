# Jev / GPT nano SBS comparison

A one-off experiment for FERODEV-8257. Compare **400 directed rows per shop**
(200 forward, 200 reverse) for furniture.co.uk's style-compatibility objective
and themeatboys.nl's complements objective. Both providers receive the same
frozen product evidence, shop policies, examples, and candidate batches.

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

The supplied snapshots produce **62 batches per provider**, 124 initial HTTP
requests total, and 1,600 pair evaluations across the two providers. Reverse
batches are often smaller than 20. Preflight prints the actual counts.

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

The separate Jev experiment command/directories are added in Step 6. Until then,
`--execute` without `--prepare-baseline` still means the original two-provider run.

## Original combined paid run — Boris operates this

Keep credentials out of source, shell history, chat, and logs. Supply
`OPENAI_API_KEY` and `TYPESAFE_API_KEY` in the process environment using your local
secret-management method. `.env.sample` lists the names; **the script does not
load `.env` files**.

Before execution, check account limits, current prices, and your intended spend
limit in the provider consoles. There is **no hard dollar-budget limiter**; usage
is reported after requests, and GPT has no output-token override.

After reviewing the offline preflight, run once:

```sh
.venv/bin/python compare.py --execute
```

The four provider/shop groups run separately; batches within each group run
concurrently. Each HTTP attempt has a 120-second I/O timeout. Transient transport
errors, 408, 429, and 5xx get at most two retries (up to 372 attempts for this
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

Credentials are required before requests begin. The run refuses to overwrite an
existing `results/run/requests.jsonl`, preventing an accidental paid rerun into the
same directory. There is no automatic resume. If a retry of the whole experiment
is necessary, inspect failures first, then explicitly choose a new `--output-dir`;
that resends the entire workload and incurs new charges.

## Results

All outputs are ignored by Git and written under `results/run/`:

| File | Contents |
|---|---|
| `requests.jsonl` | Incrementally flushed batch requests, structured responses, per-attempt usage/latency/status, retries, model IDs, and group wall times; no authorization headers |
| `gpt_classifications.jsonl`, `jev_classifications.jsonl` | One record per expected pair/provider; statuses `ok`, `error`, or `missing`; errors/missing rows have null labels |
| `paired_results.csv` | All 800 directed rows with evidence, labels, GPT reasons, Jev probabilities, statuses, and agreement |
| `disagreements.csv` | Only pairs where both providers succeeded but chose different labels |
| `summary.json` | Per-shop agreement matrices and per-provider/shop timing, counts, usage coverage, and cost estimates |

The CSV writer prefixes formula-like text with an apostrophe for spreadsheet
safety. JSON retains original text. Usage belongs to requests, not individual
pairs; it is not duplicated or arbitrarily apportioned across classification rows.

Reports can be regenerated without inputs, credentials, or network access:

```sh
.venv/bin/python compare.py --report-only
```

Use the original `--manifest` and `--output-dir` if you chose custom paths.
Reports verify the manifest's digest against the run log. Only derived reports
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
Batch context and different API interfaces can affect answers. This one run is a
prompt/extraction diagnostic, not a calibrated accuracy or capacity benchmark.
