# jev-sbs-experiment

## What this repo does
Local comparison of Jev and GPT nano identity, co-purchase, alternative, and
incompatibility judgments on frozen SBS pairs from furniture.co.uk and
themeatboys.nl. Deterministic code derives basket labels from the raw heads. This
is a standalone research script, not a production pipeline integration.

## Stack
Python 3.11+, httpx 0.28.1, pytest 8.x. One script: compare.py.

## Supported platform versions
n/a

## Commands
Run from the repository root after the README setup.
- Test: `.venv/bin/python -m pytest`
- Build: n/a — standalone script, no build step.
- Lint: n/a — no linter configured; `git diff --check` checks whitespace.
- Offline preflight: `.venv/bin/python compare.py`
- Offline reports: `.venv/bin/python compare.py --report-only`
- For a newer experiment, add `--output-dir` with its printed directory path.
- Offline fresh GPT baseline preparation: `.venv/bin/python compare.py --prepare-baseline`

## Public interfaces other repos depend on
None known. This local experiment does not change embedding-service contracts.

## Gotchas
- Boris owns the paid run. Do not invoke `--execute` autonomously.
- Credentials come from OPENAI_API_KEY and TYPESAFE_API_KEY in the process
  environment. Never open .env files or log credentials.
- data/, results/, and .claude/plans/ are ignored; never commit their contents.
- Worktrees do not inherit ignored data/results; use `--data-dir` to point at the
  original frozen catalog without copying or editing it.
- Plain `--execute` runs only Jev, requires a completed matching GPT baseline, and
  creates a fresh experiment directory. It never starts GPT inference implicitly.
- Preparation refuses to replace a different manifest. GPT baseline completion
  reuses first-valid answers; paid completion needs `--prepare-baseline --execute`.
- FERODEV-8257's GPT labels and `results/run` are not objective-aligned and must not
  seed the v2 baseline. A fresh baseline starts with all 800 judgments missing.
- Keep each experiment's snapshots and raw log immutable. Report-only processing
  uses those snapshots, rejects unavailable resolver versions, and separates cached
  GPT measurements from new Jev inference.
- Baseline preparation uses macOS/Linux file locks. Completed baselines and their
  input/history files are immutable; Jev-only criteria changes do not invalidate
  GPT, while GPT prompt/schema or evidence changes select a new baseline.
- Combined requests ask identity and all three relation heads for every pair. This
  supports direct provider comparison but does not measure two-pass cost savings.
- Failures are not uncertainty, skip, or conflict labels; missing usage is not zero;
  and agreement with GPT is not ground truth. Read README.md before interpreting
  results.
