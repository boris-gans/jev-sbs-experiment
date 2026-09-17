# jev-sbs-experiment

## What this repo does
One-off comparison of Jev and GPT nano on frozen SBS product pairs from
furniture.co.uk and themeatboys.nl. This is a standalone research script,
not a production pipeline integration.

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

## Public interfaces other repos depend on
None known. This local experiment does not change embedding-service contracts.

## Gotchas
- Boris owns the paid run. Do not invoke `--execute` autonomously.
- Credentials come from OPENAI_API_KEY and TYPESAFE_API_KEY in the process
  environment. Never open .env files or log credentials.
- data/, results/, and .claude/plans/ are ignored; never commit their contents.
- Preparation refuses to replace a different manifest. Execution refuses to
  overwrite a request log; there is no automatic resume.
- Failures are not skip labels, missing usage is not zero, and agreement with
  GPT is not ground truth. Read README.md before interpreting results.
