"""Freeze and compare SBS pairs; network access requires an explicit --execute."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

import httpx


SHOPS = {
    "furniture.co.uk": "style_compatibility",
    "themeatboys.nl": "complements",
}
PROMPTS = {
    "style_compatibility": "stage2_classify_style_compatibility",
    "complements": "stage2_classify_addons",
}
ROWS_PER_DIRECTION = 200
BATCH_SIZE = 20
GPT_SETTINGS = {
    "provider": "openai",
    "model": "gpt-5-nano-2025-08-07",
    "reasoning_effort": "medium",
    "structured_output_enabled": True,
    "structured_output_mode": "auto",
    "temperature": None,
    "max_completion_tokens": None,
}
LABELS = ("hard_negative", "positive", "skip")
ENDPOINTS = {
    "gpt": "https://api.openai.com/v1/chat/completions",
    "jev": "https://api.typesafe.ai/v1/systemone",
}
MAX_RETRIES = 2
TIMEOUT_SECONDS = 120.0
MAX_RETRY_DELAY = 60.0


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def index_records(records: list, key: str) -> dict:
    result = {}
    for record in records:
        identity = record[key]
        require(isinstance(identity, str) and bool(identity), f"Invalid {key}")
        require(identity not in result, f"Duplicate {key}: {identity}")
        result[identity] = record
    return result


def load_shop(data_dir: Path, shop: str) -> dict:
    objective = SHOPS[shop]
    stage2 = Path(shop) / "stage2"
    paths = {
        "products": Path(shop) / "stage1/products_processed.json",
        "subset": stage2 / "subset_products.json",
        "candidates": stage2 / "candidates.json",
        "context": stage2 / "classification_context.json",
        "forward": stage2 / "forward_judgments.json",
        "reverse": stage2 / "reverse_judgments.json",
        "examples": stage2 / "classification_examples.txt",
        "system_prompt": stage2 / f"{PROMPTS[objective]}.system.txt",
        "user_prompt": stage2 / f"{PROMPTS[objective]}.user.txt",
        "recommendation": stage2 / "recommendation.yaml",
    }
    raw = {key: (data_dir / path).read_bytes() for key, path in paths.items()}
    context = json.loads(raw["context"])
    version = context["context_version"]
    payload = {key: value for key, value in context.items() if key != "context_version"}
    require(version == "ctx-v1-" + sha256(canonical_json(payload)), f"{shop}: invalid context version")
    require(context["schema_version"] == "classification-context-v1", "Unsupported context schema")
    require(context["classifier_contract_version"] == "v1", "Unsupported classifier contract")
    require(context["shop"] == shop, "Context shop mismatch")
    require(context["recommendation_objective"] == objective, "Context objective mismatch")
    require(context["llm"] == GPT_SETTINGS, "Recorded GPT settings differ from the approved baseline")
    require(context["retrieval"]["top_k_candidates"] == BATCH_SIZE, "Expected retrieval top-K of 20")
    for key in ("products", "subset", "candidates", "examples", "system_prompt", "user_prompt"):
        require(sha256(raw[key]) == context["inputs"][key], f"{shop}: {key} digest mismatch")

    products = index_records(json.loads(raw["products"]), "product_id")
    subset = index_records(json.loads(raw["subset"]), "product_id")
    retrieval = index_records(json.loads(raw["candidates"]), "anchor_id")
    edges = {}
    for anchor_id, block in retrieval.items():
        require(anchor_id in subset and anchor_id in products, f"Unknown retrieved anchor: {anchor_id}")
        ranks = set()
        for candidate in block["candidates"]:
            candidate_id, rank = candidate["candidate_id"], candidate["rank"]
            require(candidate_id in products, f"Unknown retrieved product: {candidate_id}")
            require(type(rank) is int and rank > 0 and rank not in ranks, "Invalid or duplicate rank")
            require((anchor_id, candidate_id) not in edges, "Duplicate retrieved pair")
            ranks.add(rank)
            edges[anchor_id, candidate_id] = candidate

    evidence = {}
    text_differences = []
    for product_id, product in products.items():
        fields = {key: (product.get(key) or "").strip() for key in ("name", "category", "description")}
        text = ". ".join(value for value in fields.values() if value)
        if product_id in subset:
            stored_text = subset[product_id]["text"]
            require(isinstance(stored_text, str) and bool(stored_text.strip()), "Empty anchor text")
            if stored_text != text:
                text_differences.append(product_id)
            text = stored_text
        evidence[product_id] = {"id": product_id, **fields, "text": text}
    require(set(subset) <= set(products), "Subset contains products missing from Stage 1")

    populations = {}
    duplicates = {}
    for direction in ("forward", "reverse"):
        anchors = {}
        duplicate_count = 0
        for line_number, line in enumerate(raw[direction].decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            block = json.loads(line)
            require(block["prompt_version"] == version, f"{shop}: {direction} line {line_number} context mismatch")
            anchor_id = block["anchor_id"]
            require(anchor_id in evidence, f"Unknown {direction} anchor: {anchor_id}")
            candidates = anchors.setdefault(anchor_id, {})
            for judgment in block["judgments"]:
                candidate_id = judgment["candidate_id"]
                require(candidate_id in evidence, f"Unknown {direction} candidate: {candidate_id}")
                require(candidate_id != anchor_id, "Self-pair in workload")
                edge = (anchor_id, candidate_id) if direction == "forward" else (candidate_id, anchor_id)
                require(edge in edges, f"{shop}: {direction} pair absent from retrieval: {edge}")
                if candidate_id in candidates:
                    duplicate_count += 1
                    continue
                # Deliberately never consult historical label or reason fields.
                candidates[candidate_id] = {
                    "candidate_id": candidate_id,
                    "retrieved_forward_rank": edges[edge]["rank"],
                    "retrieval_source": edges[edge]["source"],
                }
        populations[direction] = {
            anchor: sorted(items.values(), key=lambda item: item["retrieved_forward_rank"])
            if direction == "forward" else list(items.values())
            for anchor, items in anchors.items()
        }
        duplicates[direction] = duplicate_count

    return {
        "context": context,
        "sources": {key: {"path": str(paths[key]), "sha256": sha256(value)} for key, value in raw.items()},
        "policy": {key: raw[key].decode("utf-8") for key in ("system_prompt", "user_prompt", "examples")},
        "populations": populations,
        "products": evidence,
        "diagnostics": {
            "duplicate_judgments_removed": duplicates,
            "subset_text_differs_from_catalog": sorted(text_differences),
        },
    }


def select_batches(shop: str, version: str, direction: str, anchors: dict, count: int) -> list:
    order = sorted(anchors, key=lambda anchor: (sha256(canonical_json([shop, version, direction, anchor])), anchor))
    remaining = count
    batches = []
    for anchor_id in order:
        candidates = anchors[anchor_id]
        for offset in range(0, len(candidates), BATCH_SIZE):
            selected = candidates[offset:offset + min(BATCH_SIZE, remaining)]
            batches.append({
                "batch_id": f"{shop}/{direction}/{len(batches):04d}",
                "direction": direction,
                "anchor_id": anchor_id,
                "candidates": [
                    {**item, "pair_id": json.dumps([shop, direction, anchor_id, item["candidate_id"]], separators=(",", ":"))}
                    for item in selected
                ],
            })
            remaining -= len(selected)
            if remaining == 0:
                return batches
    raise ValueError(f"{shop}: need {count} {direction} rows, found {count - remaining}")


def build_manifest(data_dir: Path, concurrency: int = 8) -> dict:
    require(concurrency > 0, "Concurrency must be positive")
    manifest = {
        "schema_version": "jev-sbs-manifest-v1",
        "selection": {
            "algorithm": "sha256-anchor-blocks-v1",
            "rows_per_direction": ROWS_PER_DIRECTION,
            "batch_size": BATCH_SIZE,
            "anchor_order": "SHA256 of UTF-8 canonical JSON [shop, context_version, direction, anchor_id]",
            "forward_order": "retrieval rank",
            "reverse_order": "first-seen candidate ID across saved judgment blocks",
            "deduplication": "first occurrence of (shop, direction, anchor_id, candidate_id)",
        },
        "models": {"gpt": dict(GPT_SETTINGS), "jev": {"model": "jev-1.13.0"}},
        "concurrency": concurrency,
        "limitations": [
            "Saved reverse workload was selected using historical GPT non-skip judgments.",
            "Sampling is anchor-block based, not uniform over all catalog pairs.",
            "Agreement is not correctness; direction is part of pair identity.",
        ],
        "shops": {},
    }
    for shop, objective in SHOPS.items():
        loaded = load_shop(data_dir, shop)
        version = loaded["context"]["context_version"]
        batches = []
        for direction in ("forward", "reverse"):
            batches.extend(select_batches(shop, version, direction, loaded["populations"][direction], ROWS_PER_DIRECTION))
        product_ids = {batch["anchor_id"] for batch in batches}
        product_ids.update(item["candidate_id"] for batch in batches for item in batch["candidates"])
        manifest["shops"][shop] = {
            "objective": objective,
            "context": loaded["context"],
            "sources": loaded["sources"],
            "policy": loaded["policy"],
            "products": {key: loaded["products"][key] for key in sorted(product_ids)},
            "batches": batches,
            "diagnostics": loaded["diagnostics"],
        }
    return manifest


def write_manifest(path: Path, manifest: dict) -> str:
    content = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError:
        require(path.read_bytes() == content, f"Refusing to replace different manifest: {path}; choose a new path")
    return sha256(content)


def render_request(provider: str, settings: dict, data: dict, batch: dict) -> dict:
    product = data["products"][batch["anchor_id"]]
    base = {"id": product["id"], "text": product["text"]}
    candidates = [
        {key: data["products"][item["candidate_id"]][key] for key in ("id", "name", "category", "description")}
        for item in batch["candidates"]
    ]
    policy = data["policy"]
    examples = policy["examples"].strip()
    if provider == "gpt":
        schema = {
            "type": "object", "additionalProperties": False, "required": ["judgments"],
            "properties": {"judgments": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["id", "label", "reason"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string", "enum": list(LABELS)},
                    "reason": {"type": "string", "maxLength": 80},
                },
            }}},
        }
        return {
            "model": settings["model"], "reasoning_effort": settings["reasoning_effort"],
            "messages": [
                {"role": "system", "content": policy["system_prompt"].replace("{examples}", examples)},
                {"role": "user", "content": policy["user_prompt"].format(
                    anchor_id=base["id"], anchor_text=base["text"],
                    candidates_json=json.dumps(candidates, indent=2),
                )},
            ],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "addon_classification", "strict": True, "schema": schema,
            }},
        }

    require(provider == "jev", "Unknown provider")
    system_policy, marker, _ = policy["system_prompt"].partition("\nOUTPUT\n")
    require(bool(marker), "Missing system OUTPUT boundary")
    _, marker, user_rules = policy["user_prompt"].partition("For each candidate,")
    require(bool(marker), "Missing user decision-rules boundary")
    user_rules, marker, _ = user_rules.partition("\nReturn a JSON object")
    require(bool(marker), "Missing user output boundary")
    return {
        "model": settings["model"],
        "state": {
            "base": base, "candidates": candidates,
            "label_policy": {
                "system": system_policy.replace("{examples}", examples),
                "ordered_rules": "For each candidate," + user_rules,
            },
        },
        "questions": {f"candidate_{i}": {
            "type": "choice",
            "instructions": (
                f"Classify candidates[{i}] as a recommendation for base. "
                "Apply label_policy, including its examples and ordered rules. "
                "Stop at the first applicable check; judge this direction independently."
            ),
            "criteria": {label: f"The {label} outcome defined by label_policy after applying its ordered checks."
                         for label in LABELS},
        } for i in range(len(candidates))},
    }


def prepare_requests(manifest: dict) -> dict:
    return {provider: {
        shop: [(batch, render_request(provider, settings, data, batch)) for batch in data["batches"]]
        for shop, data in manifest["shops"].items()
    } for provider, settings in manifest["models"].items()}


def strict_json(text: str) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("Non-finite JSON value")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def parse_answers(provider: str, body: dict, batch: dict) -> list:
    require(isinstance(body.get("model"), str) and bool(body["model"]), "Missing served model")
    ids = [item["candidate_id"] for item in batch["candidates"]]
    if provider == "gpt":
        require(len(body["choices"]) == 1, "Expected one completion")
        choice = body["choices"][0]
        require(choice["finish_reason"] == "stop", "Incomplete completion")
        require(not choice["message"].get("refusal"), "Provider refusal")
        parsed = strict_json(choice["message"]["content"])
        require(set(parsed) == {"judgments"}, "Invalid completion object")
        answers = index_records(parsed["judgments"], "id")
        require(set(answers) == set(ids), "Missing or unknown candidate answers")
        for answer in answers.values():
            require(set(answer) == {"id", "label", "reason"}, "Invalid judgment fields")
            require(answer["label"] in LABELS, "Invalid label")
            require(isinstance(answer["reason"], str) and len(answer["reason"]) <= 80, "Invalid reason")
        normalized = [{"label": answers[identity]["label"], "reason": answers[identity]["reason"],
                       "probabilities": None, "confidence": None} for identity in ids]
    else:
        answers = body["answers"]
        require(set(answers) == {f"candidate_{i}" for i in range(len(ids))}, "Missing or unknown question answers")
        normalized = []
        for i in range(len(ids)):
            answer = answers[f"candidate_{i}"]
            require(answer["type"] == "choice" and answer["choice"] in LABELS, "Invalid Choice answer")
            probabilities = answer["probabilities"]
            require(set(probabilities) == set(LABELS), "Incomplete probability distribution")
            values = [*probabilities.values(), answer["confidence"]]
            require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                        for value in values), "Invalid probability or confidence")
            require(math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6), "Probabilities do not sum to one")
            require(probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6, "Choice is not maximal")
            normalized.append({"label": answer["choice"], "reason": None,
                               "probabilities": probabilities, "confidence": answer["confidence"]})
    return [{"pair_id": item["pair_id"], "candidate_id": item["candidate_id"], **answer}
            for item, answer in zip(batch["candidates"], normalized, strict=True)]


def normalize_usage(provider: str, raw: dict | None) -> dict:
    raw = {} if raw is None else raw
    require(isinstance(raw, dict), "Invalid usage object")
    if provider == "gpt":
        counts = {
            "input_tokens": raw.get("prompt_tokens"), "output_tokens": raw.get("completion_tokens"),
            "cached_input_tokens": (raw.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "reasoning_tokens": (raw.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        }
    else:
        counts = {"input_tokens": raw.get("input_tokens"), "output_tokens": raw.get("output_tokens"),
                  "cached_input_tokens": None, "reasoning_tokens": None}
    require(all(value is None or (type(value) is int and value >= 0) for value in counts.values()), "Invalid usage counts")
    return counts


def retry_delay(header: str | None, attempt: int) -> float | None:
    delay = float(2 ** attempt)
    if header:
        try:
            delay = float(header)
        except ValueError:
            try:
                delay = (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                pass
    if not math.isfinite(delay):
        return None
    # Stop rather than retry sooner than a server's long Retry-After asks for.
    return max(0.0, delay) if delay <= MAX_RETRY_DELAY else None


def run_batch(client: httpx.Client, provider: str, shop: str, batch: dict, payload: dict, key: str) -> dict:
    started = time.perf_counter()
    record = {
        "record_type": "batch", "provider": provider, "shop": shop,
        "batch_id": batch["batch_id"], "direction": batch["direction"], "anchor_id": batch["anchor_id"],
        "pair_ids": [item["pair_id"] for item in batch["candidates"]], "batch_size": len(batch["candidates"]),
        "requested_model": payload["model"], "served_model": None, "request": payload,
        "status": "error", "error": None, "answers": [], "raw_response": None, "attempts": [],
    }
    for number in range(MAX_RETRIES + 1):
        attempt = {"number": number + 1, "http_status": None, "request_id": None, "served_model": None,
                   "usage": normalize_usage(provider, None), "raw_usage": None, "error": None,
                   "retry_delay_seconds": None}
        retry = False
        header = None
        tick = time.perf_counter()
        try:
            response = client.post(ENDPOINTS[provider], json=payload, headers={"Authorization": f"Bearer {key}"})
        except httpx.TransportError as error:
            attempt["error"] = type(error).__name__  # Exception messages may contain sensitive request details.
            retry = True
        else:
            attempt["http_status"] = response.status_code
            attempt["request_id"] = response.headers.get("x-request-id" if provider == "gpt" else "x-typesafe-request-id")
            header = response.headers.get("retry-after")
        attempt["latency_seconds"] = time.perf_counter() - tick
        if attempt["http_status"] is not None:
            status = attempt["http_status"]
            try:
                body = strict_json(response.text)
                require(isinstance(body, dict), "Invalid response object")
                if status == 200:
                    record["raw_response"] = body
                attempt["served_model"] = body.get("model")
                attempt["raw_usage"] = body.get("usage")
                attempt["usage"] = normalize_usage(provider, attempt["raw_usage"])
                if status == 200:
                    record["answers"] = parse_answers(provider, body, batch)
                    record["status"] = "ok"
            except (ValueError, KeyError, TypeError, AttributeError, IndexError):
                attempt["error"] = "invalid_response"
            if status != 200:
                attempt["error"] = f"http_{status}"
                retry = status in (408, 429) or 500 <= status < 600
        record["attempts"].append(attempt)
        record["served_model"] = attempt["served_model"]
        record["error"] = attempt["error"]
        if not retry or number == MAX_RETRIES:
            break
        delay = retry_delay(header, number)
        if delay is None:
            attempt["retry_not_scheduled"] = "retry_after_exceeds_budget"
            break
        attempt["retry_delay_seconds"] = delay
        time.sleep(delay)
    record["retries"] = len(record["attempts"]) - 1
    record["wall_seconds"] = time.perf_counter() - started
    return record


def execute(manifest: dict, jobs: dict, output_dir: Path, manifest_digest: str, credentials: dict,
            *, transport: httpx.BaseTransport | None = None) -> int:
    for provider in ENDPOINTS:
        require(bool(credentials.get(provider)), f"Missing credential for {provider}")
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    # One writer owns the file; exclusive creation prevents an accidental paid rerun.
    with (output_dir / "requests.jsonl").open("x", encoding="utf-8") as handle:
        def emit(record):
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()

        emit({"record_type": "run_start", "started_at": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": manifest_digest, "script_sha256": sha256(Path(__file__).read_bytes()),
              "concurrency": manifest["concurrency"], "timeout_seconds": TIMEOUT_SECONDS,
              "max_retries": MAX_RETRIES, "max_retry_delay_seconds": MAX_RETRY_DELAY})
        for provider, shops in jobs.items():
            for shop, work in shops.items():
                emit({"record_type": "group_start", "provider": provider, "shop": shop,
                      "requested_model": manifest["models"][provider]["model"], "batches": len(work)})
                started = time.perf_counter()
                with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=False, transport=transport) as client:
                    with ThreadPoolExecutor(max_workers=manifest["concurrency"]) as pool:
                        futures = [pool.submit(run_batch, client, provider, shop, batch, payload, credentials[provider])
                                   for batch, payload in work]
                        for future in as_completed(futures):
                            record = future.result()
                            emit(record)
                            failures += record["status"] != "ok"
                emit({"record_type": "group_end", "provider": provider, "shop": shop,
                      "wall_seconds": time.perf_counter() - started})
        emit({"record_type": "run_end", "failed_batches": failures})
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("results/manifest.json"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--execute", action="store_true", help="Make paid API requests (Boris runs this explicitly)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/run"))
    args = parser.parse_args()
    try:
        manifest = build_manifest(args.data_dir, args.concurrency)
        jobs = prepare_requests(manifest)
        digest = write_manifest(args.manifest, manifest)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(f"Offline manifest: {args.manifest} (SHA256 {digest})")
    print(f"Models: {GPT_SETTINGS['model']} / jev-1.13.0; concurrency: {args.concurrency}")
    for shop, data in manifest["shops"].items():
        for direction in ("forward", "reverse"):
            batches = [batch for batch in data["batches"] if batch["direction"] == direction]
            sizes = [len(batch["candidates"]) for batch in batches]
            print(f"{shop} {direction}: {sum(sizes)} rows, {len(batches)} requests/provider, batch sizes {sizes}")
        print(f"  Source checks: {json.dumps(data['diagnostics'], sort_keys=True)}")
    print(f"Request log: {args.output_dir / 'requests.jsonl'}; groups run separately: GPT then Jev, shop by shop.")
    if not args.execute:
        print("No API requests made. Use --execute only when ready for the paid run.")
        return
    print("Executing paid requests; retries may incur additional cost.", flush=True)
    try:
        credentials = {"gpt": os.environ.get("OPENAI_API_KEY"), "jev": os.environ.get("TYPESAFE_API_KEY")}
        failures = execute(manifest, jobs, args.output_dir, digest, credentials)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Execution failed: {error}\n")
    if failures:
        parser.exit(1, f"Run finished with {failures} failed batches; inspect requests.jsonl before any rerun.\n")
    print("All batches completed. Request data saved; comparison reports are added in Step 3.")


if __name__ == "__main__":
    main()
