"""Freeze the local comparison workload. Preparation is offline and credential-free."""

import argparse
import hashlib
import json
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("results/manifest.json"))
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    try:
        manifest = build_manifest(args.data_dir, args.concurrency)
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
    print("No API requests made. Paid execution is not implemented in Step 1.")


if __name__ == "__main__":
    main()
