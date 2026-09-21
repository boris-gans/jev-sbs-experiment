"""Freeze and compare SBS pairs; network access requires an explicit --execute."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import threading
import time
import uuid

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
IDENTITY_LABELS = ("duplicate", "variant", "redundant", "distinct", "uncertain")
RELATION_LABELS = ("yes", "no", "uncertain")
RELATION_HEADS = ("co_purchase", "alternative", "incompatible")
BASKET_LABELS = ("positive", "hard_negative", "skip", "conflict")
RELATION_CONTRACT_VERSION = "universal-relations-v1"
RESOLVER_VERSION = "basket-projection-v1"
ENDPOINTS = {
    "gpt": "https://api.openai.com/v1/chat/completions",
    "jev": "https://api.typesafe.ai/v1/systemone",
}
MAX_RETRIES = 2
TIMEOUT_SECONDS = 120.0
MAX_RETRY_DELAY = 60.0
PROBABILITY_SUM_TOLERANCE = 0.015  # Three probabilities rounded to two decimal places.
EXTRACTION_VERSION = "partial-answers-v2"
JEV_PROMPT_VERSION = "objective-boundaries-v3"
JEV_INSTRUCTIONS = {
    "style_compatibility": (
        "Judge candidates[{index}] as a STYLE-COMPATIBILITY recommendation for base in this direction. "
        "Apply these checks in order and stop at the first match: "
        "(1) same named model or product family differing only by size, colour, material, finish, shape, or another "
        "variant axis -> skip; a shared collection with distinct model names is not by itself a variant; "
        "(2) duplicate, or already included or integrated in base -> skip; "
        "(3) clear accessory, refill, compatible replacement part, or setup component used with base -> positive; "
        "(4) distinct models in the same primary use context, or credibly coordinating in one setting, with concrete "
        "material, finish, construction, shape, or design-era evidence -> positive; "
        "(5) explicit wrong functional role, wrong use context, incompatibility, or contradictory design evidence "
        "that makes the candidate confidently unsuitable -> hard_negative; (6) otherwise -> skip. "
        "Same role alone is not negative. Collection, category, brand, colour, or generic material alone is not "
        "positive. Do not reverse the recommendation direction."
    ),
    "complements": (
        "Judge candidates[{index}] as a COMPLEMENTS recommendation for base in this direction. "
        "Apply these checks in order and stop at the first match: "
        "(1) same kind of product differing only by size, colour, volume, scent, flavour, material, origin, finish, "
        "or another variant axis, or same brand and same category product-line extension or successor -> skip; "
        "(2) duplicate, or a non-consumable already included or integrated in base -> skip; "
        "(3) clear accessory, setup component, attachment, refill, cover, case, compatible part, or product with a "
        "direct complementary function used with base -> positive; "
        "(4) confident same-role substitute bought instead of base -> hard_negative; "
        "(5) strongly related but explicitly incompatible, unusable, or wrong-context candidate -> hard_negative; "
        "(6) otherwise -> skip. A consumable refill is not redundant. Merely another related product, additional "
        "meal variety, another unit, or something used in the same broad activity without a direct complementary "
        "function is not positive. Do not reverse the recommendation direction."
    ),
}
JEV_CRITERIA = {
    "style_compatibility": {
        "positive": (
            "Checks 1-2 do not apply. Check 3 or 4 applies: the candidate is a clear functional complement, or it "
            "is a distinct model with compatible use context and concrete style/material evidence. Distinct models "
            "may serve the same role. Weak similarity alone is insufficient. Boundary examples: a dining table "
            "with its matching chair is positive; a side table and a distinct console with stated matching "
            "construction, finish, and room context can be positive."
        ),
        "hard_negative": (
            "Checks 1-4 do not apply and check 5 applies: supplied text explicitly establishes incompatibility, a "
            "wrong functional role or use context, or contradictory design evidence that makes the candidate "
            "unsuitable. A merely different or shared role alone is not enough. Uncertainty is skip. Boundary "
            "examples: a distinct 6ft mattress for a 4ft6 use context is hard_negative; a storage bench and a pet "
            "bed are hard_negative despite a shared collection or finish."
        ),
        "skip": (
            "Check 1, 2, or 6 applies: exact-model/family variant, duplicate, redundancy, unresolved relationship, "
            "or only weak collection/category/brand/colour/generic-material similarity. Variant and redundancy "
            "checks override every later check. Boundary examples: the same named Juliette bed at 4ft6 versus 5ft "
            "is skip; the same named Tetbury 2-basket bench in White versus Truffle is skip."
        ),
    },
    "complements": {
        "positive": (
            "Checks 1-2 do not apply and check 3 applies: the candidate has a clear direct complementary function "
            "with base as an accessory, setup component, attachment, refill, cover, case, or compatible part. "
            "Relatedness, meal variety, another unit, or broad shared activity alone is insufficient. Boundary "
            "examples: from a Kamado MEDIUM base, a model-matched MEDIUM case or grate is positive; from a product "
            "that consumes a stated refill, that compatible refill is positive. Do not reverse these relationships."
        ),
        "hard_negative": (
            "Checks 1-3 do not apply and check 4 or 5 applies: the candidate is confidently a same-role substitute "
            "bought instead of base, or supplied text explicitly makes a strongly related candidate incompatible, "
            "unusable, or wrong-context. Variants and uncertainty are never hard negatives. Boundary examples: a "
            "distinct steak cut or burger offered instead of the base steak fills the same purchase role and is "
            "hard_negative; a MEDIUM accessory paired with an explicitly LARGE-only appliance or part is "
            "hard_negative."
        ),
        "skip": (
            "Check 1, 2, or 6 applies: variant or same-brand/same-category line extension, duplicate, redundancy, "
            "unrelated or merely related product, meal variety without a direct complementary function, or an "
            "unclear relationship. Variant and redundancy checks override every later check. Boundary examples: "
            "the same ribeye or tenderloin with another origin, grade, or pack size is skip; ham versus pulled pork, "
            "chorizo, or a mixed protein box is skip when the text supports only additional meal variety."
        ),
    },
}
IDENTITY_CRITERIA = {
    "duplicate": "The two listings describe the same product, not merely similar products.",
    "variant": (
        "The same named model or product family differs only on a variant axis such as size, colour, finish, scent, "
        "origin, grade, or pack size."
    ),
    "redundant": (
        "In this direction the base already includes or integrates the candidate, so buying the candidate repeats "
        "something supplied by the base."
    ),
    "distinct": "The candidate is a genuinely distinct product from the base.",
    "uncertain": "The supplied product evidence cannot safely establish identity or directional redundancy.",
}
RELATION_CRITERIA = {
    "co_purchase": {
        "yes": (
            "After selecting the base, the shopper would reasonably add the candidate to the same purchase. The "
            "supplied text establishes a direct accessory, component, refill, setup, room, meal, outfit, activity, "
            "or other concrete together-purchase relationship."
        ),
        "no": "The supplied text establishes that the candidate is not useful to buy with the selected base.",
        "uncertain": "The supplied text is insufficient to establish or reject a together-purchase relationship.",
    },
    "alternative": {
        "yes": (
            "Before selecting the base, the shopper could reasonably buy the distinct candidate instead. It has a "
            "compatible functional role and use context supported by the supplied text."
        ),
        "no": "The supplied text establishes that the candidate is not a credible instead-of choice for the base.",
        "uncertain": "The supplied text is insufficient to establish or reject an alternative relationship.",
    },
    "incompatible": {
        "yes": (
            "The candidate is strongly related to the base but explicitly unusable, wrong-context, or incompatible "
            "according to the supplied text."
        ),
        "no": "The supplied text establishes no explicit incompatibility or wrong-context relationship.",
        "uncertain": "The supplied text is insufficient to establish or reject incompatibility.",
    },
}
RELATION_INSTRUCTIONS = {
    "co_purchase": (
        "Judge candidates[{index}] independently for CO-PURCHASE in this direction: after selecting base, would the "
        "shopper reasonably add the candidate to the same purchase? Concrete functional or coordinated-use evidence "
        "is required; broad similarity alone is insufficient. Do not suppress this answer because of identity."
    ),
    "alternative": (
        "Judge candidates[{index}] independently as an ALTERNATIVE in this direction: before selecting base, would "
        "the shopper reasonably buy the candidate instead? Require a compatible functional role and use context. "
        "Do not suppress this answer because of identity."
    ),
    "incompatible": (
        "Judge candidates[{index}] independently for INCOMPATIBILITY in this direction: is the candidate strongly "
        "related but explicitly unusable, wrong-context, or incompatible with base? Do not infer incompatibility "
        "from missing evidence or suppress this answer because of identity."
    ),
}
GPT_RELATION_SYSTEM_PROMPT = """You evaluate directed product pairs for four separate SBS decisions.

IDENTITY
- duplicate: the same product appears as two listings.
- variant: the same named model or family differs only by size, colour, finish, scent, origin, grade, pack size, or another variant axis.
- redundant: in this direction, the base already includes or integrates the candidate.
- distinct: genuinely different products.
- uncertain: the supplied evidence cannot safely establish identity.

RELATIONS
- co_purchase: after selecting the base, would the shopper reasonably add the candidate to the same purchase?
- alternative: before selecting the base, would the shopper reasonably buy the candidate instead?
- incompatible: is the candidate strongly related but explicitly unusable, wrong-context, or incompatible?

Answer each relation independently with yes, no, or uncertain. Do not omit or change relation answers because of the identity answer. Require concrete evidence from the supplied name, category, and description. Broad category, brand, colour, or thematic similarity alone is insufficient. Missing evidence is uncertain, not no. Preserve direction and do not invent product facts.
"""
PRICING = {
    "as_of": "2026-09-17", "currency": "USD", "unit": "per million tokens",
    "gpt": {"input": 0.05, "cached_input": 0.005, "output": 0.40,
            "source": "https://platform.openai.com/docs/pricing"},
    "jev": {"input": 0.042, "output": 0.0, "source": "https://docs.typesafe.ai/models"},
    "note": "Estimates only; provider consoles are authoritative. Missing usage is not zero cost.",
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


def parse_recommendation_objective(raw: bytes) -> str:
    values = {}
    for line in raw.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, marker, value = stripped.partition(":")
        require(bool(marker) and bool(key.strip()) and bool(value.strip()), "Invalid recommendation config line")
        key, value = key.strip(), value.strip()
        require(key not in values, f"Duplicate recommendation config key: {key}")
        values[key] = value
    require(set(values) == {"version", "objective"}, "Recommendation config requires only version and objective")
    require(values["version"] == "1", "Unsupported recommendation config version")
    require(values["objective"] in PROMPTS, "Unsupported recommendation objective")
    return values["objective"]


def load_shop(data_dir: Path, shop: str) -> dict:
    stage2 = Path(shop) / "stage2"
    recommendation_path = stage2 / "recommendation.yaml"
    recommendation = (data_dir / recommendation_path).read_bytes()
    objective = parse_recommendation_objective(recommendation)
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
        "recommendation": recommendation_path,
    }
    raw = {key: (data_dir / path).read_bytes() for key, path in paths.items()}
    require(raw["recommendation"] == recommendation, f"{shop}: recommendation changed while loading")
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
        "objective": objective,
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
    for shop in SHOPS:
        loaded = load_shop(data_dir, shop)
        objective = loaded["objective"]
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
            "instructions": JEV_INSTRUCTIONS[data["objective"]].format(index=i),
            "criteria": {label: JEV_CRITERIA[data["objective"]][label] for label in LABELS},
        } for i in range(len(candidates))},
    }


def relation_products(data: dict, batch: dict) -> tuple[dict, list]:
    product = data["products"][batch["anchor_id"]]
    base = {"id": product["id"], "text": product["text"]}
    candidates = [
        {key: data["products"][item["candidate_id"]][key] for key in ("id", "name", "category", "description")}
        for item in batch["candidates"]
    ]
    return base, candidates


def relation_result_schema(labels: tuple[str, ...]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "reason"],
        "properties": {
            "label": {"type": "string", "enum": list(labels)},
            "reason": {"type": "string", "maxLength": 80},
        },
    }


def render_relation_request(provider: str, settings: dict, data: dict, batch: dict) -> dict:
    base, candidates = relation_products(data, batch)
    if provider == "gpt":
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["judgments"],
            "properties": {
                "judgments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "identity", "relations"],
                        "properties": {
                            "id": {"type": "string"},
                            "identity": relation_result_schema(IDENTITY_LABELS),
                            "relations": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": list(RELATION_HEADS),
                                "properties": {
                                    head: relation_result_schema(RELATION_LABELS) for head in RELATION_HEADS
                                },
                            },
                        },
                    },
                },
            },
        }
        user_prompt = (
            "Base product:\n"
            f"{json.dumps(base, indent=2, ensure_ascii=False)}\n\n"
            "Candidates, in request order:\n"
            f"{json.dumps(candidates, indent=2, ensure_ascii=False)}\n\n"
            "Return exactly one judgment per candidate ID. Give identity and all three independent relation heads. "
            "Keep every reason to one short phrase of at most 80 characters."
        )
        return {
            "model": settings["model"],
            "reasoning_effort": settings["reasoning_effort"],
            "messages": [
                {"role": "system", "content": GPT_RELATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "relation_classification", "strict": True, "schema": schema},
            },
        }

    require(provider == "jev", "Unknown provider")
    questions = {}
    for index in range(len(candidates)):
        questions[f"candidate_{index}_identity"] = {
            "type": "choice",
            "instructions": (
                f"Classify the IDENTITY of candidates[{index}] relative to base in this direction. Identity is "
                "separate from whether the products complement, replace, or conflict with each other."
            ),
            "criteria": IDENTITY_CRITERIA,
        }
        for head in RELATION_HEADS:
            questions[f"candidate_{index}_{head}"] = {
                "type": "choice",
                "instructions": RELATION_INSTRUCTIONS[head].format(index=index),
                "criteria": RELATION_CRITERIA[head],
            }
    return {
        "model": settings["model"],
        "state": {
            "contract_version": RELATION_CONTRACT_VERSION,
            "base": base,
            "candidates": candidates,
        },
        "questions": questions,
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

    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), "Non-finite JSON value")
        return number

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant, parse_float=finite_float)


def parse_answers(provider: str, body: dict, batch: dict) -> list:
    require(isinstance(body.get("model"), str) and bool(body["model"]), "Missing served model")
    items = batch["candidates"]
    ids = [item["candidate_id"] for item in items]
    if provider == "gpt":
        require(len(body["choices"]) == 1, "Expected one completion")
        choice = body["choices"][0]
        require(choice["finish_reason"] == "stop", "Incomplete completion")
        require(not choice["message"].get("refusal"), "Provider refusal")
        parsed = strict_json(choice["message"]["content"])
        require(set(parsed) == {"judgments"}, "Invalid completion object")
        answers = index_records(parsed["judgments"], "id")
        require(set(answers) <= set(ids), "Unknown candidate answers")
        for answer in answers.values():
            require(set(answer) == {"id", "label", "reason"}, "Invalid judgment fields")
            require(answer["label"] in LABELS, "Invalid label")
            require(isinstance(answer["reason"], str) and len(answer["reason"]) <= 80, "Invalid reason")
        items = [item for item in items if item["candidate_id"] in answers]
        normalized = [{"label": answers[item["candidate_id"]]["label"],
                       "reason": answers[item["candidate_id"]]["reason"],
                       "probabilities": None, "confidence": None} for item in items]
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
            probability_sum = sum(probabilities.values())
            require(math.isclose(probability_sum, 1, rel_tol=0, abs_tol=PROBABILITY_SUM_TOLERANCE + 1e-12),
                    "Probabilities do not sum to one within rounding tolerance")
            require(probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6, "Choice is not maximal")
            normalized.append({"label": answer["choice"], "reason": None,
                               "probabilities": probabilities, "confidence": answer["confidence"],
                               "probability_sum": probability_sum,
                               "probability_rounding_warning": not math.isclose(probability_sum, 1, abs_tol=1e-6)})
    return [{"pair_id": item["pair_id"], "candidate_id": item["candidate_id"], **answer}
            for item, answer in zip(items, normalized, strict=True)]


def validate_relation_result(result: object, labels: tuple[str, ...]) -> dict:
    require(isinstance(result, dict) and set(result) == {"label", "reason"}, "Invalid judgment result")
    require(result["label"] in labels, "Invalid judgment label")
    require(isinstance(result["reason"], str) and len(result["reason"]) <= 80, "Invalid judgment reason")
    return {
        "label": result["label"],
        "reason": result["reason"],
        "probabilities": None,
        "confidence": None,
    }


def parse_jev_choice(answer: object, labels: tuple[str, ...]) -> dict:
    require(isinstance(answer, dict), "Invalid Choice answer")
    require(answer.get("type") == "choice" and answer.get("choice") in labels, "Invalid Choice answer")
    probabilities = answer.get("probabilities")
    require(isinstance(probabilities, dict) and set(probabilities) == set(labels), "Incomplete probability distribution")
    confidence = answer.get("confidence")
    values = [*probabilities.values(), confidence]
    require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in values),
            "Invalid probability or confidence")
    probability_sum = sum(probabilities.values())
    tolerance = max(PROBABILITY_SUM_TOLERANCE, 0.005 * len(labels))
    require(math.isclose(probability_sum, 1, rel_tol=0, abs_tol=tolerance + 1e-12),
            "Probabilities do not sum to one within rounding tolerance")
    require(probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6, "Choice is not maximal")
    return {
        "label": answer["choice"],
        "reason": None,
        "probabilities": probabilities,
        "confidence": confidence,
        "probability_sum": probability_sum,
        "probability_rounding_warning": not math.isclose(probability_sum, 1, abs_tol=1e-6),
    }


def parse_relation_answers(provider: str, body: dict, batch: dict) -> list:
    require(isinstance(body.get("model"), str) and bool(body["model"]), "Missing served model")
    items = batch["candidates"]
    ids = [item["candidate_id"] for item in items]
    normalized = []
    if provider == "gpt":
        require(len(body["choices"]) == 1, "Expected one completion")
        choice = body["choices"][0]
        require(choice["finish_reason"] == "stop", "Incomplete completion")
        require(not choice["message"].get("refusal"), "Provider refusal")
        parsed = strict_json(choice["message"]["content"])
        require(set(parsed) == {"judgments"}, "Invalid completion object")
        answers = index_records(parsed["judgments"], "id")
        require(set(answers) <= set(ids), "Unknown candidate answers")
        items = [item for item in items if item["candidate_id"] in answers]
        for item in items:
            answer = answers[item["candidate_id"]]
            require(set(answer) == {"id", "identity", "relations"}, "Invalid judgment fields")
            relations = answer["relations"]
            require(isinstance(relations, dict) and set(relations) == set(RELATION_HEADS),
                    "Invalid relation fields")
            normalized.append({
                "identity": validate_relation_result(answer["identity"], IDENTITY_LABELS),
                "relations": {
                    head: validate_relation_result(relations[head], RELATION_LABELS) for head in RELATION_HEADS
                },
            })
    else:
        require(provider == "jev", "Unknown provider")
        answers = body["answers"]
        expected = {
            f"candidate_{index}_{head}"
            for index in range(len(ids))
            for head in ("identity", *RELATION_HEADS)
        }
        require(set(answers) == expected, "Missing or unknown question answers")
        for index in range(len(ids)):
            normalized.append({
                "identity": parse_jev_choice(answers[f"candidate_{index}_identity"], IDENTITY_LABELS),
                "relations": {
                    head: parse_jev_choice(answers[f"candidate_{index}_{head}"], RELATION_LABELS)
                    for head in RELATION_HEADS
                },
            })
    return [
        {"pair_id": item["pair_id"], "candidate_id": item["candidate_id"], **answer}
        for item, answer in zip(items, normalized, strict=True)
    ]


def resolve_basket_judgment(answer: dict) -> dict:
    identity = answer["identity"]["label"]
    require(identity in IDENTITY_LABELS, "Invalid identity label")
    relations = {head: answer["relations"][head]["label"] for head in RELATION_HEADS}
    require(all(label in RELATION_LABELS for label in relations.values()), "Invalid relation label")
    yes = {head for head, label in relations.items() if label == "yes"}
    raw_conflict = len(yes) > 1
    result = {
        "resolver_version": RESOLVER_VERSION,
        "identity_action": "continue_uncertain" if identity == "uncertain" else "continue",
        "basket_label": None,
        "resolution_status": None,
        "resolution": None,
        "conflict": raw_conflict,
    }
    if identity in {"duplicate", "variant", "redundant"}:
        result.update(
            identity_action="filter",
            basket_label="skip",
            resolution_status="filtered",
            resolution=f"identity_filtered_{identity}",
        )
        return result
    if raw_conflict:
        result.update(
            basket_label="conflict",
            resolution_status="conflict",
            resolution="conflicting_positive_relations",
        )
        return result
    labels = tuple(relations[head] for head in RELATION_HEADS)
    if labels == ("yes", "no", "no"):
        result.update(basket_label="positive", resolution_status="resolved", resolution="co_purchase_positive")
    elif labels == ("no", "yes", "no"):
        result.update(basket_label="hard_negative", resolution_status="resolved", resolution="alternative_only")
    elif labels == ("no", "no", "yes"):
        result.update(basket_label="hard_negative", resolution_status="resolved", resolution="incompatible_only")
    elif "uncertain" in labels:
        result.update(basket_label="skip", resolution_status="uncertain", resolution="uncertain_skip")
    else:
        result.update(basket_label="skip", resolution_status="resolved", resolution="unrelated_skip")
    require(result["basket_label"] in BASKET_LABELS, "Invalid basket label")
    return result


def extraction_status(answers: list, expected_count: int) -> tuple[str, str | None]:
    if len(answers) == expected_count:
        return "ok", None
    return ("partial" if answers else "error"), "missing_candidate_answers"


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


def run_batch(client: httpx.Client, provider: str, shop: str, batch: dict, payload: dict, key: str,
              *, initial_answers: list | None = None, on_attempt=None) -> dict:
    started = time.perf_counter()
    record = {
        "record_type": "batch", "provider": provider, "shop": shop,
        "batch_id": batch["batch_id"], "direction": batch["direction"], "anchor_id": batch["anchor_id"],
        "pair_ids": [item["pair_id"] for item in batch["candidates"]], "batch_size": len(batch["candidates"]),
        "requested_model": payload["model"], "served_model": None, "request": payload,
        "status": "error", "error": None, "answers": [], "raw_response": None, "attempts": [],
    }
    collected = index_records(initial_answers or [], "pair_id")
    require(set(collected) <= set(record["pair_ids"]), "Unexpected initial answers")
    if len(collected) == record["batch_size"]:
        record.update(status="ok", answers=list(collected.values()), retries=0, wall_seconds=0.0)
        return record
    for number in range(MAX_RETRIES + 1):
        record["raw_response"] = None
        attempt = {"number": number + 1, "http_status": None, "request_id": None, "served_model": None,
                   "usage": normalize_usage(provider, None), "raw_usage": None, "error": None,
                   "retry_delay_seconds": None, "raw_response": None, "answers": []}
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
                    attempt["raw_response"] = body
                attempt["served_model"] = body.get("model")
                attempt["raw_usage"] = body.get("usage")
                attempt["usage"] = normalize_usage(provider, attempt["raw_usage"])
                if status == 200:
                    attempt["answers"] = parse_answers(provider, body, batch)
                    _, attempt["error"] = extraction_status(attempt["answers"], record["batch_size"])
                    for answer in attempt["answers"]:
                        collected.setdefault(answer["pair_id"], answer)
                    retry = provider == "gpt" and len(collected) < record["batch_size"]
            except (ValueError, KeyError, TypeError, AttributeError, IndexError):
                attempt["error"] = "invalid_response"
            if status != 200:
                attempt["error"] = f"http_{status}"
                retry = status in (408, 429) or 500 <= status < 600
        record["attempts"].append(attempt)
        if attempt["served_model"] is not None:
            record["served_model"] = attempt["served_model"]
        record["answers"] = [collected[identity] for identity in record["pair_ids"] if identity in collected]
        record["status"], missing_error = extraction_status(record["answers"], record["batch_size"])
        record["error"] = (attempt["error"] or missing_error) if missing_error else None
        delay = retry_delay(header, number) if retry and number < MAX_RETRIES else None
        if retry and number < MAX_RETRIES and delay is None:
            attempt["retry_not_scheduled"] = "retry_after_exceeds_budget"
        attempt["retry_delay_seconds"] = delay
        if on_attempt is not None:
            checkpoint = copy.deepcopy(record)
            checkpoint.update(answers=attempt["answers"], attempts=[attempt], raw_response=attempt["raw_response"],
                              retries=0, wall_seconds=attempt["latency_seconds"])
            checkpoint["status"], missing_error = extraction_status(attempt["answers"], record["batch_size"])
            checkpoint["error"] = attempt["error"] or missing_error
            on_attempt(checkpoint)
        if delay is None:
            break
        time.sleep(delay)
    record["retries"] = len(record["attempts"]) - 1
    record["wall_seconds"] = time.perf_counter() - started
    return record


def execute(manifest: dict, jobs: dict, output_dir: Path, manifest_digest: str, credentials: dict,
            *, transport: httpx.BaseTransport | None = None, run_metadata: dict | None = None) -> int:
    for provider in jobs:
        require(provider in ENDPOINTS, "Unknown execution provider")
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
              "max_retries": MAX_RETRIES, "max_retry_delay_seconds": MAX_RETRY_DELAY, **(run_metadata or {})})
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


def read_request_log(path: Path) -> tuple[list, bool]:
    lines = path.read_bytes().splitlines(keepends=True)
    records = []
    truncated = False
    for i, line in enumerate(lines):
        try:
            record = strict_json(line.decode("utf-8"))
        except ValueError:
            if i == len(lines) - 1 and not line.endswith(b"\n"):
                truncated = True
                break
            raise ValueError(f"Invalid request log at line {i + 1}") from None
        require(isinstance(record, dict), "Invalid log record")
        records.append(record)
    require(bool(records) and records[0].get("record_type") == "run_start", "Missing run_start record")
    return records, truncated


def attempt_cost(provider: str, usage: dict, pricing: dict | None = None) -> float | None:
    inputs = usage["input_tokens"]
    if inputs is None:
        return None
    rates = (PRICING if pricing is None else pricing)[provider]
    if provider == "jev":
        return inputs * rates["input"] / 1_000_000
    cached, outputs = usage["cached_input_tokens"], usage["output_tokens"]
    if cached is None or outputs is None:
        return None
    require(cached <= inputs, "Cached token count exceeds input count")
    # Completion tokens already include reasoning tokens; never charge them twice.
    return ((inputs - cached) * rates["input"] + cached * rates["cached_input"] + outputs * rates["output"]) / 1_000_000


def reextract_record(record: dict, batch: dict) -> dict:
    if (record["status"] != "error" or record["answers"] or record["error"] != "invalid_response" or record["raw_response"] is None
            or record["attempts"][-1]["http_status"] != 200):
        return record
    try:
        normalize_usage(record["provider"], record["raw_response"].get("usage"))
        answers = parse_answers(record["provider"], record["raw_response"], batch)
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        return record
    result = copy.deepcopy(record)
    result["recorded_status"], result["recorded_error"] = record["status"], record["error"]
    result["answers"] = answers
    result["status"], result["error"] = extraction_status(answers, record["batch_size"])
    result["attempts"][-1]["recorded_error"] = record["attempts"][-1]["error"]
    result["attempts"][-1]["error"] = result["error"]
    return result


def baseline_spec(manifest: dict) -> dict:
    requests = []
    for shop, data in sorted(manifest["shops"].items()):
        for batch in sorted(data["batches"], key=lambda item: item["batch_id"]):
            identity = {key: batch[key] for key in ("batch_id", "direction", "anchor_id")}
            identity["candidates"] = [{key: item[key] for key in ("pair_id", "candidate_id")}
                                      for item in batch["candidates"]]
            requests.append({"shop": shop, "batch": identity,
                             "request": render_request("gpt", manifest["models"]["gpt"], data, batch)})
    return {"schema_version": "gpt-input-v1", "requests": requests}


def immutable_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            try:
                os.link(temporary, path)  # Atomic publication without replacing an existing file.
            except FileExistsError:
                require(path.read_bytes() == content, f"Refusing to replace immutable baseline file: {path}")
        finally:
            temporary.unlink()


def check_baseline_record(record: dict, expected: dict) -> dict:
    key = (record["shop"], record["batch_id"])
    require(record["provider"] == "gpt" and key in expected, "Unknown GPT baseline batch")
    job = expected[key]
    batch = job["batch"]
    require(record["request"] == job["request"], "GPT baseline request mismatch")
    require(record["requested_model"] == job["request"]["model"], "GPT baseline model mismatch")
    require(record["pair_ids"] == [item["pair_id"] for item in batch["candidates"]], "GPT baseline pair mismatch")
    require(record["anchor_id"] == batch["anchor_id"] and record["direction"] == batch["direction"], "Baseline direction mismatch")
    require(record["batch_size"] == len(batch["candidates"]) and bool(record["attempts"]), "Invalid baseline attempt")
    return job


def baseline_state(directory: Path, spec: dict, pricing: dict | None = None) -> dict:
    fingerprint = sha256(canonical_json(spec))
    pricing = copy.deepcopy({key: PRICING[key] for key in ("as_of", "currency", "unit", "gpt")}) if pricing is None else pricing
    expected = {(job["shop"], job["batch"]["batch_id"]): job for job in spec["requests"]}
    all_ids = [item["pair_id"] for job in spec["requests"] for item in job["batch"]["candidates"]]
    require(len(all_ids) == len(set(all_ids)), "Duplicate baseline pair identity")
    answers, sources = {}, []
    attempts = {"historical": [], "repair": []}
    wall_times = {"historical": [], "repair": []}
    finished = {"historical": True, "repair": True}
    paths = ([directory / "imported.jsonl"] if (directory / "imported.jsonl").exists() else [])
    paths += sorted((directory / "repairs").glob("*.jsonl"))
    for path in paths:
        kind = "historical" if path.name == "imported.jsonl" else "repair"
        relative = str(path.relative_to(directory))
        records, truncated = read_request_log(path)
        if kind == "repair":
            require(records[0]["baseline_fingerprint"] == fingerprint, "Repair log fingerprint mismatch")
        finished[kind] &= not truncated and records[-1]["record_type"] == "run_end"
        sources.append({"path": relative, "sha256": sha256(path.read_bytes()), "kind": kind,
                        "truncated_final_line": truncated, "run_settings": records[0]})
        seen = set()
        for index, record in enumerate(records):
            if record["record_type"] == "group_end" and record["provider"] == "gpt":
                wall_times[kind].append({"source_log": relative, "shop": record["shop"], "wall_seconds": record["wall_seconds"]})
            if record["record_type"] != "batch" or record["provider"] != "gpt":
                continue
            job = check_baseline_record(record, expected)
            identity = record["baseline_attempt_id"] if kind == "repair" else record["batch_id"]
            require(identity not in seen, "Duplicate baseline history record")
            seen.add(identity)
            attempts[kind].extend(record["attempts"])
            for number, attempt in enumerate(record["attempts"]):
                body = attempt.get("raw_response") if "raw_response" in attempt else (
                    record["raw_response"] if number == len(record["attempts"]) - 1 else None)
                if attempt["http_status"] != 200 or body is None:
                    continue
                try:
                    normalize_usage("gpt", body.get("usage"))
                    parsed = parse_answers("gpt", body, job["batch"])
                except (ValueError, KeyError, TypeError, AttributeError, IndexError):
                    continue
                for answer in parsed:
                    answers.setdefault(answer["pair_id"], {
                        **answer, "provider": "gpt", "status": "ok", "shop": job["shop"],
                        "direction": job["batch"]["direction"], "anchor_id": job["batch"]["anchor_id"],
                        "batch_id": job["batch"]["batch_id"], "served_model": body["model"],
                        "source_log": relative, "source_record": index, "source_attempt": number,
                    })
    accounting = {}
    for kind, calls in attempts.items():
        usage = {}
        for field in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"):
            values = [call["usage"][field] for call in calls if call["usage"][field] is not None]
            require(all(type(value) is int and value >= 0 for value in values), "Invalid baseline token usage")
            subtotal = sum(values) if values else None
            usage[field] = {"reported_subtotal": subtotal, "missing_attempts": len(calls) - len(values),
                            "total": subtotal if finished[kind] and len(values) == len(calls) else None}
        costs = [attempt_cost("gpt", call["usage"], pricing) for call in calls]
        priced = [cost for cost in costs if cost is not None]
        subtotal = math.fsum(priced) if priced else None
        accounting[kind] = {
            "http_attempts": len(calls), "sessions_complete": finished[kind], "usage": usage,
            "group_wall_times": wall_times[kind], "priced_attempts_subtotal_usd": subtotal,
            "cost_estimate_usd": subtotal if finished[kind] and len(priced) == len(calls) else None,
        }
    missing = [identity for identity in all_ids if identity not in answers]
    return {"schema_version": "gpt-baseline-v1", "fingerprint": fingerprint, "complete": not missing,
            "classifications": [answers[identity] for identity in all_ids if identity in answers],
            "missing_pair_ids": missing, "sources": sources, "accounting": accounting, "pricing": pricing}


def load_completed_baseline(manifest: dict, root: Path) -> dict:
    spec = baseline_spec(manifest)
    directory = root / sha256(canonical_json(spec))
    require((directory / "baseline.json").is_file(),
            f"No completed matching GPT baseline at {directory}; prepare it explicitly before running experiments")
    require(strict_json((directory / "spec.json").read_text()) == spec, "GPT baseline spec mismatch")
    saved = strict_json((directory / "baseline.json").read_text())
    state = baseline_state(directory, spec, saved["pricing"])
    require(state["complete"] and saved == state, "Completed GPT baseline or its response history changed")
    return saved


def prepare_gpt_baseline(manifest: dict, root: Path, *, import_run: Path | None = None,
                         paid: bool = False, transport: httpx.BaseTransport | None = None) -> tuple[Path, dict]:
    spec = baseline_spec(manifest)
    fingerprint = sha256(canonical_json(spec))
    directory = root / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process is preparing this GPT baseline; no requests started") from None
        immutable_bytes(directory / "spec.json", canonical_json(spec))
        if import_run is not None:
            source = import_run / "requests.jsonl"
            records, _ = read_request_log(source)
            expected = {(job["shop"], job["batch"]["batch_id"]): job for job in spec["requests"]}
            imported = [record for record in records if record["record_type"] == "batch" and record["provider"] == "gpt"]
            require(bool(imported), "Import contains no GPT batches")
            for record in imported:
                check_baseline_record(record, expected)
            require((directory / "imported.jsonl").exists() or not any((directory / "repairs").glob("*.jsonl")),
                    "Cannot introduce a historical import after baseline repair has started")
            immutable_bytes(directory / "imported.jsonl", source.read_bytes())
        if (directory / "baseline.json").exists():
            return directory, load_completed_baseline(manifest, root)
        state = baseline_state(directory, spec)
        if paid and not state["complete"]:
            key = os.environ.get("OPENAI_API_KEY")
            require(bool(key), "Missing credential: OPENAI_API_KEY; no requests started")
            missing = set(state["missing_pair_ids"])
            jobs = [job for job in spec["requests"] if any(item["pair_id"] in missing for item in job["batch"]["candidates"])]
            seed = {answer["pair_id"]: answer for answer in state["classifications"]}
            repairs = directory / "repairs"
            repairs.mkdir(exist_ok=True)
            sequence = len(list(repairs.glob("*.jsonl")))
            name = f"{sequence:06d}-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
            with (repairs / f"{name}.jsonl").open("x", encoding="utf-8") as handle:
                writer_lock = threading.Lock()

                def emit(record):
                    with writer_lock:
                        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())

                def checkpoint(record):
                    record["baseline_attempt_id"] = uuid.uuid4().hex
                    emit(record)

                emit({"record_type": "run_start", "baseline_fingerprint": fingerprint,
                      "started_at": datetime.now(timezone.utc).isoformat(), "concurrency": manifest["concurrency"],
                      "script_sha256": sha256(Path(__file__).read_bytes()), "max_retries": MAX_RETRIES,
                      "timeout_seconds": TIMEOUT_SECONDS})
                print(f"GPT baseline completion: {len(jobs)} incomplete batches, {len(missing)} missing pairs; "
                      f"at most {len(jobs) * (MAX_RETRIES + 1)} HTTP attempts this invocation.", flush=True)
                for shop in sorted({job["shop"] for job in jobs}):
                    work = [job for job in jobs if job["shop"] == shop]
                    emit({"record_type": "group_start", "provider": "gpt", "shop": shop, "batches": len(work)})
                    started = time.perf_counter()
                    with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=False, transport=transport) as client:
                        with ThreadPoolExecutor(max_workers=manifest["concurrency"]) as pool:
                            futures = [pool.submit(
                                run_batch, client, "gpt", shop, job["batch"], job["request"], key,
                                initial_answers=[seed[item["pair_id"]] for item in job["batch"]["candidates"]
                                                 if item["pair_id"] in seed], on_attempt=checkpoint,
                            ) for job in work]
                            for future in as_completed(futures):
                                future.result()
                    emit({"record_type": "group_end", "provider": "gpt", "shop": shop,
                          "wall_seconds": time.perf_counter() - started})
                emit({"record_type": "run_end"})
            state = baseline_state(directory, spec)
        if state["complete"]:
            immutable_bytes(directory / "baseline.json", canonical_json(state))
        return directory, state


def run_jev_experiment(manifest: dict, baseline_root: Path, output_dir: Path | None = None,
                       *, experiments_root: Path = Path("results/experiments"),
                       transport: httpx.BaseTransport | None = None) -> tuple[Path, int]:
    baseline = load_completed_baseline(manifest, baseline_root)
    key = os.environ.get("TYPESAFE_API_KEY")
    require(bool(key), "Missing credential: TYPESAFE_API_KEY; no requests started")
    jobs = {"jev": {shop: [(batch, render_request("jev", manifest["models"]["jev"], data, batch))
                          for batch in data["batches"]] for shop, data in manifest["shops"].items()}}
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:12]
    directory = output_dir if output_dir is not None else experiments_root / identifier
    directory.mkdir(parents=True, exist_ok=False)
    manifest_digest = write_manifest(directory / "manifest.json", manifest)
    baseline_bytes = canonical_json(baseline)
    immutable_bytes(directory / "baseline.json", baseline_bytes)
    experiment = {
        "schema_version": "jev-experiment-v1", "id": directory.name,
        "manifest_sha256": manifest_digest, "baseline_fingerprint": baseline["fingerprint"],
        "baseline_sha256": sha256(baseline_bytes), "jev_prompt_version": JEV_PROMPT_VERSION,
        "jev_criteria": JEV_CRITERIA, "jev_instructions": JEV_INSTRUCTIONS, "pricing": PRICING,
        "jev_request_hashes": {batch["batch_id"]: sha256(canonical_json(payload))
                               for work in jobs["jev"].values() for batch, payload in work},
    }
    experiment_bytes = canonical_json(experiment)
    immutable_bytes(directory / "experiment.json", experiment_bytes)
    print(f"Jev experiment: {directory}; reusing {len(baseline['classifications'])} GPT labels. "
          "No GPT requests will be made.", flush=True)
    failures = execute(manifest, jobs, directory, manifest_digest, {"jev": key}, transport=transport,
                       run_metadata={"experiment_sha256": sha256(experiment_bytes),
                                     "baseline_fingerprint": baseline["fingerprint"]})
    write_reports(directory / "manifest.json", directory)
    return directory, failures


def experiment_snapshot(directory: Path, raw_manifest: bytes, manifest: dict, start: dict) -> tuple[dict, dict]:
    raw_experiment = (directory / "experiment.json").read_bytes()
    raw_baseline = (directory / "baseline.json").read_bytes()
    require(start.get("experiment_sha256") == sha256(raw_experiment), "Experiment snapshot digest mismatch")
    experiment = strict_json(raw_experiment.decode("utf-8"))
    baseline = strict_json(raw_baseline.decode("utf-8"))
    require(experiment["schema_version"] == "jev-experiment-v1", "Unsupported experiment schema")
    require(experiment["manifest_sha256"] == sha256(raw_manifest), "Experiment manifest mismatch")
    require((directory / "manifest.json").read_bytes() == raw_manifest, "Experiment must use its saved manifest")
    require(experiment["baseline_sha256"] == sha256(raw_baseline), "Baseline snapshot digest mismatch")
    fingerprint = sha256(canonical_json(baseline_spec(manifest)))
    require(baseline["schema_version"] == "gpt-baseline-v1" and baseline["complete"], "Incomplete GPT snapshot")
    require(start["baseline_fingerprint"] == experiment["baseline_fingerprint"] == baseline["fingerprint"] == fingerprint,
            "Experiment baseline fingerprint mismatch")
    expected = {item["pair_id"]: (shop, batch, item) for shop, data in manifest["shops"].items()
                for batch in data["batches"] for item in batch["candidates"]}
    answers = index_records(baseline["classifications"], "pair_id")
    require(set(answers) == set(expected), "Baseline snapshot answer coverage mismatch")
    for identity, answer in answers.items():
        shop, batch, item = expected[identity]
        require(answer["shop"] == shop and answer["candidate_id"] == item["candidate_id"]
                and answer["anchor_id"] == batch["anchor_id"] and answer["direction"] == batch["direction"],
                "Baseline snapshot pair mismatch")
        require(answer["status"] == "ok" and answer["label"] in LABELS, "Invalid cached classification")
    require(set(experiment["jev_request_hashes"]) == {batch["batch_id"] for data in manifest["shops"].values()
                                                    for batch in data["batches"]}, "Experiment request coverage mismatch")
    return experiment, baseline


def summarize_group(provider: str, records: list, expected: list, end: dict | None, concurrency: int,
                    pricing: dict | None = None) -> dict:
    attempts = [attempt for record in records for attempt in record["attempts"]]
    complete = end is not None and len(records) == len(expected)
    wall = end["wall_seconds"] if complete else None
    if wall is not None:
        require(type(wall) in (int, float) and math.isfinite(wall) and wall >= 0, "Invalid group duration")
    latencies = sorted(attempt["latency_seconds"] for attempt in attempts)
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in latencies), "Invalid request latency")
    usage = {}
    for field in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"):
        reported = [attempt["usage"][field] for attempt in attempts if attempt["usage"][field] is not None]
        require(all(type(value) is int and value >= 0 for value in reported), "Invalid recorded usage")
        subtotal = sum(reported) if reported else None
        usage[field] = {
            "reported_subtotal": subtotal,
            "total": subtotal if complete and len(reported) == len(attempts) else None,
            "reported_attempts": len(reported), "missing_attempts": len(attempts) - len(reported),
        }
    costs = [attempt_cost(provider, attempt["usage"], pricing) for attempt in attempts]
    priced = [cost for cost in costs if cost is not None]
    subtotal = math.fsum(priced) if priced else None
    cost_complete = complete and bool(attempts) and len(priced) == len(attempts)
    successes = [answer for record in records if record["status"] in ("ok", "partial") for answer in record["answers"]]
    expected_pairs = sum(len(batch["candidates"]) for batch in expected)
    recorded_pairs = sum(record["batch_size"] for record in records)
    return {
        "complete": complete, "wall_seconds": wall, "concurrency": concurrency,
        "expected_batches": len(expected), "batch_requests": len(records), "http_attempts": len(attempts),
        "retries": sum(len(record["attempts"]) - 1 for record in records),
        "failed_attempts": sum(attempt["error"] is not None for attempt in attempts),
        "recorded_failed_attempts": sum(attempt.get("recorded_error", attempt["error"]) is not None for attempt in attempts),
        "failed_batches": sum(record["status"] == "error" for record in records),
        "partial_batches": sum(record["status"] == "partial" for record in records),
        "reextracted_batches": sum("recorded_status" in record for record in records),
        "probability_rounding_warnings": sum(answer.get("probability_rounding_warning", False) for answer in successes),
        "successful_pairs": len(successes), "failed_pairs": recorded_pairs - len(successes),
        "missing_pairs": expected_pairs - recorded_pairs,
        "pairs_per_second": len(successes) / wall if wall else None,
        "served_models": sorted({attempt["served_model"] for attempt in attempts
                                 if isinstance(attempt["served_model"], str)}),
        "batch_sizes": dict(sorted(Counter(str(record["batch_size"]) for record in records).items())),
        "label_counts": {label: sum(answer["label"] == label for answer in successes) for label in LABELS},
        "request_latency_seconds": {
            "count": len(latencies), "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": statistics.median(latencies) if latencies else None,
            "p95": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
        },
        "usage": usage,
        "cost_estimate": {
            "total_usd": subtotal if cost_complete else None, "priced_attempts_subtotal_usd": subtotal,
            "priced_attempts": len(priced), "unpriced_attempts": len(costs) - len(priced),
            "complete_usage_coverage": cost_complete,
        },
    }


def write_csv(path: Path, rows: list, columns: list) -> None:
    def cell(value):
        # Keep catalog text from becoming executable spreadsheet formulas.
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: cell(value) for key, value in row.items()} for row in rows)


def write_reports(manifest_path: Path, output_dir: Path) -> dict:
    raw_manifest = manifest_path.read_bytes()
    manifest = strict_json(raw_manifest.decode("utf-8"))
    records, truncated = read_request_log(output_dir / "requests.jsonl")
    start = records[0]
    require(start["manifest_sha256"] == sha256(raw_manifest), "Run/manifest digest mismatch")
    require(start["concurrency"] == manifest["concurrency"], "Run/manifest concurrency mismatch")
    experiment, baseline = None, None
    if "experiment_sha256" in start or (output_dir / "experiment.json").exists():
        experiment, baseline = experiment_snapshot(output_dir, raw_manifest, manifest, start)
    providers = ("jev",) if experiment else manifest["models"]
    pricing = experiment["pricing"] if experiment else PRICING
    expected = {(provider, shop, batch["batch_id"]): batch for provider in providers
                for shop, data in manifest["shops"].items() for batch in data["batches"]}
    groups = {(provider, shop) for provider, shop, _ in expected}
    observed, starts, ends = {}, {}, {}
    finished = False
    for record in records[1:]:
        require(not finished, "Unexpected records after run_end")
        kind = record["record_type"]
        if kind == "run_end":
            finished = True
            continue
        group = (record["provider"], record["shop"])
        require(group in groups, "Unknown provider/shop in request log")
        if kind in ("group_start", "group_end"):
            if kind == "group_start":
                require(record["requested_model"] == manifest["models"][group[0]]["model"], "Group model mismatch")
                require(record["batches"] == len(manifest["shops"][group[1]]["batches"]), "Group batch-count mismatch")
            target = starts if kind == "group_start" else ends
            require(group not in target, "Duplicate group record")
            target[group] = record
        else:
            require(kind == "batch", "Unknown request-log record type")
            key = (*group, record["batch_id"])
            require(key in expected and key not in observed, "Unknown or duplicate batch record")
            batch = expected[key]
            settings = manifest["models"][group[0]]
            require(record["requested_model"] == record["request"]["model"] == settings["model"], "Batch model mismatch")
            if experiment:
                require(sha256(canonical_json(record["request"])) == experiment["jev_request_hashes"][record["batch_id"]],
                        "Jev request differs from experiment snapshot")
            if group[0] == "gpt":
                require(record["request"]["reasoning_effort"] == settings["reasoning_effort"], "Reasoning setting mismatch")
                require("temperature" not in record["request"] and "max_completion_tokens" not in record["request"],
                        "Unexpected generation override")
            require(record["pair_ids"] == [item["pair_id"] for item in batch["candidates"]], "Batch pair mismatch")
            require(record["batch_size"] == len(batch["candidates"]), "Batch size mismatch")
            require(record["status"] in ("ok", "partial", "error") and bool(record["attempts"]), "Invalid batch status/attempts")
            answers = index_records(record["answers"], "pair_id")
            if record["status"] == "partial":
                require(bool(answers) and set(answers) < set(record["pair_ids"]), "Invalid partial answer coverage")
            else:
                required_ids = set(record["pair_ids"]) if record["status"] == "ok" else set()
                require(set(answers) == required_ids, "Invalid batch answer coverage")
            require(all(answer["label"] in LABELS for answer in answers.values()), "Invalid recorded label")
            observed[key] = reextract_record(record, batch)

    classifications = {provider: {} for provider in manifest["models"]}
    summary = {
        "schema_version": "jev-sbs-report-v2" if experiment else "jev-sbs-report-v1", "manifest_sha256": sha256(raw_manifest),
        "extraction": {"version": EXTRACTION_VERSION, "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
                       "request_log_sha256": sha256((output_dir / "requests.jsonl").read_bytes())},
        "models": manifest["models"], "pricing": pricing, "run_settings": start, "truncated_final_line": truncated,
        "limitations": [*manifest["limitations"],
                        "Interrupted runs can omit in-flight attempts; usage subtotals cover recorded attempts only."],
        "groups": {}, "agreement": {},
    }
    if baseline is not None:
        classifications["gpt"] = {answer["pair_id"]: {**answer, "cached": True, "error": None,
                                                     "baseline_fingerprint": baseline["fingerprint"]}
                                  for answer in baseline["classifications"]}
        summary["experiment"] = experiment
        summary["gpt_baseline"] = {
            "fingerprint": baseline["fingerprint"], "cached": True, "classifications": len(classifications["gpt"]),
            "accounting": baseline["accounting"], "pricing": baseline["pricing"],
            "label_counts": {shop: {label: sum(answer["shop"] == shop and answer["label"] == label
                                              for answer in baseline["classifications"]) for label in LABELS}
                             for shop in manifest["shops"]},
            "note": "Historical measurements, not inference performed during this Jev experiment.",
        }
    for provider, shop in sorted(groups):
        data = manifest["shops"][shop]
        group_records = [observed[provider, shop, batch["batch_id"]] for batch in data["batches"]
                         if (provider, shop, batch["batch_id"]) in observed]
        end = ends.get((provider, shop)) if (provider, shop) in starts else None
        summary["groups"][f"{provider}/{shop}"] = summarize_group(
            provider, group_records, data["batches"], end, manifest["concurrency"], pricing)
        for batch in data["batches"]:
            record = observed.get((provider, shop, batch["batch_id"]))
            answers = {answer["pair_id"]: answer for answer in record["answers"]} if record else {}
            for item in batch["candidates"]:
                answer = answers.get(item["pair_id"], {})
                classifications[provider][item["pair_id"]] = {
                    "provider": provider, "shop": shop, "direction": batch["direction"],
                    "pair_id": item["pair_id"], "batch_id": batch["batch_id"],
                    "anchor_id": batch["anchor_id"], "candidate_id": item["candidate_id"],
                    "status": "ok" if answer else ("error" if record else "missing"),
                    "error": None if answer else ("missing_candidate_answer" if record and record["status"] == "partial"
                                                  else record["error"] if record else "batch_not_recorded"),
                    "recorded_batch_status": record.get("recorded_status", record["status"]) if record else None,
                    "served_model": record["served_model"] if record else None,
                    **{key: answer.get(key) for key in ("label", "reason", "probabilities", "confidence",
                                                      "probability_sum", "probability_rounding_warning")},
                }

    paired = []
    for identity, gpt in classifications["gpt"].items():
        jev = classifications["jev"][identity]
        data = manifest["shops"][gpt["shop"]]
        anchor, candidate = data["products"][gpt["anchor_id"]], data["products"][gpt["candidate_id"]]
        success = gpt["status"] == jev["status"] == "ok"
        paired.append({
            **{key: gpt[key] for key in ("pair_id", "shop", "direction", "anchor_id", "candidate_id")},
            "anchor_name": anchor["name"], "anchor_text": anchor["text"],
            "candidate_name": candidate["name"], "candidate_category": candidate["category"],
            "candidate_description": candidate["description"],
            "gpt_status": gpt["status"], "gpt_label": gpt["label"], "gpt_reason": gpt["reason"],
            "jev_status": jev["status"], "jev_label": jev["label"], "jev_confidence": jev["confidence"],
            "jev_probabilities": json.dumps(jev["probabilities"], sort_keys=True) if jev["probabilities"] else None,
            "paired_success": success, "agreement": gpt["label"] == jev["label"] if success else None,
        })
    for shop in manifest["shops"]:
        eligible = [row for row in paired if row["shop"] == shop and row["paired_success"]]
        agrees = sum(row["agreement"] for row in eligible)
        summary["agreement"][shop] = {
            "paired_success": len(eligible), "agreements": agrees, "disagreements": len(eligible) - agrees,
            "unpaired": sum(row["shop"] == shop and not row["paired_success"] for row in paired),
            "agreement_rate": agrees / len(eligible) if eligible else None,
            "matrix_gpt_rows_jev_columns": {left: {right: sum(row["gpt_label"] == left and row["jev_label"] == right
                                                            for row in eligible) for right in LABELS} for left in LABELS},
        }
    summary["run_complete"] = finished and not truncated and all(group["complete"] for group in summary["groups"].values())
    summary["all_pairs_classified"] = all(row["paired_success"] for row in paired)
    if experiment:
        costs = [group["cost_estimate"]["total_usd"] for group in summary["groups"].values()]
        summary["new_inference"] = {
            "gpt_http_attempts": 0, "gpt_cost_usd": 0.0,
            "jev_http_attempts": sum(group["http_attempts"] for group in summary["groups"].values()),
            "jev_cost_estimate_usd": math.fsum(costs) if all(cost is not None for cost in costs) else None,
        }
    for provider, rows in classifications.items():
        (output_dir / f"{provider}_classifications.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows.values()), encoding="utf-8")
    columns = list(paired[0])
    write_csv(output_dir / "paired_results.csv", paired, columns)
    write_csv(output_dir / "disagreements.csv", [row for row in paired if row["agreement"] is False], columns)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, help="Frozen workload path; defaults to results/manifest.json or the experiment snapshot")
    parser.add_argument("--concurrency", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Make paid API requests (Boris runs this explicitly)")
    mode.add_argument("--report-only", action="store_true", help="Rebuild reports from the saved manifest and request log, offline")
    parser.add_argument("--output-dir", type=Path, help="New experiment directory for execution, or existing directory for --report-only")
    parser.add_argument("--prepare-baseline", action="store_true", help="Import/prepare the GPT baseline; add --execute for paid completion")
    parser.add_argument("--import-run", type=Path, help="Existing comparison run directory to seed the GPT baseline")
    parser.add_argument("--baseline-root", type=Path, default=Path("results/baselines"))
    args = parser.parse_args()
    if args.import_run and not args.prepare_baseline:
        parser.error("--import-run requires --prepare-baseline")
    if args.prepare_baseline and args.report_only:
        parser.error("--prepare-baseline cannot be combined with --report-only")
    if args.report_only:
        directory = args.output_dir or Path("results/run")
        manifest_path = args.manifest or (directory / "manifest.json" if (directory / "experiment.json").exists()
                                          else Path("results/manifest.json"))
        try:
            summary = write_reports(manifest_path, directory)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
            parser.exit(1, f"Reporting failed: {error}\n")
        print(f"Reports rebuilt offline in {directory}; run_complete={summary['run_complete']}, "
              f"all_pairs_classified={summary['all_pairs_classified']}")
        return
    try:
        manifest = build_manifest(args.data_dir, args.concurrency)
        manifest_path = args.manifest or Path("results/manifest.json")
        digest = write_manifest(manifest_path, manifest)
        if args.prepare_baseline:
            directory, baseline = prepare_gpt_baseline(
                manifest, args.baseline_root, import_run=args.import_run, paid=args.execute)
            print(f"GPT baseline: {directory}")
            print(f"Retained {len(baseline['classifications'])} labels; {len(baseline['missing_pair_ids'])} missing. "
                  f"Complete: {baseline['complete']}.")
            if not args.execute:
                print("No API requests made. Add --execute only when ready to complete missing GPT answers.")
            elif not baseline["complete"]:
                parser.exit(1, "GPT baseline remains incomplete after bounded attempts; saved progress is reusable.\n")
            return
        prepare_requests(manifest)  # Validate payload rendering during offline preflight.
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(f"Offline manifest: {manifest_path} (SHA256 {digest})")
    print(f"Models: {GPT_SETTINGS['model']} / jev-1.13.0; concurrency: {args.concurrency}")
    for shop, data in manifest["shops"].items():
        for direction in ("forward", "reverse"):
            batches = [batch for batch in data["batches"] if batch["direction"] == direction]
            sizes = [len(batch["candidates"]) for batch in batches]
            print(f"{shop} {direction}: {sum(sizes)} rows, {len(batches)} Jev requests, batch sizes {sizes}")
        print(f"  Source checks: {json.dumps(data['diagnostics'], sort_keys=True)}")
    print(f"Jev prompt: {JEV_PROMPT_VERSION}. GPT labels must come from a completed saved baseline.")
    if not args.execute:
        print("No API requests made. Use --execute only when ready for the paid run.")
        return
    try:
        directory, failures = run_jev_experiment(manifest, args.baseline_root, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(1, f"Execution failed: {error}\n")
    if failures:
        parser.exit(1, f"Run finished with {failures} incomplete batches; inspect requests.jsonl before any rerun.\n")
    print(f"All batches completed. Inspect {directory / 'summary.json'} and disagreements.csv.")


if __name__ == "__main__":
    main()
