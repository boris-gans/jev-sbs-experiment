import copy
import csv
import fcntl
import hashlib
import itertools
import json
import socket
import threading

import httpx

import pytest

import compare


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_blocks(path, blocks):
    path.write_text("\n".join(json.dumps(block) for block in blocks) + "\n", encoding="utf-8")


def refresh_context(root, shop):
    stage2 = root / shop / "stage2"
    context_path = stage2 / "classification_context.json"
    context = json.loads(context_path.read_text())
    objective = compare.parse_recommendation_objective((stage2 / "recommendation.yaml").read_bytes())
    stem = compare.PROMPTS[objective]
    paths = {
        "products": root / shop / "stage1/products_processed.json",
        "subset": stage2 / "subset_products.json",
        "candidates": stage2 / "candidates.json",
        "examples": stage2 / "classification_examples.txt",
        "system_prompt": stage2 / f"{stem}.system.txt",
        "user_prompt": stage2 / f"{stem}.user.txt",
    }
    context["inputs"] = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    context.pop("context_version", None)
    encoded = json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    context["context_version"] = "ctx-v1-" + hashlib.sha256(encoded).hexdigest()
    write_json(context_path, context)
    for direction in ("forward", "reverse"):
        path = stage2 / f"{direction}_judgments.json"
        blocks = [json.loads(line) for line in path.read_text().splitlines()]
        for block in blocks:
            block["prompt_version"] = context["context_version"]
        write_blocks(path, blocks)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail("Offline tests must never access the network")
    monkeypatch.setattr(socket.socket, "connect", deny)


@pytest.fixture
def catalogs(tmp_path):
    for shop, objective in compare.SHOPS.items():
        stage1, stage2 = tmp_path / shop / "stage1", tmp_path / shop / "stage2"
        stage1.mkdir(parents=True)
        stage2.mkdir()
        products = [{
            "product_id": str(i), "name": f"Item {i}",
            "category": "Synthetic café", "description": f"Evidence {i}",
        } for i in range(21)]
        subset = [{**item, "text": ". ".join(item[key] for key in ("name", "category", "description"))}
                  for item in products]
        retrieval, forward, reverse = [], [], []
        for product in products:
            anchor = product["product_id"]
            ids = [item["product_id"] for item in products if item["product_id"] != anchor]
            retrieval.append({"anchor_id": anchor, "candidates": [
                {"candidate_id": identity, "rank": rank, "source": "e5_topk"}
                for rank, identity in enumerate(ids, 1)
            ]})
            block = {"anchor_id": anchor, "prompt_version": "pending", "judgments": [
                {"candidate_id": identity, "label": "skip", "reason": "Historical label must not be used"}
                for identity in ids
            ]}
            forward.append(block)
            reverse.append({**block, "judgments": list(reversed(block["judgments"]))})
        write_json(stage1 / "products_processed.json", products)
        write_json(stage2 / "subset_products.json", subset)
        write_json(stage2 / "candidates.json", retrieval)
        write_blocks(stage2 / "forward_judgments.json", forward)
        write_blocks(stage2 / "reverse_judgments.json", reverse)
        stem = compare.PROMPTS[objective]
        for name, content in {
            f"{stem}.system.txt": "Exact policy with {examples}\nOUTPUT\nGPT-only output instructions\n",
            f"{stem}.user.txt": (
                "Base: {anchor_id} {anchor_text}\n{candidates_json}\n"
                "For each candidate, apply the ordered rules.\n"
                "Category guidance must survive.\nReturn a JSON object with judgments.\n"
            ),
            "classification_examples.txt": "Synthetic example\n",
            "recommendation.yaml": f"version: 1\nobjective: {objective}\n",
        }.items():
            (stage2 / name).write_text(content, encoding="utf-8")
        write_json(stage2 / "classification_context.json", {
            "schema_version": "classification-context-v1", "classifier_contract_version": "v1",
            "shop": shop, "recommendation_objective": objective, "llm": compare.GPT_SETTINGS,
            "retrieval": {"base_embedding_model": "intfloat/e5-large-v2", "top_k_candidates": 20},
        })
        refresh_context(tmp_path, shop)
    return tmp_path


def test_deterministic_manifest_and_evidence(catalogs):
    first = compare.build_manifest(catalogs)
    assert first == compare.build_manifest(catalogs)
    assert first["models"]["gpt"]["model"] == "gpt-5-nano-2025-08-07"
    assert first["models"]["jev"]["model"] == "jev-1.13.0"
    for shop, data in first["shops"].items():
        seen = set()
        for direction in ("forward", "reverse"):
            batches = [batch for batch in data["batches"] if batch["direction"] == direction]
            assert sum(len(batch["candidates"]) for batch in batches) == 200
            for batch in batches:
                assert 1 <= len(batch["candidates"]) <= 20
                anchor = data["products"][batch["anchor_id"]]
                assert anchor["text"] == f"Item {anchor['id']}. Synthetic café. Evidence {anchor['id']}"
                for candidate in batch["candidates"]:
                    identity = (shop, direction, anchor["id"], candidate["candidate_id"])
                    assert tuple(json.loads(candidate["pair_id"])) == identity
                    assert identity not in seen
                    seen.add(identity)
                    evidence = data["products"][candidate["candidate_id"]]
                    assert evidence["description"] == f"Evidence {candidate['candidate_id']}"
        assert len(seen) == 400
        assert data["policy"]["examples"] == "Synthetic example\n"


def test_recommendation_config_selects_objective(catalogs):
    shop = "furniture.co.uk"
    stage2 = catalogs / shop / "stage2"
    source = catalogs / "themeatboys.nl" / "stage2"
    for suffix in ("system.txt", "user.txt"):
        name = f"{compare.PROMPTS['complements']}.{suffix}"
        (stage2 / name).write_bytes((source / name).read_bytes())
    (stage2 / "recommendation.yaml").write_text("version: 1\nobjective: complements\n", encoding="utf-8")
    context_path = stage2 / "classification_context.json"
    context = json.loads(context_path.read_text())
    context["recommendation_objective"] = "complements"
    write_json(context_path, context)
    refresh_context(catalogs, shop)

    manifest = compare.build_manifest(catalogs)
    assert manifest["shops"][shop]["objective"] == "complements"
    _, payload = compare.prepare_requests(manifest)["jev"][shop][0]
    assert "COMPLEMENTS" in payload["questions"]["candidate_0"]["instructions"]


@pytest.mark.parametrize("config", [
    b"objective: complements\n",
    b"version: 2\nobjective: complements\n",
    b"version: 1\nobjective: unsupported\n",
    b"version: 1\nobjective: complements\nextra: value\n",
])
def test_invalid_recommendation_config_is_rejected(config):
    with pytest.raises(ValueError):
        compare.parse_recommendation_objective(config)


def test_label_independent_sampling(catalogs):
    before = compare.build_manifest(catalogs)
    for shop in compare.SHOPS:
        for direction in ("forward", "reverse"):
            path = catalogs / shop / "stage2" / f"{direction}_judgments.json"
            blocks = [json.loads(line) for line in path.read_text().splitlines()]
            for block in blocks:
                for judgment in block["judgments"]:
                    del judgment["label"]
                    del judgment["reason"]
            write_blocks(path, blocks)
    after = compare.build_manifest(catalogs)
    for shop in compare.SHOPS:
        assert before["shops"][shop]["batches"] == after["shops"][shop]["batches"]
        assert before["shops"][shop]["sources"]["forward"] != after["shops"][shop]["sources"]["forward"]


def test_repeated_blocks_deduplicate_and_preserve_order(catalogs):
    shop = "furniture.co.uk"
    for direction in ("forward", "reverse"):
        path = catalogs / shop / "stage2" / f"{direction}_judgments.json"
        blocks = [json.loads(line) for line in path.read_text().splitlines()]
        block = blocks[0]
        # Splits and repeats mimic appended/resumed judgment artifacts.
        write_blocks(path, [
            {**block, "judgments": block["judgments"][:7]},
            {**block, "judgments": block["judgments"][7:]},
            block,
            *blocks[1:],
        ])
    loaded = compare.load_shop(catalogs, shop)
    assert loaded["diagnostics"]["duplicate_judgments_removed"] == {"forward": 20, "reverse": 20}
    assert [row["candidate_id"] for row in loaded["populations"]["forward"]["0"]] == [str(i) for i in range(1, 21)]
    assert [row["candidate_id"] for row in loaded["populations"]["reverse"]["0"]] == [str(i) for i in range(20, 0, -1)]


def test_selection_final_batch_trim_and_insufficient_population():
    rows = [{"candidate_id": str(i), "retrieved_forward_rank": i + 1} for i in range(45)]
    batches = compare.select_batches("shop", "version", "reverse", {"anchor": rows}, 23)
    assert [len(batch["candidates"]) for batch in batches] == [20, 3]
    assert [row["candidate_id"] for batch in batches for row in batch["candidates"]] == [str(i) for i in range(23)]
    with pytest.raises(ValueError, match="need 46 reverse rows, found 45"):
        compare.select_batches("shop", "version", "reverse", {"anchor": rows}, 46)


@pytest.mark.parametrize("failure", ["digest", "context_version", "judgment_context", "missing_product", "unknown_pair", "duplicate_product"])
def test_invalid_artifacts_fail_before_manifest(catalogs, failure):
    shop = "furniture.co.uk"
    stage2 = catalogs / shop / "stage2"
    if failure in ("digest", "context_version"):
        path = stage2 / ("classification_examples.txt" if failure == "digest" else "classification_context.json")
        if failure == "digest":
            path.write_text("Changed examples")
        else:
            context = json.loads(path.read_text())
            context["context_version"] = "ctx-v1-invalid"
            write_json(path, context)
    elif failure in ("missing_product", "duplicate_product"):
        path = catalogs / shop / "stage1/products_processed.json"
        products = json.loads(path.read_text())
        products = products[:-1] if failure == "missing_product" else products + [products[0]]
        write_json(path, products)
        refresh_context(catalogs, shop)
    else:
        path = stage2 / "forward_judgments.json"
        blocks = [json.loads(line) for line in path.read_text().splitlines()]
        if failure == "judgment_context":
            blocks[0]["prompt_version"] = "stale"
        else:
            blocks[0]["judgments"][0]["candidate_id"] = "unknown"
        write_blocks(path, blocks)
    with pytest.raises(ValueError):
        compare.build_manifest(catalogs)


def test_subset_text_preserved_and_difference_recorded(catalogs):
    shop = "furniture.co.uk"
    path = catalogs / shop / "stage2/subset_products.json"
    subset = json.loads(path.read_text())
    subset[0]["text"] = "Frozen anchor text differs from catalog."
    write_json(path, subset)
    refresh_context(catalogs, shop)
    data = compare.build_manifest(catalogs)["shops"][shop]
    assert data["products"]["0"]["text"] == "Frozen anchor text differs from catalog."
    assert data["products"]["0"]["description"] == "Evidence 0"
    assert data["diagnostics"]["subset_text_differs_from_catalog"] == ["0"]


def test_immutable_manifest(catalogs, tmp_path):
    manifest = compare.build_manifest(catalogs)
    path = tmp_path / "output/manifest.json"
    digest = compare.write_manifest(path, manifest)
    original = path.read_bytes()
    assert digest == hashlib.sha256(original).hexdigest()
    assert compare.write_manifest(path, manifest) == digest
    changed = copy.deepcopy(manifest)
    changed["concurrency"] = 1
    with pytest.raises(ValueError, match="Refusing to replace"):
        compare.write_manifest(path, changed)
    assert path.read_bytes() == original


def test_offline_cli(catalogs, tmp_path, monkeypatch, capsys):
    output = tmp_path / "output/manifest.json"
    monkeypatch.setattr("sys.argv", ["compare.py", "--data-dir", str(catalogs), "--manifest", str(output)])
    compare.main()
    assert output.exists()
    assert "No API requests made" in capsys.readouterr().out


def test_invalid_concurrency(catalogs):
    with pytest.raises(ValueError, match="Concurrency must be positive"):
        compare.build_manifest(catalogs, concurrency=0)


@pytest.fixture
def prepared(catalogs):
    manifest = compare.build_manifest(catalogs)
    return manifest, compare.prepare_requests(manifest)


@pytest.fixture
def relation_prepared(catalogs):
    manifest = compare.build_relation_manifest(catalogs)
    return manifest, compare.prepare_requests(manifest)


def response_for(provider, payload):
    if provider == "gpt":
        # The fixture's user message has a standalone JSON candidate array.
        text = payload["messages"][1]["content"]
        candidates = json.loads(text[text.index("["):text.index("\nFor each candidate,")])
        return {
            "model": payload["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"judgments": [
                {"id": item["id"], "label": "positive", "reason": "Synthetic evidence"}
                for item in reversed(candidates)
            ]})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "prompt_tokens_details": {"cached_tokens": 0},
                      "completion_tokens_details": {"reasoning_tokens": 5}},
        }
    return {
        "model": payload["model"],
        "answers": {key: {"type": "choice", "choice": "positive", "confidence": 0.7,
                          "probabilities": {"positive": 0.8, "skip": 0.15, "hard_negative": 0.05}}
                    for key in reversed(payload["questions"])},
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }


def test_provider_payload_parity(prepared):
    manifest, jobs = prepared
    original = copy.deepcopy(manifest)
    for shop, data in manifest["shops"].items():
        for (batch, gpt), (jev_batch, jev) in zip(jobs["gpt"][shop], jobs["jev"][shop], strict=True):
            assert batch == jev_batch
            anchor = data["products"][batch["anchor_id"]]
            candidates = [{key: data["products"][item["candidate_id"]][key]
                           for key in ("id", "name", "category", "description")}
                          for item in batch["candidates"]]
            assert jev["state"]["base"] == {"id": anchor["id"], "text": anchor["text"]}
            assert jev["state"]["candidates"] == candidates
            assert gpt["messages"][0]["content"] == data["policy"]["system_prompt"].replace(
                "{examples}", data["policy"]["examples"].strip())
            assert gpt["messages"][1]["content"] == data["policy"]["user_prompt"].format(
                anchor_id=anchor["id"], anchor_text=anchor["text"], candidates_json=json.dumps(candidates, indent=2))
            assert gpt["reasoning_effort"] == "medium"
            assert "temperature" not in gpt and "max_completion_tokens" not in gpt
            assert gpt["response_format"]["json_schema"]["strict"] is True
            assert "GPT-only output instructions" not in json.dumps(jev)
            assert "Category guidance must survive." in jev["state"]["label_policy"]["ordered_rules"]
            assert "Synthetic example" in jev["state"]["label_policy"]["system"]
            for i, question in enumerate(jev["questions"].values()):
                assert f"candidates[{i}]" in question["instructions"]
                assert set(question["criteria"]) == set(compare.LABELS)
                assert question["type"] == "choice"
    assert manifest == original


def relation_response_for(provider, payload, batch):
    if provider == "gpt":
        judgments = [{
            "id": item["candidate_id"],
            "identity": {"label": "distinct", "reason": "different products"},
            "relations": {
                "co_purchase": {"label": "yes", "reason": "used together"},
                "alternative": {"label": "no", "reason": "different roles"},
                "incompatible": {"label": "no", "reason": "compatible context"},
            },
        } for item in reversed(batch["candidates"])]
        return {
            "model": payload["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"judgments": judgments})}}],
        }
    answers = {}
    for key in payload["questions"]:
        if key.endswith("_identity"):
            labels, choice = compare.IDENTITY_LABELS, "distinct"
            probabilities = {label: (0.8 if label == choice else 0.05) for label in labels}
        else:
            labels = compare.RELATION_LABELS
            choice = "yes" if key.endswith("_co_purchase") else "no"
            probabilities = {label: (0.8 if label == choice else 0.1) for label in labels}
        answers[key] = {
            "type": "choice", "choice": choice, "confidence": 0.7, "probabilities": probabilities,
        }
    return {"model": payload["model"], "answers": answers}


def test_combined_relation_payloads_use_same_products_and_all_heads(prepared):
    manifest, _ = prepared
    data = manifest["shops"]["furniture.co.uk"]
    batch = data["batches"][0]
    gpt = compare.render_relation_request("gpt", manifest["models"]["gpt"], data, batch)
    jev = compare.render_relation_request("jev", manifest["models"]["jev"], data, batch)

    anchor = data["products"][batch["anchor_id"]]
    candidates = [{key: data["products"][item["candidate_id"]][key]
                   for key in ("id", "name", "category", "description")}
                  for item in batch["candidates"]]
    assert jev["state"]["base"] == {"id": anchor["id"], "text": anchor["text"]}
    assert jev["state"]["candidates"] == candidates
    assert jev["state"]["contract_version"] == compare.RELATION_CONTRACT_VERSION
    assert len(jev["questions"]) == len(candidates) * 4
    for index in range(len(candidates)):
        assert set(jev["questions"][f"candidate_{index}_identity"]["criteria"]) == set(compare.IDENTITY_LABELS)
        assert (jev["questions"][f"candidate_{index}_identity"]["instructions"]
                == compare.IDENTITY_INSTRUCTION.format(index=index))
        for head in compare.RELATION_HEADS:
            question = jev["questions"][f"candidate_{index}_{head}"]
            assert set(question["criteria"]) == set(compare.RELATION_LABELS)
            assert f"candidates[{index}]" in question["instructions"]
    schema = gpt["response_format"]["json_schema"]["schema"]
    judgment = schema["properties"]["judgments"]["items"]
    assert judgment["properties"]["identity"]["properties"]["label"]["enum"] == list(compare.IDENTITY_LABELS)
    assert set(judgment["properties"]["relations"]["properties"]) == set(compare.RELATION_HEADS)
    assert all(head in gpt["messages"][0]["content"] for head in compare.RELATION_HEADS)
    assert all(candidate["id"] in gpt["messages"][1]["content"] for candidate in candidates)
    assert "temperature" not in gpt and "max_completion_tokens" not in gpt


def test_relation_request_uses_saved_contract_version_and_identity_instruction(prepared):
    manifest, _ = prepared
    data = manifest["shops"]["furniture.co.uk"]
    batch = data["batches"][0]
    contract = compare.relation_contract_snapshot()
    contract["version"] = "saved-contract-version"
    contract["identity_instruction"] = "Saved identity instruction for candidates[{index}]."

    payload = compare.render_relation_request("jev", manifest["models"]["jev"], data, batch, contract)
    assert payload["state"]["contract_version"] == "saved-contract-version"
    for index in range(len(batch["candidates"])):
        assert (payload["questions"][f"candidate_{index}_identity"]["instructions"]
                == f"Saved identity instruction for candidates[{index}].")


def test_v2_manifest_preserves_v1_frozen_workload(catalogs, relation_prepared):
    legacy = compare.build_manifest(catalogs)
    manifest, jobs = relation_prepared
    comparable = copy.deepcopy(manifest)
    assert comparable.pop("contract")["resolver_version"] == compare.RESOLVER_VERSION
    comparable["schema_version"] = compare.MANIFEST_V1
    assert comparable == legacy
    assert manifest["schema_version"] == compare.MANIFEST_V2
    assert manifest["contract"] == compare.relation_contract_snapshot()
    assert manifest["limitations"] == legacy["limitations"]
    for shop, data in manifest["shops"].items():
        assert len({item["pair_id"] for batch in data["batches"] for item in batch["candidates"]}) == 400
        for provider in ("gpt", "jev"):
            assert jobs[provider][shop][0][0] == data["batches"][0]
        assert len(jobs["jev"][shop][0][1]["questions"]) == len(data["batches"][0]["candidates"]) * 4


@pytest.mark.parametrize("provider", ["gpt", "jev"])
def test_combined_relation_answers_are_normalized_in_pair_order(prepared, provider):
    manifest, _ = prepared
    data = manifest["shops"]["furniture.co.uk"]
    batch = data["batches"][0]
    payload = compare.render_relation_request(provider, manifest["models"][provider], data, batch)
    answers = compare.parse_relation_answers(provider, relation_response_for(provider, payload, batch), batch)

    assert [answer["pair_id"] for answer in answers] == [item["pair_id"] for item in batch["candidates"]]
    assert all(answer["identity"]["label"] == "distinct" for answer in answers)
    assert all(answer["relations"]["co_purchase"]["label"] == "yes" for answer in answers)
    assert all(answer["relations"]["alternative"]["label"] == "no" for answer in answers)
    if provider == "gpt":
        assert all(answer["identity"]["reason"] == "different products" for answer in answers)
        assert all(answer["relations"]["co_purchase"]["probabilities"] is None for answer in answers)
    else:
        assert all(answer["identity"]["reason"] is None for answer in answers)
        assert all(answer["identity"]["probability_sum"] == pytest.approx(1) for answer in answers)


@pytest.mark.parametrize("provider", ["gpt", "jev"])
def test_combined_relation_answers_reject_missing_heads(prepared, provider):
    manifest, _ = prepared
    data = manifest["shops"]["furniture.co.uk"]
    batch = data["batches"][0]
    payload = compare.render_relation_request(provider, manifest["models"][provider], data, batch)
    body = relation_response_for(provider, payload, batch)
    if provider == "gpt":
        parsed = json.loads(body["choices"][0]["message"]["content"])
        del parsed["judgments"][0]["relations"]["alternative"]
        body["choices"][0]["message"]["content"] = json.dumps(parsed)
    else:
        del body["answers"]["candidate_0_alternative"]
    with pytest.raises(ValueError):
        compare.parse_relation_answers(provider, body, batch)


def test_v2_run_batch_retains_partial_gpt_judgments(relation_prepared, monkeypatch):
    manifest, jobs = relation_prepared
    batch, payload = jobs["gpt"]["furniture.co.uk"][0]
    body = relation_response_for("gpt", payload, batch)
    parsed = json.loads(body["choices"][0]["message"]["content"])
    parsed["judgments"] = parsed["judgments"][:-2]
    body["choices"][0]["message"]["content"] = json.dumps(parsed)
    calls = []
    monkeypatch.setattr(compare.time, "sleep", lambda delay: None)

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = compare.run_batch(
            client, "gpt", "furniture.co.uk", batch, payload, "dummy",
            schema_version=compare.MANIFEST_V2,
        )
    assert result["status"] == "partial" and result["error"] == "missing_candidate_answers"
    assert len(result["answers"]) == len(batch["candidates"]) - 2
    assert result["retries"] == compare.MAX_RETRIES
    assert calls == [payload] * (compare.MAX_RETRIES + 1)
    assert all("identity" in answer and set(answer["relations"]) == set(compare.RELATION_HEADS)
               for answer in result["answers"])


@pytest.mark.parametrize(("identity", "relations", "basket", "status", "resolution"), [
    ("duplicate", ("yes", "yes", "yes"), "skip", "filtered", "identity_filtered_duplicate"),
    ("variant", ("no", "yes", "no"), "skip", "filtered", "identity_filtered_variant"),
    ("redundant", ("yes", "no", "no"), "skip", "filtered", "identity_filtered_redundant"),
    ("distinct", ("yes", "no", "no"), "positive", "resolved", "co_purchase_positive"),
    ("distinct", ("no", "yes", "no"), "hard_negative", "resolved", "alternative_only"),
    ("distinct", ("no", "no", "yes"), "hard_negative", "resolved", "incompatible_only"),
    ("distinct", ("no", "no", "no"), "skip", "resolved", "unrelated_skip"),
    ("uncertain", ("uncertain", "no", "no"), "skip", "uncertain", "uncertain_skip"),
    ("distinct", ("yes", "yes", "no"), "conflict", "conflict", "conflicting_positive_relations"),
    ("distinct", ("yes", "no", "yes"), "conflict", "conflict", "conflicting_positive_relations"),
    ("distinct", ("no", "yes", "yes"), "conflict", "conflict", "conflicting_positive_relations"),
])
def test_basket_resolver(identity, relations, basket, status, resolution):
    answer = {
        "identity": {"label": identity},
        "relations": {
            head: {"label": label} for head, label in zip(compare.RELATION_HEADS, relations, strict=True)
        },
    }
    result = compare.resolve_basket_judgment(answer)
    assert result["basket_label"] == basket
    assert result["resolution_status"] == status
    assert result["resolution"] == resolution
    assert result["resolver_version"] == compare.RESOLVER_VERSION
    assert result["identity_action"] == (
        "filter" if identity in {"duplicate", "variant", "redundant"}
        else "continue_uncertain" if identity == "uncertain" else "continue"
    )


@pytest.mark.parametrize("identity", compare.IDENTITY_LABELS)
@pytest.mark.parametrize("relations", itertools.product(compare.RELATION_LABELS, repeat=3))
def test_basket_resolver_covers_complete_decision_matrix(identity, relations):
    answer = {
        "identity": {"label": identity},
        "relations": {
            head: {"label": label} for head, label in zip(compare.RELATION_HEADS, relations, strict=True)
        },
    }
    result = compare.resolve_basket_judgment(answer, compare.RESOLVER_VERSION)
    yes_count = relations.count("yes")
    assert result["conflict"] == (yes_count > 1)
    if identity in {"duplicate", "variant", "redundant"}:
        assert result["basket_label"] == "skip" and result["resolution_status"] == "filtered"
    elif yes_count > 1:
        assert result["basket_label"] == "conflict" and result["resolution_status"] == "conflict"
    elif relations == ("yes", "no", "no"):
        assert result["basket_label"] == "positive" and result["resolution_status"] == "resolved"
    elif relations in {("no", "yes", "no"), ("no", "no", "yes")}:
        assert result["basket_label"] == "hard_negative" and result["resolution_status"] == "resolved"
    elif "uncertain" in relations:
        assert result["basket_label"] == "skip" and result["resolution_status"] == "uncertain"
    else:
        assert relations == ("no", "no", "no")
        assert result["basket_label"] == "skip" and result["resolution_status"] == "resolved"


def test_basket_resolver_rejects_unknown_saved_version():
    answer = {
        "identity": {"label": "distinct"},
        "relations": {head: {"label": "no"} for head in compare.RELATION_HEADS},
    }
    with pytest.raises(ValueError, match="Unsupported saved resolver version"):
        compare.resolve_basket_judgment(answer, "future-unknown-resolver")


def test_jev_payload_preserves_recommendation_direction(prepared):
    manifest, _ = prepared
    data = manifest["shops"]["themeatboys.nl"]
    source = data["batches"][0]
    anchor_id = source["anchor_id"]
    candidate_id = source["candidates"][0]["candidate_id"]
    forward = {**source, "candidates": [source["candidates"][0]]}
    reverse = {
        "batch_id": "themeatboys.nl/reverse/direction-test", "direction": "reverse",
        "anchor_id": candidate_id,
        "candidates": [{"candidate_id": anchor_id, "pair_id": "direction-test"}],
    }
    settings = manifest["models"]["jev"]
    forward_payload = compare.render_request("jev", settings, data, forward)
    reverse_payload = compare.render_request("jev", settings, data, reverse)

    assert forward_payload["state"]["base"]["id"] == anchor_id
    assert forward_payload["state"]["candidates"][0]["id"] == candidate_id
    assert reverse_payload["state"]["base"]["id"] == candidate_id
    assert reverse_payload["state"]["candidates"][0]["id"] == anchor_id
    assert "Do not reverse the recommendation direction" in forward_payload["questions"]["candidate_0"]["instructions"]


@pytest.mark.parametrize("provider", ["gpt", "jev"])
def test_answer_mapping_and_usage(prepared, provider):
    _, jobs = prepared
    batch, payload = jobs[provider]["furniture.co.uk"][0]
    body = response_for(provider, payload)
    answers = compare.parse_answers(provider, body, batch)
    assert [answer["pair_id"] for answer in answers] == [item["pair_id"] for item in batch["candidates"]]
    assert all(answer["label"] == "positive" for answer in answers)
    usage = compare.normalize_usage(provider, body["usage"])
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == (20 if provider == "gpt" else 0)
    assert compare.normalize_usage(provider, None)["input_tokens"] is None
    assert compare.normalize_usage(provider, {})["output_tokens"] is None
    if provider == "gpt":
        assert usage["cached_input_tokens"] == 0 and usage["reasoning_tokens"] == 5
    else:
        assert all(answer["reason"] is None for answer in answers)


@pytest.mark.parametrize("failure", ["extra", "duplicate", "label", "reason", "refusal", "truncated"])
def test_gpt_rejects_invalid_answers(prepared, failure):
    _, jobs = prepared
    batch, payload = jobs["gpt"]["furniture.co.uk"][0]
    body = response_for("gpt", payload)
    message = body["choices"][0]["message"]
    judgments = json.loads(message["content"])["judgments"]
    if failure == "extra":
        judgments.append({"id": "unknown", "label": "skip", "reason": ""})
    elif failure == "duplicate":
        judgments.append(judgments[0])
    elif failure == "label":
        judgments[0]["label"] = "negative"
    elif failure == "reason":
        judgments[0]["reason"] = "x" * 81
    elif failure == "refusal":
        message["refusal"] = "Synthetic refusal"
    else:
        body["choices"][0]["finish_reason"] = "length"
    message["content"] = json.dumps({"judgments": judgments})
    with pytest.raises(ValueError):
        compare.parse_answers("gpt", body, batch)


@pytest.mark.parametrize("failure", ["missing", "extra", "label", "probabilities", "sum", "confidence", "maximal"])
def test_jev_rejects_invalid_answers(prepared, failure):
    _, jobs = prepared
    batch, payload = jobs["jev"]["furniture.co.uk"][0]
    body = response_for("jev", payload)
    answer = body["answers"]["candidate_0"]
    if failure == "missing":
        del body["answers"]["candidate_0"]
    elif failure == "extra":
        body["answers"]["extra"] = answer
    elif failure == "label":
        answer["choice"] = "negative"
    elif failure == "probabilities":
        answer["probabilities"]["skip"] = float("nan")
    elif failure == "sum":
        answer["probabilities"]["positive"] = 0.9
    elif failure == "confidence":
        answer["confidence"] = True
    else:
        answer["choice"] = "skip"
    with pytest.raises(ValueError):
        compare.parse_answers("jev", body, batch)


@pytest.mark.parametrize("response_text", ['{"answers":{},"answers":{}}', '{"usage":NaN}', '{"usage":1e999}'])
def test_strict_json_rejects_duplicate_keys_and_nonfinite(response_text):
    with pytest.raises(ValueError):
        compare.strict_json(response_text)


@pytest.mark.parametrize("provider", ["gpt", "jev"])
@pytest.mark.parametrize("first_status", [429, 500, 529, "timeout"])
def test_retry_accounting(prepared, provider, first_status, monkeypatch):
    _, jobs = prepared
    batch, payload = jobs[provider]["furniture.co.uk"][0]
    calls, sleeps = [], []
    monkeypatch.setattr(compare.time, "sleep", sleeps.append)

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            if first_status == "timeout":
                raise httpx.ReadTimeout("Sensitive request detail must not be logged", request=request)
            return httpx.Response(first_status, json={"error": "Synthetic transient error"}, headers={"retry-after": "0.25"})
        return httpx.Response(200, json=response_for(provider, payload),
                              headers={"x-request-id": "gpt-trace", "x-typesafe-request-id": "jev-trace"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = compare.run_batch(client, provider, "furniture.co.uk", batch, payload, "dummy-test-key")
    assert result["status"] == "ok" and result["error"] is None
    assert len(calls) == 2 and result["retries"] == 1
    assert sleeps == ([1] if first_status == "timeout" else [0.25])
    assert result["attempts"][1]["request_id"] == f"{provider}-trace"
    assert result["attempts"][0]["usage"]["input_tokens"] is None
    assert result["raw_response"] == response_for(provider, payload)
    assert all(attempt["latency_seconds"] >= 0 for attempt in result["attempts"])
    assert "Sensitive request detail" not in json.dumps(result)
    assert "dummy-test-key" not in json.dumps(result)


@pytest.mark.parametrize("status", [401, 403, 422, 200])
def test_nonretryable_failures_never_become_skip(prepared, status):
    _, jobs = prepared
    batch, payload = jobs["jev"]["furniture.co.uk"][0]
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"model": "jev-1.13.0", "answers": {}, "usage": {"input_tokens": 0}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = compare.run_batch(client, "jev", "furniture.co.uk", batch, payload, "dummy")
    assert len(calls) == 1 and result["retries"] == 0
    assert result["status"] == "error" and result["answers"] == []
    assert result["attempts"][0]["usage"]["input_tokens"] == 0
    assert result["error"] == ("invalid_response" if status == 200 else f"http_{status}")


def test_retry_exhaustion_and_long_retry_after(prepared, monkeypatch):
    _, jobs = prepared
    batch, payload = jobs["jev"]["furniture.co.uk"][0]
    sleeps = []
    monkeypatch.setattr(compare.time, "sleep", sleeps.append)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
        result = compare.run_batch(client, "jev", "furniture.co.uk", batch, payload, "dummy")
    assert result["status"] == "error" and len(result["attempts"]) == 3
    assert sleeps == [1, 2] and result["retries"] == 2
    assert compare.retry_delay("120", 0) is None
    assert compare.retry_delay("not-a-date", 0) == 1
    assert compare.retry_delay("nan", 0) is None


def test_execute_groups_are_sequential_and_outputs_exclusive(prepared, tmp_path):
    manifest, jobs = prepared
    # Two batches per group suffice to exercise concurrency and persistence.
    jobs = {provider: {shop: work[:2] for shop, work in shops.items()} for provider, shops in jobs.items()}
    calls = []

    def handler(request):
        provider = "gpt" if request.url.host == "api.openai.com" else "jev"
        calls.append(provider)
        return httpx.Response(200, json=response_for(provider, json.loads(request.content)))

    directory = tmp_path / "run"
    keys = {"gpt": "dummy-gpt-key", "jev": "dummy-jev-key"}
    failures = compare.execute(manifest, jobs, directory, "manifest-digest", keys, transport=httpx.MockTransport(handler))
    assert failures == 0 and len(calls) == 8
    records = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    assert records[0]["record_type"] == "run_start" and records[0]["concurrency"] == 8
    assert records[-1] == {"record_type": "run_end", "failed_batches": 0}
    active = None
    for record in records[1:-1]:
        group = (record["provider"], record["shop"])
        if record["record_type"] == "group_start":
            assert active is None
            active = group
        elif record["record_type"] == "group_end":
            assert active == group and record["wall_seconds"] >= 0
            active = None
        else:
            assert active == group and record["answers"]
    assert active is None
    assert "dummy-gpt-key" not in json.dumps(records) and "dummy-jev-key" not in json.dumps(records)
    with pytest.raises(FileExistsError):
        compare.execute(manifest, jobs, directory, "manifest-digest", keys, transport=httpx.MockTransport(handler))
    assert len(calls) == 8


def test_completed_batch_is_flushed_before_group_finishes(prepared, tmp_path):
    manifest, jobs = prepared
    manifest["concurrency"] = 2
    work = jobs["jev"]["furniture.co.uk"][:2]
    log = tmp_path / "run/requests.jsonl"
    second_started = threading.Event()
    allow_second = threading.Event()
    first_id = work[0][1]["state"]["base"]["id"]

    def handler(request):
        payload = json.loads(request.content)
        if payload["state"]["base"]["id"] == first_id:
            assert second_started.wait(5)
        else:
            second_started.set()
            assert allow_second.wait(5)
        return httpx.Response(200, json=response_for("jev", payload))

    errors = []

    def run():
        try:
            compare.execute(manifest, {"jev": {"furniture.co.uk": work}}, log.parent, "digest",
                            {"gpt": "dummy", "jev": "dummy"}, transport=httpx.MockTransport(handler))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert second_started.wait(5)
        deadline = compare.time.monotonic() + 5
        while compare.time.monotonic() < deadline:
            text = log.read_text()
            if '"record_type": "batch"' in text:
                assert '"record_type": "group_end"' not in text
                break
            compare.time.sleep(0.01)
        else:
            pytest.fail("Completed batch was not flushed")
    finally:
        allow_second.set()
        thread.join(5)
    assert not thread.is_alive() and not errors


def test_execute_flag_requires_completed_baseline_before_network(catalogs, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    output = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["compare.py", "--data-dir", str(catalogs),
                                    "--manifest", str(tmp_path / "manifest.json"),
                                    "--baseline-root", str(tmp_path / "no-baseline"),
                                    "--output-dir", str(output), "--execute"])
    with pytest.raises(SystemExit) as result:
        compare.main()
    assert result.value.code == 1
    assert "No completed matching GPT baseline" in capsys.readouterr().err
    assert not output.exists()


def test_malformed_json_response_is_not_retried(prepared):
    _, jobs = prepared
    batch, payload = jobs["gpt"]["furniture.co.uk"][0]
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="not JSON"))) as client:
        result = compare.run_batch(client, "gpt", "furniture.co.uk", batch, payload, "dummy")
    assert result["status"] == "error" and result["error"] == "invalid_response"
    assert result["answers"] == [] and result["raw_response"] is None
    assert len(result["attempts"]) == 1 and result["retries"] == 0


def test_long_retry_after_stops_without_early_retry(prepared, monkeypatch):
    _, jobs = prepared
    batch, payload = jobs["jev"]["furniture.co.uk"][0]
    sleeps = []
    monkeypatch.setattr(compare.time, "sleep", sleeps.append)
    transport = httpx.MockTransport(lambda request: httpx.Response(429, headers={"retry-after": "120"}))
    with httpx.Client(transport=transport) as client:
        result = compare.run_batch(client, "jev", "furniture.co.uk", batch, payload, "dummy")
    assert result["status"] == "error" and result["error"] == "http_429"
    assert len(result["attempts"]) == 1 and result["retries"] == 0 and sleeps == []
    assert result["attempts"][0]["retry_not_scheduled"] == "retry_after_exceeds_budget"


def test_execute_persists_partial_failure_and_finishes_groups(prepared, tmp_path):
    manifest, jobs = prepared
    jobs = {provider: {shop: work[:1] for shop, work in shops.items()} for provider, shops in jobs.items()}
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(422, json={"error": "Synthetic schema error"})
        provider = "gpt" if request.url.host == "api.openai.com" else "jev"
        return httpx.Response(200, json=response_for(provider, json.loads(request.content)))

    directory = tmp_path / "partial-run"
    failures = compare.execute(manifest, jobs, directory, "digest", {"gpt": "dummy", "jev": "dummy"},
                               transport=httpx.MockTransport(handler))
    assert failures == 1 and len(calls) == 4
    records = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    batches = [record for record in records if record["record_type"] == "batch"]
    assert batches[0]["status"] == "error" and batches[0]["answers"] == []
    assert all(record["status"] == "ok" for record in batches[1:])
    assert sum(record["record_type"] == "group_end" for record in records) == 4
    assert records[-1] == {"record_type": "run_end", "failed_batches": 1}


@pytest.fixture
def completed_run(prepared, tmp_path):
    manifest, jobs = prepared
    path, directory = tmp_path / "manifest.json", tmp_path / "run"
    digest = compare.write_manifest(path, manifest)

    def handler(request):
        provider = "gpt" if request.url.host == "api.openai.com" else "jev"
        return httpx.Response(200, json=response_for(provider, json.loads(request.content)))

    assert compare.execute(manifest, jobs, directory, digest, {"gpt": "dummy", "jev": "dummy"},
                           transport=httpx.MockTransport(handler)) == 0
    return path, directory


def test_complete_reports_and_deterministic_regeneration(completed_run):
    path, directory = completed_run
    summary = compare.write_reports(path, directory)
    assert summary["run_complete"] and summary["all_pairs_classified"]
    for shop in compare.SHOPS:
        agreement = summary["agreement"][shop]
        assert agreement["paired_success"] == agreement["agreements"] == 400
        assert agreement["agreement_rate"] == 1 and agreement["unpaired"] == 0
        for provider in ("gpt", "jev"):
            group = summary["groups"][f"{provider}/{shop}"]
            assert group["http_attempts"] == group["batch_requests"] == 20
            assert group["retries"] == group["failed_attempts"] == group["failed_pairs"] == 0
            assert group["successful_pairs"] == group["label_counts"]["positive"] == 400
            assert group["pairs_per_second"] == pytest.approx(400 / group["wall_seconds"])
            assert group["batch_sizes"] == {"20": 20}
            assert group["request_latency_seconds"]["count"] == 20
            assert group["usage"]["input_tokens"]["total"] == 2000
            unit_cost = 0.000013 if provider == "gpt" else 0.0000042
            assert group["cost_estimate"]["total_usd"] == pytest.approx(20 * unit_cost)
    with (directory / "paired_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 800 and len({row["pair_id"] for row in rows}) == 800
    with (directory / "disagreements.csv").open(newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    for provider in ("gpt", "jev"):
        records = [json.loads(line) for line in (directory / f"{provider}_classifications.jsonl").read_text().splitlines()]
        assert len(records) == 800 and all(row["status"] == "ok" for row in records)
    before = {file.name: file.read_bytes() for file in directory.iterdir()}
    assert compare.write_reports(path, directory) == summary
    assert {file.name: file.read_bytes() for file in directory.iterdir()} == before


def test_disagreement_including_semantic_skip(completed_run):
    path, directory = completed_run
    log_path = directory / "requests.jsonl"
    records, _ = compare.read_request_log(log_path)
    batch = next(record for record in records if record["record_type"] == "batch" and record["provider"] == "jev")
    answer = batch["answers"][0]
    answer["label"] = "skip"
    answer["probabilities"] = {"positive": 0.1, "hard_negative": 0.1, "skip": 0.8}
    write_blocks(log_path, records)
    summary = compare.write_reports(path, directory)
    agreement = summary["agreement"][batch["shop"]]
    assert agreement["paired_success"] == 400 and agreement["agreements"] == 399
    assert agreement["matrix_gpt_rows_jev_columns"]["positive"]["skip"] == 1
    with (directory / "disagreements.csv").open(newline="") as handle:
        disagreements = list(csv.DictReader(handle))
    assert len(disagreements) == 1 and disagreements[0]["pair_id"] == answer["pair_id"]
    assert disagreements[0]["jev_label"] == "skip" and disagreements[0]["jev_status"] == "ok"


def test_failed_and_missing_pairs_do_not_enter_agreement(completed_run):
    path, directory = completed_run
    log_path = directory / "requests.jsonl"
    records, _ = compare.read_request_log(log_path)
    failed = next(record for record in records if record["record_type"] == "batch")
    failed.update(status="error", error="invalid_response", answers=[])
    failed["attempts"][0]["error"] = "invalid_response"
    failed["raw_response"]["choices"][0]["message"]["content"] = "unparseable response"
    missing = next(record for record in reversed(records) if record["record_type"] == "batch")
    records = [record for record in records if record is not missing and record["record_type"] != "run_end"
               and not (record["record_type"] == "group_end" and record["provider"] == missing["provider"]
                        and record["shop"] == missing["shop"])]
    write_blocks(log_path, records)
    summary = compare.write_reports(path, directory)
    assert not summary["run_complete"] and not summary["all_pairs_classified"]
    for shop in compare.SHOPS:
        assert summary["agreement"][shop]["paired_success"] == 380
        assert summary["agreement"][shop]["unpaired"] == 20
        assert summary["agreement"][shop]["disagreements"] == 0
    missing_group = summary["groups"][f"{missing['provider']}/{missing['shop']}"]
    assert missing_group["missing_pairs"] == 20
    assert missing_group["wall_seconds"] is None and missing_group["pairs_per_second"] is None
    assert missing_group["usage"]["input_tokens"]["total"] is None
    assert missing_group["cost_estimate"]["total_usd"] is None
    error_group = summary["groups"][f"{failed['provider']}/{failed['shop']}"]
    assert error_group["failed_pairs"] == 20 and error_group["failed_batches"] == 1
    with (directory / "paired_results.csv").open(newline="") as handle:
        unpaired = [row for row in csv.DictReader(handle) if row["paired_success"] == "False"]
    assert len(unpaired) == 40 and all(row["agreement"] == "" for row in unpaired)
    assert all(row["gpt_label"] == "" or row["jev_label"] == "" for row in unpaired)


def test_retry_usage_and_unknown_cost_coverage(completed_run):
    path, directory = completed_run
    log_path = directory / "requests.jsonl"
    records, _ = compare.read_request_log(log_path)
    batch = next(record for record in records if record["record_type"] == "batch")
    retry = copy.deepcopy(batch["attempts"][0])
    retry.update(error="http_429", http_status=429, usage=compare.normalize_usage("gpt", None))
    batch["attempts"].insert(0, retry)
    # An explicitly reported zero is still a known count, unlike the failed attempt.
    batch["attempts"][1]["usage"] = {key: 0 for key in batch["attempts"][1]["usage"]}
    write_blocks(log_path, records)
    summary = compare.write_reports(path, directory)
    group = summary["groups"][f"{batch['provider']}/{batch['shop']}"]
    assert group["http_attempts"] == 21 and group["retries"] == group["failed_attempts"] == 1
    assert group["request_latency_seconds"]["count"] == 21
    assert group["usage"]["input_tokens"] == {
        "reported_subtotal": 1900, "total": None, "reported_attempts": 20, "missing_attempts": 1,
    }
    assert group["cost_estimate"]["priced_attempts"] == 20
    assert group["cost_estimate"]["unpriced_attempts"] == 1
    assert group["cost_estimate"]["total_usd"] is None
    assert group["cost_estimate"]["priced_attempts_subtotal_usd"] == pytest.approx(19 * 0.000013)


def test_cached_cost_and_reasoning_not_double_counted():
    usage = {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20, "reasoning_tokens": 5}
    assert compare.attempt_cost("gpt", usage) == pytest.approx(0.0000112)
    assert compare.attempt_cost("gpt", {**usage, "cached_input_tokens": None}) is None
    assert compare.attempt_cost("jev", {**usage, "output_tokens": None}) == pytest.approx(0.0000042)
    assert compare.attempt_cost("jev", {**usage, "input_tokens": None}) is None
    assert compare.attempt_cost("gpt", {key: 0 for key in usage}) == 0


@pytest.mark.parametrize("failure", [
    "manifest_digest", "duplicate_batch", "duplicate_answer", "interior_line",
    "group_model", "batch_model", "payload_model", "batch_count", "reasoning",
])
def test_report_rejects_conflicting_logs(completed_run, failure):
    path, directory = completed_run
    log_path = directory / "requests.jsonl"
    records, _ = compare.read_request_log(log_path)
    batch = next(record for record in records if record["record_type"] == "batch")
    if failure == "manifest_digest":
        records[0]["manifest_sha256"] = "wrong-digest"
    elif failure == "duplicate_batch":
        records.insert(-1, batch)
    elif failure == "duplicate_answer":
        batch["answers"].append(batch["answers"][0])
    elif failure in ("group_model", "batch_count"):
        group = next(record for record in records if record["record_type"] == "group_start")
        group["requested_model" if failure == "group_model" else "batches"] = "wrong"
    elif failure == "batch_model":
        batch["requested_model"] = "different-model"
    elif failure == "payload_model":
        batch["request"]["model"] = "different-model"
    elif failure == "reasoning":
        batch["request"]["reasoning_effort"] = "low"
    write_blocks(log_path, records)
    if failure == "interior_line":
        log_path.write_text(log_path.read_text().replace("\n", "\ninvalid json\n", 1))
    with pytest.raises(ValueError):
        compare.write_reports(path, directory)
    assert not (directory / "summary.json").exists()


def test_truncated_log_tail_is_flagged(completed_run):
    path, directory = completed_run
    log_path = directory / "requests.jsonl"
    records, _ = compare.read_request_log(log_path)
    write_blocks(log_path, records[:-1])
    with log_path.open("a") as handle:
        handle.write('{"record_type": "run_')
    summary = compare.write_reports(path, directory)
    assert summary["truncated_final_line"] and not summary["run_complete"]
    assert summary["all_pairs_classified"]


def test_report_only_bypasses_preparation_and_execution(completed_run, monkeypatch, capsys):
    path, directory = completed_run

    def deny(*args, **kwargs):
        pytest.fail("Report-only must not prepare new work or execute requests")

    monkeypatch.setattr(compare, "build_manifest", deny)
    monkeypatch.setattr(compare, "execute", deny)
    monkeypatch.setattr(compare.httpx, "Client", deny)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["compare.py", "--report-only", "--manifest", str(path), "--output-dir", str(directory)])
    compare.main()
    assert "Reports rebuilt offline" in capsys.readouterr().out


def test_csv_catalog_text_is_not_a_formula(tmp_path):
    path = tmp_path / "safe.csv"
    compare.write_csv(path, [{"name": "  =1+1", "label": "skip", "confidence": 0.9}], ["name", "label", "confidence"])
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row == {"name": "'  =1+1", "label": "skip", "confidence": "0.9"}


def test_no_completed_batches_reports_unknown_not_zero(completed_run):
    path, directory = completed_run
    log_path = directory / "requests.jsonl"
    records, _ = compare.read_request_log(log_path)
    write_blocks(log_path, records[:1])
    summary = compare.write_reports(path, directory)
    for group in summary["groups"].values():
        assert group["http_attempts"] == 0 and group["missing_pairs"] == 400
        assert group["request_latency_seconds"]["mean"] is None
        assert group["usage"]["input_tokens"]["reported_subtotal"] is None
        assert group["cost_estimate"]["priced_attempts_subtotal_usd"] is None
    assert all(value["agreement_rate"] is None for value in summary["agreement"].values())
    assert not summary["all_pairs_classified"] and not summary["run_complete"]


@pytest.mark.parametrize("probability_sum", [0.985, 0.99, 1.01, 1.015])
def test_jev_rounding_preserves_reported_values(prepared, probability_sum):
    _, jobs = prepared
    batch, payload = jobs["jev"]["furniture.co.uk"][0]
    body = response_for("jev", payload)
    original = body["answers"]["candidate_0"]["probabilities"]
    original["positive"] += probability_sum - 1
    snapshot = copy.deepcopy(body)
    answers = compare.parse_answers("jev", body, batch)
    assert answers[0]["probabilities"] == original and body == snapshot
    assert answers[0]["probability_sum"] == pytest.approx(probability_sum)
    assert answers[0]["probability_rounding_warning"] is True
    assert answers[0]["label"] == "positive"


@pytest.mark.parametrize("probability_sum", [0.984, 1.016])
def test_jev_rounding_tolerance_remains_bounded(prepared, probability_sum):
    _, jobs = prepared
    batch, payload = jobs["jev"]["furniture.co.uk"][0]
    body = response_for("jev", payload)
    body["answers"]["candidate_0"]["probabilities"]["positive"] += probability_sum - 1
    with pytest.raises(ValueError, match="rounding tolerance"):
        compare.parse_answers("jev", body, batch)


@pytest.mark.parametrize("missing_count", [1, 4, 20])
def test_gpt_partial_answers_preserve_id_mapping(prepared, missing_count, monkeypatch):
    monkeypatch.setattr(compare.time, "sleep", lambda delay: None)
    _, jobs = prepared
    batch, payload = jobs["gpt"]["furniture.co.uk"][0]
    body = response_for("gpt", payload)
    message = body["choices"][0]["message"]
    judgments = json.loads(message["content"])["judgments"]
    returned = judgments[missing_count:]
    message["content"] = json.dumps({"judgments": returned})
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))) as client:
        result = compare.run_batch(client, "gpt", "furniture.co.uk", batch, payload, "dummy")
    assert result["status"] == ("partial" if returned else "error")
    assert result["error"] == "missing_candidate_answers" and result["retries"] == 2
    assert len(result["answers"]) == len(returned)
    returned_ids = {row["id"] for row in returned}
    assert {answer["candidate_id"] for answer in result["answers"]} == returned_ids
    assert all(json.loads(answer["pair_id"])[3] == answer["candidate_id"] for answer in result["answers"])


def test_offline_reextraction_preserves_log_and_accounting(completed_run):
    path, directory = completed_run
    log_path = directory / "requests.jsonl"
    baseline = compare.write_reports(path, directory)
    records, _ = compare.read_request_log(log_path)
    gpt = next(record for record in records if record["record_type"] == "batch" and record["provider"] == "gpt")
    jev = next(record for record in records if record["record_type"] == "batch" and record["provider"] == "jev")
    message = gpt["raw_response"]["choices"][0]["message"]
    judgments = json.loads(message["content"])["judgments"]
    missing_id = judgments.pop()["id"]
    message["content"] = json.dumps({"judgments": judgments})
    jev["raw_response"]["answers"]["candidate_0"]["probabilities"]["positive"] = 0.79
    for record in (gpt, jev):
        record.update(status="error", error="invalid_response", answers=[])
        record["attempts"][-1]["error"] = "invalid_response"
    write_blocks(log_path, records)
    original_log, original_manifest = log_path.read_bytes(), path.read_bytes()
    summary = compare.write_reports(path, directory)
    assert log_path.read_bytes() == original_log and path.read_bytes() == original_manifest
    assert summary["extraction"]["request_log_sha256"] == hashlib.sha256(original_log).hexdigest()
    for key, old in baseline["groups"].items():
        new = summary["groups"][key]
        assert new["wall_seconds"] == old["wall_seconds"]
        assert new["request_latency_seconds"] == old["request_latency_seconds"]
        assert new["cost_estimate"] == old["cost_estimate"] and new["usage"] == old["usage"]
    shop = gpt["shop"]
    assert summary["agreement"][shop]["paired_success"] == 399
    gpt_group, jev_group = summary["groups"][f"gpt/{shop}"], summary["groups"][f"jev/{shop}"]
    assert gpt_group["successful_pairs"] == 399 and gpt_group["failed_pairs"] == 1
    assert gpt_group["partial_batches"] == 1 and gpt_group["failed_batches"] == 0
    assert jev_group["successful_pairs"] == 400 and jev_group["failed_pairs"] == 0
    assert jev_group["failed_attempts"] == 0 and jev_group["recorded_failed_attempts"] == 1
    assert jev_group["probability_rounding_warnings"] == 1
    assert gpt_group["reextracted_batches"] == jev_group["reextracted_batches"] == 1
    classifications = [json.loads(line) for line in (directory / "gpt_classifications.jsonl").read_text().splitlines()]
    errors = [row for row in classifications if row["status"] != "ok"]
    assert len(errors) == 1 and errors[0]["candidate_id"] == missing_id
    assert errors[0]["label"] is None and errors[0]["error"] == "missing_candidate_answer"
    assert summary["run_complete"] and not summary["all_pairs_classified"]
    before = {file.name: file.read_bytes() for file in directory.iterdir()}
    assert compare.write_reports(path, directory) == summary
    assert {file.name: file.read_bytes() for file in directory.iterdir()} == before


def test_new_partial_record_reports_returned_answers(completed_run):
    path, directory = completed_run
    log_path = directory / "requests.jsonl"
    records, _ = compare.read_request_log(log_path)
    record = next(record for record in records if record["record_type"] == "batch")
    record["answers"].pop()
    record["status"], record["error"] = "partial", "missing_candidate_answers"
    record["attempts"][-1]["error"] = record["error"]
    write_blocks(log_path, records)
    summary = compare.write_reports(path, directory)
    group = summary["groups"][f"{record['provider']}/{record['shop']}"]
    assert group["successful_pairs"] == 399 and group["failed_pairs"] == 1
    assert group["partial_batches"] == 1 and group["reextracted_batches"] == 0


@pytest.fixture
def baseline_seed(completed_run):
    manifest_path, directory = completed_run
    manifest = json.loads(manifest_path.read_text())
    records, _ = compare.read_request_log(directory / "requests.jsonl")
    missing = set()
    for shop, count in (("furniture.co.uk", 4), ("themeatboys.nl", 1)):
        record = next(row for row in records if row["record_type"] == "batch" and row["provider"] == "gpt" and row["shop"] == shop)
        body = record["raw_response"]
        message = body["choices"][0]["message"]
        judgments = json.loads(message["content"])["judgments"]
        ids = {row["id"] for row in judgments[:count]}
        missing.update(pair for pair in record["pair_ids"] if json.loads(pair)[3] in ids)
        message["content"] = json.dumps({"judgments": judgments[count:]})
        record.update(status="partial", error="missing_candidate_answers", answers=[])
        # Change the raw attempt too, as new request logs retain every response.
        record["attempts"][-1].update(raw_response=body, answers=[], error="missing_candidate_answers")
    write_blocks(directory / "requests.jsonl", records)
    assert len(missing) == 5
    return manifest, directory, missing


def test_baseline_offline_import_and_incomplete_loader(baseline_seed, tmp_path, monkeypatch):
    manifest, source, missing = baseline_seed
    original = (source / "requests.jsonl").read_bytes()
    monkeypatch.setattr(compare.httpx, "Client", lambda *a, **k: pytest.fail("Import must be offline"))
    root = tmp_path / "baselines"
    directory, state = compare.prepare_gpt_baseline(manifest, root, import_run=source)
    assert len(state["classifications"]) == 795 and set(state["missing_pair_ids"]) == missing
    assert not state["complete"] and not (directory / "baseline.json").exists()
    assert state["accounting"]["historical"]["http_attempts"] == 40
    assert state["accounting"]["repair"]["http_attempts"] == 0
    assert (directory / "imported.jsonl").read_bytes() == original
    assert (source / "requests.jsonl").read_bytes() == original
    assert compare.prepare_gpt_baseline(manifest, root, import_run=source)[1] == state
    with pytest.raises(ValueError, match="No completed matching GPT baseline"):
        compare.load_completed_baseline(manifest, root)


def test_v2_baseline_starts_fresh_and_rejects_v1_import(relation_prepared, completed_run, tmp_path, monkeypatch):
    manifest, _ = relation_prepared
    legacy_manifest_path, legacy_run = completed_run
    legacy_manifest = json.loads(legacy_manifest_path.read_text())
    root = tmp_path / "relation-baselines"
    monkeypatch.setattr(compare.httpx, "Client", lambda *a, **k: pytest.fail("Offline preparation made HTTP call"))

    directory, state = compare.prepare_gpt_baseline(manifest, root)
    spec = compare.baseline_spec(manifest)
    assert spec["schema_version"] == compare.GPT_INPUT_V2
    assert spec["manifest_schema_version"] == compare.MANIFEST_V2
    assert state["schema_version"] == compare.GPT_BASELINE_V2
    assert not state["complete"] and state["classifications"] == []
    assert len(state["missing_pair_ids"]) == 800
    assert json.loads((directory / "spec.json").read_text()) == spec
    assert compare.baseline_spec(legacy_manifest)["schema_version"] == compare.GPT_INPUT_V1
    assert compare.baseline_spec(legacy_manifest) != spec
    with pytest.raises(ValueError, match="request mismatch"):
        compare.prepare_gpt_baseline(manifest, root, import_run=legacy_run)
    assert not (directory / "imported.jsonl").exists()


def test_v2_gpt_fingerprint_tracks_only_gpt_relation_inputs(relation_prepared):
    manifest, _ = relation_prepared
    original = compare.baseline_spec(manifest)
    changed = copy.deepcopy(manifest)
    changed["contract"]["gpt_system_prompt"] += " Changed."
    assert compare.baseline_spec(changed) != original
    changed = copy.deepcopy(manifest)
    changed["contract"]["relation_criteria"]["co_purchase"]["yes"] += " Jev-only change."
    assert compare.baseline_spec(changed) == original
    changed = copy.deepcopy(manifest)
    changed["shops"]["furniture.co.uk"]["batches"][0]["candidates"].reverse()
    assert compare.baseline_spec(changed) != original


@pytest.fixture
def relation_baseline_ready(relation_prepared, tmp_path, monkeypatch):
    manifest, jobs = relation_prepared
    root = tmp_path / "relation-baselines"
    lookup = {
        compare.canonical_json(payload): (batch, payload)
        for batch, payload in [job for work in jobs["gpt"].values() for job in work]
    }
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        batch, expected_payload = lookup[compare.canonical_json(payload)]
        return httpx.Response(200, json=relation_response_for("gpt", expected_payload, batch))

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-relation-baseline-key")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    directory, state = compare.prepare_gpt_baseline(
        manifest, root, paid=True, transport=httpx.MockTransport(handler)
    )
    assert state["complete"] and state["schema_version"] == compare.GPT_BASELINE_V2
    assert len(calls) == 40 and len(state["classifications"]) == 800
    assert all(row["identity"]["label"] == "distinct" for row in state["classifications"])
    assert all(row["relations"]["co_purchase"]["label"] == "yes" for row in state["classifications"])
    assert compare.load_completed_baseline(manifest, root) == state
    return manifest, root, directory, state


def test_baseline_completes_only_missing_batches_and_reuses_without_keys(baseline_seed, tmp_path, monkeypatch):
    manifest, source, missing = baseline_seed
    root = tmp_path / "baselines"
    _, imported = compare.prepare_gpt_baseline(manifest, root, import_run=source)
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        body = response_for("gpt", payload)
        message = body["choices"][0]["message"]
        judgments = json.loads(message["content"])["judgments"]
        for judgment in judgments:
            judgment["label"] = "skip"  # Existing positive labels must not be replaced.
        message["content"] = json.dumps({"judgments": judgments})
        return httpx.Response(200, json=body)

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-baseline-key")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    directory, state = compare.prepare_gpt_baseline(manifest, root, paid=True, transport=httpx.MockTransport(handler))
    assert state["complete"] and len(calls) == 2 and len(state["classifications"]) == 800
    rows = {row["pair_id"]: row for row in state["classifications"]}
    assert all(rows[row["pair_id"]] == row for row in imported["classifications"])
    assert all(rows[identity]["label"] == "skip" for identity in missing)
    assert state["accounting"]["historical"] == imported["accounting"]["historical"]
    assert state["accounting"]["repair"]["http_attempts"] == 2
    assert len(state["accounting"]["repair"]["group_wall_times"]) == 2
    expected = [job["request"] for job in compare.baseline_spec(manifest)["requests"]
                if any(item["pair_id"] in missing for item in job["batch"]["candidates"])]
    assert calls == expected
    before = {str(file.relative_to(directory)): file.read_bytes() for file in directory.rglob("*") if file.is_file()}
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setattr(compare.httpx, "Client", lambda *a, **k: pytest.fail("Completed baseline must not call GPT"))
    assert compare.load_completed_baseline(manifest, root) == state
    assert compare.prepare_gpt_baseline(manifest, root, paid=True)[1] == state
    assert {str(file.relative_to(directory)): file.read_bytes() for file in directory.rglob("*") if file.is_file()} == before


@pytest.mark.parametrize("change", ["evidence", "system_prompt", "examples", "model", "reasoning", "pair_identity", "batch_order"])
def test_baseline_fingerprint_changes_with_gpt_inputs(prepared, change):
    manifest, _ = prepared
    changed = copy.deepcopy(manifest)
    data = changed["shops"]["furniture.co.uk"]
    if change == "evidence":
        data["products"][data["batches"][0]["anchor_id"]]["text"] += " Changed evidence."
    elif change in ("system_prompt", "examples"):
        data["policy"][change] += " Changed policy."
    elif change == "model":
        changed["models"]["gpt"]["model"] = "different-model"
    elif change == "reasoning":
        changed["models"]["gpt"]["reasoning_effort"] = "low"
    elif change == "pair_identity":
        data["batches"][0]["candidates"][0]["pair_id"] += "changed"
    else:
        data["batches"][0]["candidates"].reverse()
    assert compare.baseline_spec(changed) != compare.baseline_spec(manifest)


def test_baseline_ignores_jev_configuration_and_rejects_corruption(completed_run, tmp_path, monkeypatch):
    path, source = completed_run
    manifest = json.loads(path.read_text())
    root = tmp_path / "baselines"
    directory, state = compare.prepare_gpt_baseline(manifest, root, import_run=source)
    changed = copy.deepcopy(manifest)
    changed["models"]["jev"] = {"model": "another-jev", "criteria": "a prompt-only experiment"}
    assert compare.baseline_spec(changed) == compare.baseline_spec(manifest)
    monkeypatch.setattr(compare.httpx, "Client", lambda *a, **k: pytest.fail("Load is offline-only"))
    assert compare.load_completed_baseline(changed, root) == state
    monkeypatch.setitem(compare.PRICING, "gpt", {**compare.PRICING["gpt"], "input": 123})
    assert compare.load_completed_baseline(changed, root) == state  # Preserve the baseline's dated prices.
    changed["models"]["gpt"]["model"] = "another-gpt"
    with pytest.raises(ValueError, match="No completed matching"):
        compare.load_completed_baseline(changed, root)
    damaged = copy.deepcopy(state)
    damaged["classifications"][0]["label"] = "skip"
    write_json(directory / "baseline.json", damaged)
    with pytest.raises(ValueError, match="response history changed"):
        compare.load_completed_baseline(manifest, root)


def test_baseline_rejects_mismatched_or_replaced_import(baseline_seed, tmp_path):
    manifest, source, _ = baseline_seed
    root = tmp_path / "baselines"
    directory, _ = compare.prepare_gpt_baseline(manifest, root, import_run=source)
    snapshot = (directory / "imported.jsonl").read_bytes()
    records, _ = compare.read_request_log(source / "requests.jsonl")
    batch = next(row for row in records if row["record_type"] == "batch" and row["provider"] == "gpt")
    batch["request"]["reasoning_effort"] = "low"
    write_blocks(source / "requests.jsonl", records)
    with pytest.raises(ValueError, match="request mismatch"):
        compare.prepare_gpt_baseline(manifest, root, import_run=source)
    assert (directory / "imported.jsonl").read_bytes() == snapshot
    original_records = [json.loads(line) for line in snapshot.decode().splitlines()]
    original_records[0]["started_at"] = "a different valid run"
    write_blocks(source / "requests.jsonl", original_records)
    with pytest.raises(ValueError, match="Refusing to replace immutable"):
        compare.prepare_gpt_baseline(manifest, root, import_run=source)
    assert (directory / "imported.jsonl").read_bytes() == snapshot


def test_partial_and_transport_retries_share_budget(prepared, monkeypatch):
    _, jobs = prepared
    batch, payload = jobs["gpt"]["furniture.co.uk"][0]
    full = response_for("gpt", payload)
    seed = compare.parse_answers("gpt", full, batch)[:-2]
    ids = [item["candidate_id"] for item in batch["candidates"]]
    calls, checkpoints, sleeps = [], [], []
    monkeypatch.setattr(compare.time, "sleep", sleeps.append)

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 2:
            raise httpx.ReadTimeout("synthetic", request=request)
        body = copy.deepcopy(full)
        message = body["choices"][0]["message"]
        judgments = json.loads(message["content"])["judgments"]
        wanted = set(ids[:-1]) if len(calls) == 1 else {ids[-1]}
        for row in judgments:
            row["label"] = "skip"
        message["content"] = json.dumps({"judgments": [row for row in judgments if row["id"] in wanted]})
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = compare.run_batch(client, "gpt", "furniture.co.uk", batch, payload, "dummy",
                                   initial_answers=seed, on_attempt=checkpoints.append)
    assert result["status"] == "ok" and result["error"] is None and result["retries"] == 2
    assert calls == [payload, payload, payload] and sleeps == [1, 2]
    assert len(checkpoints) == 3 and all(len(row["attempts"]) == 1 for row in checkpoints)
    assert [row["label"] for row in result["answers"]] == ["positive"] * 18 + ["skip"] * 2
    assert checkpoints[0]["raw_response"] is not None and checkpoints[1]["raw_response"] is None


def test_incomplete_baseline_resumes_only_remaining_batches(baseline_seed, tmp_path, monkeypatch):
    manifest, source, _ = baseline_seed
    root = tmp_path / "baselines"
    compare.prepare_gpt_baseline(manifest, root, import_run=source)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr(compare.time, "sleep", lambda delay: None)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(200, json=response_for("gpt", calls[-1]))
        return httpx.Response(503)

    directory, partial = compare.prepare_gpt_baseline(manifest, root, paid=True, transport=httpx.MockTransport(handler))
    assert len(calls) == 4 and len(partial["classifications"]) == 799
    assert not partial["complete"] and not (directory / "baseline.json").exists()
    old_logs = {path: path.read_bytes() for path in (directory / "repairs").glob("*.jsonl")}
    resumed = []

    def succeed(request):
        payload = json.loads(request.content)
        resumed.append(payload)
        return httpx.Response(200, json=response_for("gpt", payload))

    _, state = compare.prepare_gpt_baseline(manifest, root, paid=True, transport=httpx.MockTransport(succeed))
    assert state["complete"] and len(resumed) == 1 and resumed[0] == calls[-1]
    assert state["accounting"]["repair"]["http_attempts"] == 5
    assert all(path.read_bytes() == value for path, value in old_logs.items())
    assert compare.load_completed_baseline(manifest, root) == state


def test_baseline_lock_prevents_concurrent_paid_preparation(prepared, tmp_path, monkeypatch):
    manifest, _ = prepared
    root = tmp_path / "baselines"
    directory, _ = compare.prepare_gpt_baseline(manifest, root)
    monkeypatch.setattr(compare.httpx, "Client", lambda *a, **k: pytest.fail("Locked preparation must not send requests"))
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another process"):
            compare.prepare_gpt_baseline(manifest, root, paid=True)


def test_baseline_cli_import_needs_no_credentials(baseline_seed, tmp_path, monkeypatch, capsys):
    manifest, source, _ = baseline_seed
    monkeypatch.setattr(compare, "build_relation_manifest", lambda *args: manifest)
    monkeypatch.setattr(compare.httpx, "Client", lambda *a, **k: pytest.fail("Offline CLI import made HTTP call"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["compare.py", "--prepare-baseline", "--import-run", str(source),
                                    "--manifest", str(tmp_path / "frozen.json"), "--baseline-root", str(tmp_path / "baselines")])
    compare.main()
    output = capsys.readouterr().out
    assert "Retained 795 labels; 5 missing" in output and "No API requests made" in output


def test_late_malformed_response_does_not_erase_accumulated_answers(prepared, monkeypatch):
    _, jobs = prepared
    batch, payload = jobs["gpt"]["furniture.co.uk"][0]
    ids = [item["candidate_id"] for item in batch["candidates"]]
    calls = []
    monkeypatch.setattr(compare.time, "sleep", lambda delay: None)

    def handler(request):
        calls.append(request)
        if len(calls) == 3:
            return httpx.Response(200, text="malformed")
        body = response_for("gpt", payload)
        message = body["choices"][0]["message"]
        judgments = json.loads(message["content"])["judgments"]
        wanted = ids[:10] if len(calls) == 1 else ids[10:15]
        message["content"] = json.dumps({"judgments": [row for row in judgments if row["id"] in wanted]})
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = compare.run_batch(client, "gpt", "furniture.co.uk", batch, payload, "dummy")
    assert result["status"] == "partial" and len(result["answers"]) == 15
    assert result["error"] == "invalid_response" and result["raw_response"] is None
    assert result["served_model"] == payload["model"]
    assert compare.reextract_record(result, batch)["answers"] == result["answers"]
    assert [row["candidate_id"] for row in result["answers"]] == ids[:15]


def test_baseline_recovers_checkpoint_before_interrupted_tail(baseline_seed, tmp_path, monkeypatch):
    manifest, source, _ = baseline_seed
    root = tmp_path / "baselines"
    directory, state = compare.prepare_gpt_baseline(manifest, root, import_run=source)
    spec = compare.baseline_spec(manifest)
    missing = set(state["missing_pair_ids"])
    job = next(job for job in spec["requests"] if any(item["pair_id"] in missing for item in job["batch"]["candidates"]))
    original, _ = compare.read_request_log(source / "requests.jsonl")
    record = copy.deepcopy(next(row for row in original if row["record_type"] == "batch"
                                and row["provider"] == "gpt" and row["batch_id"] == job["batch"]["batch_id"]))
    body = response_for("gpt", job["request"])
    answers = compare.parse_answers("gpt", body, job["batch"])
    record.update(status="ok", error=None, answers=answers, raw_response=body, baseline_attempt_id="checkpoint")
    record["attempts"][-1].update(raw_response=body, answers=answers, error=None)
    repairs = directory / "repairs"
    repairs.mkdir()
    interrupted = repairs / "000000-interrupted.jsonl"
    write_blocks(interrupted, [{"record_type": "run_start", "baseline_fingerprint": state["fingerprint"]}, record])
    with interrupted.open("a") as handle:
        handle.write('{"record_type":')
    snapshot = interrupted.read_bytes()
    _, recovered = compare.prepare_gpt_baseline(manifest, root)
    assert len(recovered["classifications"]) == 799 and len(recovered["missing_pair_ids"]) == 1
    assert not recovered["accounting"]["repair"]["sessions_complete"]
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json=response_for("gpt", payload))

    _, complete = compare.prepare_gpt_baseline(manifest, root, paid=True, transport=httpx.MockTransport(handler))
    assert complete["complete"] and len(calls) == 1
    assert interrupted.read_bytes() == snapshot
    assert complete["accounting"]["repair"]["cost_estimate_usd"] is None
    assert compare.load_completed_baseline(manifest, root) == complete


def test_v2_jev_experiment_uses_immutable_relation_snapshots(
        relation_baseline_ready, tmp_path, monkeypatch, capsys):
    manifest, root, baseline_directory, baseline = relation_baseline_ready
    baseline_files = {str(path.relative_to(baseline_directory)): path.read_bytes()
                      for path in baseline_directory.rglob("*") if path.is_file()}
    calls = []

    def handler(request):
        assert request.url.host == "api.typesafe.ai"
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json=relation_response_for("jev", payload, None))

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-relation-jev-key")
    directory, failures = compare.run_jev_experiment(
        manifest, root, experiments_root=tmp_path / "relation-experiments",
        transport=httpx.MockTransport(handler),
    )
    assert failures == 0 and len(calls) == 40
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["schema_version"] == "jev-sbs-relation-report-v1"
    assert summary["run_complete"] and summary["all_pairs_classified"]
    assert summary["new_inference"]["gpt_http_attempts"] == 0
    assert summary["new_inference"]["jev_http_attempts"] == 40
    assert summary["gpt_baseline"]["classifications"] == 800
    for shop in compare.SHOPS:
        agreement = summary["agreement"][shop]
        assert agreement["paired_success"] == 400 and agreement["pairs_with_any_disagreement"] == 0
        assert all(head["agreements"] == 400 and head["agreement_rate"] == 1
                   for head in agreement["heads"].values())
        for provider in ("gpt", "jev"):
            distribution = summary["classifications"][provider][shop]
            assert distribution["identity_counts"]["distinct"] == 400
            assert distribution["relation_counts"]["co_purchase"]["yes"] == 400
            assert distribution["basket_label_counts"]["positive"] == 400
    paired = list(csv.DictReader((directory / "paired_results.csv").open()))
    assert len(paired) == 800
    assert {"gpt_identity", "jev_identity", "gpt_co_purchase", "jev_co_purchase",
            "gpt_basket_label", "jev_basket_label", "any_disagreement"} <= set(paired[0])
    assert (directory / "disagreements.csv").read_text().count("\n") == 1
    experiment = json.loads((directory / "experiment.json").read_text())
    assert experiment["schema_version"] == compare.JEV_EXPERIMENT_V2
    assert experiment["manifest_schema_version"] == compare.MANIFEST_V2
    assert experiment["relation_contract_version"] == manifest["contract"]["version"]
    assert experiment["resolver_version"] == compare.RESOLVER_VERSION
    assert experiment["jev_identity_instruction"] == manifest["contract"]["identity_instruction"]
    assert experiment["jev_identity_criteria"] == manifest["contract"]["identity_criteria"]
    assert json.loads((directory / "baseline.json").read_text()) == baseline
    validated_experiment, validated_baseline = compare.validate_v2_experiment_log(directory)
    assert validated_experiment == experiment and validated_baseline == baseline
    assert {str(path.relative_to(baseline_directory)): path.read_bytes()
            for path in baseline_directory.rglob("*") if path.is_file()} == baseline_files
    report_files = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    assert compare.write_reports(directory / "manifest.json", directory) == summary
    assert {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()} == report_files
    with monkeypatch.context() as patch:
        patch.setattr(compare, "build_relation_manifest", lambda *a, **k: pytest.fail("Report-only prepared data"))
        patch.setattr(compare.httpx, "Client", lambda *a, **k: pytest.fail("Report-only used network"))
        patch.setattr("sys.argv", ["compare.py", "--report-only", "--output-dir", str(directory)])
        compare.main()
    assert "Reports rebuilt offline" in capsys.readouterr().out
    assert {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()} == report_files

    def future_resolver(answer):
        result = compare.resolve_basket_judgment_v1(answer)
        return {**result, "resolver_version": "future-resolver", "basket_label": "skip",
                "resolution_status": "future", "resolution": "future_resolution"}

    with monkeypatch.context() as patch:
        patch.setattr(compare, "GPT_RELATION_SYSTEM_PROMPT", "future GPT prompt")
        patch.setattr(compare, "IDENTITY_INSTRUCTION", "future identity instruction")
        patch.setattr(compare, "RELATION_CRITERIA", {"future": "Jev criteria"})
        patch.setattr(compare, "RESOLVER_VERSION", "future-resolver")
        patch.setattr(compare, "RESOLVERS", {
            compare.RESOLVER_VERSION: future_resolver,
            experiment["resolver_version"]: compare.resolve_basket_judgment_v1,
        })
        assert compare.validate_v2_experiment_log(directory) == (experiment, baseline)
        assert compare.write_reports(directory / "manifest.json", directory) == summary
    with monkeypatch.context() as patch:
        patch.setattr(compare, "RESOLVERS", {"future-resolver": future_resolver})
        with pytest.raises(ValueError, match="Unsupported saved resolver version"):
            compare.write_reports(directory / "manifest.json", directory)

    log_path = directory / "requests.jsonl"
    original_log = log_path.read_bytes()
    records, _ = compare.read_request_log(log_path)
    failed = next(record for record in records if record["record_type"] == "batch")
    failed.update(status="error", error="http_422", answers=[], raw_response=None)
    failed["attempts"][-1].update(error="http_422", answers=[], raw_response=None, http_status=422)
    write_blocks(log_path, records)
    partial = compare.write_reports(directory / "manifest.json", directory)
    shop = failed["shop"]
    assert not partial["all_pairs_classified"] and partial["agreement"][shop]["unpaired"] == failed["batch_size"]
    failed_rows = [json.loads(line) for line in (directory / "jev_classifications.jsonl").read_text().splitlines()
                   if json.loads(line)["status"] != "ok"]
    assert len(failed_rows) == failed["batch_size"]
    assert all(row["error"] == "http_422" and row["identity"] is None and row["resolution"] is None
               for row in failed_rows)
    log_path.write_bytes(original_log)
    assert compare.write_reports(directory / "manifest.json", directory) == summary

    records, _ = compare.read_request_log(log_path)
    batch = next(record for record in records if record["record_type"] == "batch")
    batch["request"]["questions"]["candidate_0_identity"]["criteria"]["distinct"] = "tampered"
    write_blocks(log_path, records)
    with pytest.raises(ValueError, match="differs from experiment snapshot"):
        compare.validate_v2_experiment_log(directory)
    with pytest.raises(ValueError, match="differs from experiment snapshot"):
        compare.write_reports(directory / "manifest.json", directory)


@pytest.fixture
def experiment_ready(completed_run, tmp_path, monkeypatch):
    path, original_run = completed_run
    manifest = json.loads(path.read_text())
    root = tmp_path / "baselines"
    compare.prepare_gpt_baseline(manifest, root, import_run=original_run)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-jev-key")
    return manifest, root, original_run


def jev_only_response(request):
    assert request.url.host == "api.typesafe.ai", "A Jev experiment must never call GPT"
    return httpx.Response(200, json=response_for("jev", json.loads(request.content)))


@pytest.fixture
def saved_experiment(experiment_ready, tmp_path):
    manifest, root, _ = experiment_ready
    directory, failures = compare.run_jev_experiment(
        manifest, root, experiments_root=tmp_path / "experiments", transport=httpx.MockTransport(jev_only_response))
    assert failures == 0
    return directory


def test_objective_specific_prompts_preserve_evidence_and_gpt(prepared, monkeypatch):
    manifest, jobs = prepared
    original_spec = compare.baseline_spec(manifest)
    prompts = {}
    for shop, data in manifest["shops"].items():
        batch, payload = jobs["jev"][shop][0]
        question = payload["questions"]["candidate_0"]
        instructions, criteria = question["instructions"], question["criteria"]
        assert instructions == compare.JEV_INSTRUCTIONS[data["objective"]].format(index=0)
        assert criteria == compare.JEV_CRITERIA[data["objective"]]
        assert set(criteria) == set(compare.LABELS)
        assert instructions.index("(1)") < instructions.index("(2)") < instructions.index("(3)")
        assert instructions.index("(3)") < instructions.index("(4)") < instructions.index("(5)")
        assert instructions.index("(5)") < instructions.index("(6)")
        assert "uncertainty" in criteria["hard_negative"].lower()
        assert "variant" in criteria["skip"] and "override" in criteria["skip"]
        if data["objective"] == "style_compatibility":
            assert "STYLE-COMPATIBILITY" in instructions
            assert "distinct models" in instructions and "construction" in instructions
            assert "Same role alone is not negative" in instructions
            assert "matching chair" in criteria["positive"]
            assert "6ft mattress" in criteria["hard_negative"] and "pet bed" in criteria["hard_negative"]
            assert "Juliette bed" in criteria["skip"] and "Tetbury 2-basket bench" in criteria["skip"]
        else:
            assert "COMPLEMENTS" in instructions
            assert "same brand and same category" in instructions
            assert "meal variety" in instructions and "direct complementary function" in instructions
            assert "Kamado MEDIUM" in criteria["positive"] and "Do not reverse" in criteria["positive"]
            assert "distinct steak cut or burger" in criteria["hard_negative"]
            assert "same ribeye or tenderloin" in criteria["skip"] and "mixed protein box" in criteria["skip"]
        prompts[data["objective"]] = (instructions, criteria)
        changed = copy.deepcopy(compare.JEV_CRITERIA)
        changed[data["objective"]]["positive"] = "Another experiment"
        with monkeypatch.context() as patch:
            patch.setattr(compare, "JEV_CRITERIA", changed)
            rerendered = compare.render_request("jev", manifest["models"]["jev"], data, batch)
            assert rerendered["state"] == payload["state"]
            assert rerendered["questions"]["candidate_0"]["instructions"] == payload["questions"]["candidate_0"]["instructions"]
            assert compare.baseline_spec(manifest) == original_spec
    assert prompts["style_compatibility"] != prompts["complements"]


def test_repeated_experiments_are_isolated_and_jev_only(experiment_ready, tmp_path):
    manifest, baseline_root, original = experiment_ready
    original_bytes = {path.name: path.read_bytes() for path in original.iterdir()}
    baseline_bytes = {str(path.relative_to(baseline_root)): path.read_bytes()
                      for path in baseline_root.rglob("*") if path.is_file()}
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return jev_only_response(request)

    first, _ = compare.run_jev_experiment(manifest, baseline_root, experiments_root=tmp_path / "experiments",
                                         transport=httpx.MockTransport(handler))
    first_bytes = {path.name: path.read_bytes() for path in first.iterdir()}
    second, _ = compare.run_jev_experiment(manifest, baseline_root, experiments_root=tmp_path / "experiments",
                                          transport=httpx.MockTransport(handler))
    assert first != second and first.parent == second.parent
    assert len(calls) == 80 and set(calls) == {"api.typesafe.ai"}
    assert {path.name: path.read_bytes() for path in first.iterdir()} == first_bytes
    assert {path.name: path.read_bytes() for path in original.iterdir()} == original_bytes
    assert {str(path.relative_to(baseline_root)): path.read_bytes()
            for path in baseline_root.rglob("*") if path.is_file()} == baseline_bytes
    assert {"manifest.json", "baseline.json", "experiment.json", "requests.jsonl"} <= set(first_bytes)
    assert "imported.jsonl" not in first_bytes and not (first / "repairs").exists()
    experiment = json.loads(first_bytes["experiment.json"])
    assert experiment["jev_criteria"] == compare.JEV_CRITERIA
    assert experiment["jev_instructions"] == compare.JEV_INSTRUCTIONS
    assert experiment["jev_prompt_version"] == compare.JEV_PROMPT_VERSION
    summary = json.loads(first_bytes["summary.json"])
    assert set(summary["groups"]) == {f"jev/{shop}" for shop in compare.SHOPS}
    assert summary["new_inference"]["gpt_http_attempts"] == 0 and summary["new_inference"]["gpt_cost_usd"] == 0
    assert summary["new_inference"]["jev_http_attempts"] == 40
    assert summary["new_inference"]["jev_cost_estimate_usd"] == pytest.approx(40 * 0.0000042)
    assert summary["gpt_baseline"]["cached"] and summary["gpt_baseline"]["classifications"] == 800
    assert summary["gpt_baseline"]["accounting"]["historical"]["http_attempts"] == 40
    assert summary["all_pairs_classified"] and summary["run_complete"]
    assert all(value["paired_success"] == 400 for value in summary["agreement"].values())
    with pytest.raises(FileExistsError):
        compare.run_jev_experiment(manifest, baseline_root, first, transport=httpx.MockTransport(handler))
    assert len(calls) == 80


def test_experiment_report_uses_snapshots_not_live_jev_definitions(saved_experiment, monkeypatch, capsys):
    directory = saved_experiment
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    render = compare.render_request

    def only_gpt(*args, **kwargs):
        assert args[0] == "gpt", "Historical Jev reports must not rerender the current prompt"
        return render(*args, **kwargs)

    def deny(*args, **kwargs):
        pytest.fail("Offline reporting must not access catalog, baseline history, or network")

    monkeypatch.setattr(compare, "render_request", only_gpt)
    monkeypatch.setattr(compare, "JEV_CRITERIA", {"changed": "prompt"})
    monkeypatch.setattr(compare, "JEV_INSTRUCTIONS", "A future experiment")
    monkeypatch.setattr(compare, "JEV_PROMPT_VERSION", "future-version")
    monkeypatch.setattr(compare, "PRICING", {"changed": "rates"})
    monkeypatch.setattr(compare, "build_manifest", deny)
    monkeypatch.setattr(compare, "load_completed_baseline", deny)
    monkeypatch.setattr(compare.httpx, "Client", deny)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["compare.py", "--report-only", "--output-dir", str(directory)])
    compare.main()
    assert "Reports rebuilt offline" in capsys.readouterr().out
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original


@pytest.mark.parametrize("tamper", ["manifest", "baseline", "criteria", "pricing", "payload", "gpt_group"])
def test_experiment_reports_reject_snapshot_mismatches(saved_experiment, tamper):
    directory = saved_experiment
    original_summary = (directory / "summary.json").read_bytes()
    if tamper == "manifest":
        path = directory / "manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif tamper == "baseline":
        path = directory / "baseline.json"
        data = json.loads(path.read_text())
        data["classifications"][0]["label"] = "skip"
        write_json(path, data)
    elif tamper in ("criteria", "pricing"):
        path = directory / "experiment.json"
        data = json.loads(path.read_text())
        data["jev_criteria" if tamper == "criteria" else "pricing"] = {"tampered": True}
        write_json(path, data)
    else:
        path = directory / "requests.jsonl"
        records, _ = compare.read_request_log(path)
        if tamper == "payload":
            batch = next(record for record in records if record["record_type"] == "batch")
            batch["request"]["questions"]["candidate_0"]["criteria"]["positive"] = "Not the snapshotted criterion"
        else:
            records.insert(1, {"record_type": "group_start", "provider": "gpt", "shop": "furniture.co.uk"})
        write_blocks(path, records)
    with pytest.raises(ValueError):
        compare.write_reports(directory / "manifest.json", directory)
    assert (directory / "summary.json").read_bytes() == original_summary


def test_ready_experiment_requires_only_jev_credential_before_directory_creation(experiment_ready, tmp_path, monkeypatch):
    manifest, root, _ = experiment_ready
    monkeypatch.delenv("TYPESAFE_API_KEY")
    monkeypatch.setattr(compare.httpx, "Client", lambda *a, **k: pytest.fail("Missing credential must prevent HTTP"))
    output = tmp_path / "experiment"
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        compare.run_jev_experiment(manifest, root, output)
    assert not output.exists()


def test_default_execute_cli_calls_only_jev(experiment_ready, tmp_path, monkeypatch, capsys):
    manifest, root, _ = experiment_ready
    execute = compare.execute

    def mocked_execute(*args, **kwargs):
        assert set(args[1]) == {"jev"}
        assert set(args[4]) == {"jev"}
        kwargs["transport"] = httpx.MockTransport(jev_only_response)
        return execute(*args, **kwargs)

    monkeypatch.setattr(compare, "execute", mocked_execute)
    monkeypatch.setattr(compare, "build_relation_manifest", lambda *a: manifest)
    output = tmp_path / "cli-experiment"
    monkeypatch.setattr("sys.argv", ["compare.py", "--execute", "--baseline-root", str(root),
                                    "--manifest", str(tmp_path / "frozen.json"), "--output-dir", str(output)])
    compare.main()
    assert "No GPT requests will be made" in capsys.readouterr().out
    assert (output / "summary.json").exists()
