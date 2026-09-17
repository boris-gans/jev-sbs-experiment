import copy
import csv
import hashlib
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
    stem = compare.PROMPTS[compare.SHOPS[shop]]
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
            "recommendation.yaml": f"objective: {objective}\n",
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


@pytest.mark.parametrize("failure", ["missing", "extra", "duplicate", "label", "reason", "refusal", "truncated"])
def test_gpt_rejects_invalid_answers(prepared, failure):
    _, jobs = prepared
    batch, payload = jobs["gpt"]["furniture.co.uk"][0]
    body = response_for("gpt", payload)
    message = body["choices"][0]["message"]
    judgments = json.loads(message["content"])["judgments"]
    if failure == "missing":
        judgments.pop()
    elif failure == "extra":
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


def test_execute_flag_requires_credentials_before_network(catalogs, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    output = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["compare.py", "--data-dir", str(catalogs),
                                    "--manifest", str(tmp_path / "manifest.json"),
                                    "--output-dir", str(output), "--execute"])
    with pytest.raises(SystemExit) as result:
        compare.main()
    assert result.value.code == 1
    assert "Missing credential" in capsys.readouterr().err
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
