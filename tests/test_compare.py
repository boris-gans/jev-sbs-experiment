import copy
import hashlib
import json
import socket

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
            f"{stem}.system.txt": "Exact policy with {examples}\n",
            f"{stem}.user.txt": "Base: {anchor_id} {anchor_text}\n{candidates_json}\n",
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
