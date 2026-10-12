import json
import os
import gc
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open

import llmtraits.routing as routing_module
from llmtraits.atlas import scan_local_atlas
from llmtraits.atlas_analysis import analyze_local_atlas
from llmtraits.manifest import load_manifest
from llmtraits.recipes import load_analysis
from llmtraits.routing import (
    ROUTED_SCHEMA_VERSION,
    compose_routed_recipe,
    compute_residual_coefficients,
    default_routed_recipe,
    load_route_preview,
    preview_routed_recipe,
    routed_recipe_identity,
    validate_routed_recipe,
)
from llmtraits.svd import SVDAnalyzer
from llmtraits.utils import sha256_file
from tests.test_pipeline import (
    _write_sharded_model,
    create_model,
    load_delta_tensors,
    load_model_tensors,
    run_delta,
    tensor_names,
    tensor_shape,
)


def _case(tmp_path: Path, source_dtype=torch.float32, sharded=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base_values = {}
    for index, name in enumerate(tensor_names()):
        shape = tensor_shape(name, 64)
        base_values[name] = (
            torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
            / 100.0
            + 0.5
            + index / 100.0
        )
    create_model(base_dir, 1, dtype=source_dtype, override=base_values)

    labels = ["departure", "donor_a", "donor_b", "donor_c", "donor_d"]
    delta_dirs = []
    target_dirs = []
    for model_index, label in enumerate(labels):
        overrides = {}
        for tensor_index, name in enumerate(tensor_names()):
            base = base_values[name]
            flat = torch.zeros(base.numel(), dtype=torch.float32)
            # A common departure direction plus one donor-exclusive direction.
            flat[0] = 0.20 * (1.0 + 0.03 * tensor_index)
            if model_index:
                flat[0] *= 0.5 - 0.1 * model_index
                flat[model_index] = (0.32 - 0.04 * model_index) * (
                    1.0 + 0.02 * tensor_index
                )
                # Block-local overlap makes protection non-trivial while the
                # global residual construction remains auditable.
                if base.numel() > 6:
                    flat[5] = 0.02 * ((-1) ** (tensor_index + model_index))
            overrides[name] = base + flat.reshape(base.shape)
        target = tmp_path / f"target_{model_index}"
        target.mkdir()
        create_model(target, 10 + model_index, dtype=source_dtype, override=overrides)
        delta = tmp_path / f"delta_{model_index}"
        run_delta(base_dir, target, delta, dtype="float32")
        target_dirs.append(target)
        delta_dirs.append(delta)
    if sharded:
        _write_sharded_model(target_dirs[0])

    analyzer = SVDAnalyzer(top_k=5)
    for label, delta_dir in zip(labels, delta_dirs):
        analyzer.add_model(str(delta_dir), label)
    analyzer.compute()
    analysis_dir = tmp_path / "analysis"
    analyzer.save(analysis_dir)
    analysis = load_analysis(str(analysis_dir))
    return analysis_dir, analysis, labels, delta_dirs, target_dirs


def _recipe(tmp_path, analysis_dir, analysis, target_dirs, *, dtype="float32"):
    recipe = default_routed_recipe(
        analysis,
        "departure",
        str(target_dirs[0]),
        str(tmp_path / "output"),
    )
    recipe["analysis_dir"] = str(analysis_dir)
    recipe["dtype"] = dtype
    recipe["donors"] = [
        {"label": "donor_a", "alpha": 0.2, "components": [1]},
        {"label": "donor_b", "alpha": 0.1, "components": [2]},
    ]
    recipe["routing"].update(
        {
            "fine_block_size": 17,
            "coarse_block_size": 34,
            "protection_energy_fraction": 0.0,
            "hard_exclusions": ["embed_tokens", "lm_head", "norm", "layernorm"],
            "require_coarse_stability": False,
            "max_rms_ratio": 10.0,
            "protected_component": 4,
        }
    )
    return recipe


def _write_recipe(path: Path, recipe: dict):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(recipe, handle, indent=2)


def _preplan_case(tmp_path: Path):
    analysis_dir, analysis, _, delta_dirs, targets = _case(tmp_path / "case")
    atlas_dir = tmp_path / "atlas"
    scan_local_atlas(str(analysis_dir), str(atlas_dir), block_size=17)
    derived_dir = tmp_path / "derived"
    analyze_local_atlas(
        str(atlas_dir),
        str(analysis_dir),
        str(derived_dir),
        departure_label="departure",
    )
    recipe_path = derived_dir / "candidate_recipes" / "balanced_top5.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["output_path"] = str(tmp_path / "cached-output")
    recipe["routing_path"] = str(tmp_path / "cached-routing")
    return recipe, analysis, delta_dirs, targets


def test_defaults_validation_and_label_matching(tmp_path):
    analysis = {"labels": ["My Departure", "donor-one", "donor-two"]}
    recipe = default_routed_recipe(analysis, "my departure", "d", "o")
    assert recipe["version"] == ROUTED_SCHEMA_VERSION
    assert recipe["departure_label"] == "My Departure"
    assert recipe["donors"] == []
    assert recipe["routing"]["fine_block_size"] == 262144
    assert recipe["routing"]["protection_energy_fraction"] == pytest.approx(0.90)

    analysis_dir, full_analysis, _, _, targets = _case(tmp_path / "full")
    valid = _recipe(tmp_path / "full", analysis_dir, full_analysis, targets)
    validate_routed_recipe(valid, full_analysis)
    invalid = dict(valid)
    invalid["dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="dtype"):
        validate_routed_recipe(invalid, full_analysis)

    invalid_component = json.loads(json.dumps(valid))
    invalid_component["donors"][0]["components"] = [
        int(full_analysis["n_components"]) + 1
    ]
    with pytest.raises(ValueError, match="outside"):
        validate_routed_recipe(invalid_component, full_analysis)

    invalid_alpha = json.loads(json.dumps(valid))
    invalid_alpha["donors"][0]["alpha"] = -0.1
    with pytest.raises(ValueError, match="Alpha"):
        validate_routed_recipe(invalid_alpha, full_analysis)


def test_residuals_are_sequentially_orthogonal(tmp_path):
    _, analysis, _, _, _ = _case(tmp_path)
    coefficients = compute_residual_coefficients(
        analysis, "departure", ["donor_a", "donor_b", "donor_c"]
    )
    assert coefficients.shape == (3, 5)
    gram = np.asarray(analysis["gram"], dtype=np.float64)
    departure = np.array([1, 0, 0, 0, 0], dtype=np.float64)
    for index, residual in enumerate(coefficients):
        assert residual @ gram @ departure == pytest.approx(0.0, abs=2e-6)
        for prior in coefficients[:index]:
            assert residual @ gram @ prior == pytest.approx(0.0, abs=2e-6)


def test_preview_is_deterministic_and_identity_bound(tmp_path):
    analysis_dir, analysis, _, _, targets = _case(tmp_path)
    recipe = _recipe(tmp_path, analysis_dir, analysis, targets)
    first = preview_routed_recipe(recipe)
    assert first["status"] == "complete"
    assert first == preview_routed_recipe(recipe)
    assert load_route_preview(recipe["routing_path"]) == first
    identity = routed_recipe_identity(recipe, analysis)
    assert identity["identity_hash"] == first["identity_hash"]

    changed = json.loads(json.dumps(recipe))
    changed["donors"][0]["alpha"] = 0.3
    with pytest.raises(ValueError, match="identity mismatch"):
        preview_routed_recipe(changed)


def test_conflict_and_protection_fail_closed(tmp_path):
    analysis_dir, analysis, _, _, targets = _case(tmp_path)
    conflict = _recipe(tmp_path, analysis_dir, analysis, targets)
    conflict["routing"]["conflict_threshold"] = 0.99
    conflict["routing_path"] = str(tmp_path / "conflict.routing")
    conflict_summary = preview_routed_recipe(conflict)
    assert conflict_summary["accepted"] == 0

    protected = _recipe(tmp_path, analysis_dir, analysis, targets)
    protected["routing"]["protection_energy_fraction"] = 1.0
    protected["routing_path"] = str(tmp_path / "protected.routing")
    protected_summary = preview_routed_recipe(protected)
    assert protected_summary["accepted"] == 0
    assert protected_summary["protected"] > 0


def test_output_formula_c4_and_excluded_identity(tmp_path):
    analysis_dir, analysis, _, delta_dirs, targets = _case(tmp_path)
    recipe = _recipe(tmp_path, analysis_dir, analysis, targets)
    recipe_path = tmp_path / "recipe.json"
    _write_recipe(recipe_path, recipe)
    compose_routed_recipe(str(recipe_path))

    departure = load_model_tensors(targets[0])
    output = load_model_tensors(Path(recipe["output_path"]))
    assert torch.equal(
        output["model.embed_tokens.weight"], departure["model.embed_tokens.weight"]
    )
    assert torch.equal(output["model.norm.weight"], departure["model.norm.weight"])

    with open(Path(recipe["routing_path"]) / "route_index.json", encoding="utf-8") as handle:
        route_index = json.load(handle)
    coefficients = np.asarray(route_index["residual_coefficients"], dtype=np.float64)
    manifests = [load_manifest(str(path)) for path in delta_dirs]
    deltas = [
        load_delta_tensors(path, manifest)
        for path, manifest in zip(delta_dirs, manifests)
    ]
    name = "model.layers.0.self_attn.q_proj.weight"
    record = route_index["tensors"][name]
    with np.load(
        Path(recipe["routing_path"]) / record["plan"], allow_pickle=False
    ) as plan:
        expected = departure[name].to(torch.float32).reshape(-1).numpy().copy()
        matrix = np.stack(
            [item[name].to(torch.float32).reshape(-1).numpy() for item in deltas]
        )
        k = route_index["protected_component"] - 1
        v4 = (
            np.asarray(analysis["left_vectors"], dtype=np.float64)[:, k] @ matrix
        ) / float(analysis["singular_values"][k])
        for block, (start, end) in enumerate(zip(plan["starts"], plan["ends"])):
            winner = int(plan["winner"][block])
            if winner < 0:
                continue
            start, end = int(start), int(end)
            update = (
                recipe["donors"][winner]["alpha"]
                * (1.0 - float(plan["mix"][block]))
                * (coefficients[winner] @ matrix[:, start:end])
            )
            second = int(plan["second"][block])
            if second >= 0:
                update += (
                    recipe["donors"][second]["alpha"]
                    * float(plan["mix"][block])
                    * (coefficients[second] @ matrix[:, start:end])
                )
            update -= float(plan["c4_beta"]) * v4[start:end]
            expected[start:end] += float(plan["scale"][block]) * update
    assert np.allclose(output[name].reshape(-1).numpy(), expected, atol=2e-6)
    with open(Path(recipe["output_path"]) / "provenance.json", encoding="utf-8") as handle:
        provenance = json.load(handle)
    assert abs(provenance["routed"]["achieved_c4_leakage"]) < 2e-7


def test_source_dtype_and_corrupt_shard_resume(tmp_path):
    analysis_dir, analysis, _, _, targets = _case(
        tmp_path, source_dtype=torch.float16, sharded=True
    )
    recipe = _recipe(
        tmp_path, analysis_dir, analysis, targets, dtype="source"
    )
    recipe_path = tmp_path / "recipe.json"
    _write_recipe(recipe_path, recipe)
    compose_routed_recipe(str(recipe_path))
    output_dir = Path(recipe["output_path"])
    output = load_model_tensors(output_dir)
    assert all(tensor.dtype == torch.float16 for tensor in output.values())
    del output
    gc.collect()

    with open(output_dir / "compose_state.json", encoding="utf-8") as handle:
        state = json.load(handle)
    shard_names = sorted(state["shards"])
    good = output_dir / shard_names[0]
    bad = output_dir / shard_names[-1]
    good_sha = sha256_file(str(good))
    good_mtime = os.path.getmtime(good)
    with open(bad, "wb") as handle:
        handle.write(b"corrupt")
    compose_routed_recipe(str(recipe_path))
    assert sha256_file(str(good)) == good_sha
    assert os.path.getmtime(good) == pytest.approx(good_mtime)
    with safe_open(str(bad), framework="pt", device="cpu") as handle:
        assert list(handle.keys())


def test_cached_atlas_preplan_matches_direct_and_resumes(tmp_path, monkeypatch):
    recipe, analysis, delta_dirs, _ = _preplan_case(tmp_path)
    original_safe_open = routing_module.safe_open
    delta_roots = [_normalized.resolve() for _normalized in map(Path, delta_dirs)]

    def departure_only_safe_open(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if any(root in resolved.parents for root in delta_roots):
            raise AssertionError(f"Finalization opened a delta shard: {resolved}")
        return original_safe_open(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(routing_module, "safe_open", departure_only_safe_open)
        cached_summary = preview_routed_recipe(recipe)
    assert cached_summary["finalization_source"] == (
        "departure_weights_and_cached_atlas_gram"
    )
    assert cached_summary["maximum_update_rms_ratio"] <= (
        recipe["routing"]["max_rms_ratio"] + 1e-12
    )
    assert abs(cached_summary["precast_c4_leakage"]) < 1e-10

    direct = json.loads(json.dumps(recipe))
    direct.pop("atlas_preplan")
    direct["routing_path"] = str(tmp_path / "direct-routing")
    direct["output_path"] = str(tmp_path / "direct-output")
    direct_summary = preview_routed_recipe(direct)
    assert cached_summary["c4_beta"] == pytest.approx(
        direct_summary["c4_beta"], rel=1e-9, abs=1e-11
    )

    cached_root = Path(recipe["routing_path"])
    direct_root = Path(direct["routing_path"])
    cached_index = json.loads(
        (cached_root / "route_index.json").read_text(encoding="utf-8")
    )
    direct_index = json.loads(
        (direct_root / "route_index.json").read_text(encoding="utf-8")
    )
    for name, cached_record in cached_index["tensors"].items():
        with np.load(cached_root / cached_record["plan"], allow_pickle=False) as cached:
            with np.load(
                direct_root / direct_index["tensors"][name]["plan"],
                allow_pickle=False,
            ) as direct_plan:
                np.testing.assert_array_equal(cached["winner"], direct_plan["winner"])
                np.testing.assert_array_equal(cached["second"], direct_plan["second"])
                np.testing.assert_array_equal(cached["mix"], direct_plan["mix"])
                np.testing.assert_allclose(
                    cached["scale"], direct_plan["scale"], rtol=1e-9, atol=1e-11
                )
            with np.load(
                cached_root / cached_record["metrics"], allow_pickle=False
            ) as metric:
                beta = float(cached["c4_beta"])
                corrected = np.maximum(
                    metric["raw_update_energy"]
                    - 2.0 * beta * metric["raw_c4_dot"]
                    + beta * beta * metric["c4_energy"],
                    0.0,
                )
                final_energy = cached["scale"] ** 2 * corrected
                accepted = cached["winner"] >= 0
                assert np.all(
                    final_energy[accepted]
                    <= recipe["routing"]["max_rms_ratio"] ** 2
                    * metric["weight_energy"][accepted]
                    + 1e-12
                )

    metric_path = cached_root / next(iter(cached_index["tensors"].values()))["metrics"]
    metric_mtime = metric_path.stat().st_mtime_ns
    (cached_root / "route_index.json").unlink()
    (cached_root / "route_preview.json").unlink()
    plan_path = cached_root / next(iter(cached_index["tensors"].values()))["plan"]
    plan_path.unlink()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            routing_module,
            "_read_departure_weight_energy",
            lambda *args, **kwargs: pytest.fail(
                "resumed finalization reread departure weights"
            ),
        )
        resumed = preview_routed_recipe(recipe)
    assert resumed["identity_hash"] == cached_summary["identity_hash"]
    assert metric_path.stat().st_mtime_ns == metric_mtime


def test_cached_atlas_preplan_fails_closed_on_mismatch_and_corruption(tmp_path):
    recipe, _, _, _ = _preplan_case(tmp_path)
    mismatched = json.loads(json.dumps(recipe))
    mismatched["donors"][0]["alpha"] += 0.01
    mismatched["routing_path"] = str(tmp_path / "mismatched-routing")
    with pytest.raises(ValueError, match="does not match the cached candidate"):
        preview_routed_recipe(mismatched)

    preplan_root = Path(recipe["atlas_preplan"]["path"])
    preplan_index = json.loads(
        (preplan_root / "route_index.json").read_text(encoding="utf-8")
    )
    plan_path = preplan_root / next(iter(preplan_index["tensors"].values()))["plan"]
    with open(plan_path, "ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="preplan SHA-256 mismatch"):
        preview_routed_recipe(recipe)


def test_bfloat16_source_dtype_roundtrip(tmp_path):
    analysis_dir, analysis, _, _, targets = _case(
        tmp_path, source_dtype=torch.bfloat16
    )
    recipe = _recipe(
        tmp_path, analysis_dir, analysis, targets, dtype="source"
    )
    recipe_path = tmp_path / "recipe.json"
    _write_recipe(recipe_path, recipe)
    compose_routed_recipe(str(recipe_path))
    output = load_model_tensors(Path(recipe["output_path"]))
    assert all(tensor.dtype == torch.bfloat16 for tensor in output.values())


def test_public_recipe_dispatch_and_cli(tmp_path, monkeypatch):
    analysis_dir, analysis, _, _, targets = _case(tmp_path)
    recipe = _recipe(tmp_path, analysis_dir, analysis, targets)
    recipe_path = tmp_path / "recipe.json"
    _write_recipe(recipe_path, recipe)

    from llmtraits import recipes
    from llmtraits.cli import build_parser

    recipes.validate_recipe(recipe)
    called = {}

    def fake_compose(path, dry_run=False):
        called["path"] = path
        called["dry_run"] = dry_run

    monkeypatch.setattr("llmtraits.routing.compose_routed_recipe", fake_compose)
    recipes.compose_svd(str(recipe_path), dry_run=True)
    assert called == {"path": str(recipe_path), "dry_run": True}

    args = build_parser().parse_args(["route-preview", "--recipe", str(recipe_path)])
    assert args.recipe == str(recipe_path)


def test_gui_saves_routing_plan_with_job(tmp_path):
    from llmtraits.gui import _save_recipe_for_job

    recipe = {
        "schema": "routed_recipe",
        "version": "1.0",
        "routing_path": "",
    }
    path = _save_recipe_for_job(
        recipe,
        str(tmp_path / "jobs"),
        "routed-job",
        str(tmp_path / "departure"),
        str(tmp_path / "composed"),
        str(tmp_path / "analysis"),
    )
    with open(path, encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["routing_path"] == str(tmp_path / "jobs" / "routed-job" / "routing")


def test_route_prior_bands_boost_and_damp_by_architecture_and_layer():
    donor = {
        "route_prior": 1.0,
        "bands": [
            {"architectures": ["v_proj"], "layer_range": [40, 65], "factor": 1.6},
            {"architectures": ["down_proj"], "layer_range": [60, 79], "factor": 0.5},
            {"layer_range": [0, 3], "factor": 2.0},
        ],
    }
    assert routing_module._route_prior(donor, "model.layers.50.self_attn.v_proj.weight") == 1.6
    assert routing_module._route_prior(donor, "model.layers.70.mlp.down_proj.weight") == 0.5
    assert routing_module._route_prior(donor, "model.layers.2.mlp.down_proj.weight") == 2.0
    assert routing_module._route_prior(donor, "model.layers.20.mlp.gate_proj.weight") == 1.0
    assert routing_module._route_prior(donor, "model.layers.39.self_attn.v_proj.weight") == 1.0


def test_cached_atlas_preplan_blend_mode_matches_direct(tmp_path):
    recipe, _, _, _ = _preplan_case(tmp_path)
    recipe["routing"]["winner_only"] = False
    recipe["routing"]["blend_cosine_threshold"] = -1.0
    recipe["routing"]["blend_score_ratio"] = 0.0
    recipe["routing"]["protection_energy_fraction"] = 0.0
    recipe["candidate_id"] = "balanced_top5_blend"
    derived_dir = tmp_path / "derived"
    recipe["atlas_preplan"]["path"] = str(derived_dir / "route_plans" / recipe["candidate_id"])
    from llmtraits.atlas_analysis import HARD_EXCLUSIONS, _build_preplan, _protection_threshold
    atlas_index = json.loads((tmp_path / "atlas" / "atlas_index.json").read_text(encoding="utf-8"))
    derived_index = json.loads((derived_dir / "derived_index.json").read_text(encoding="utf-8"))
    analysis = load_analysis(recipe["analysis_dir"])
    analysis["_atlas_block_size"] = int(atlas_index["block_size"])
    departure_index = [str(v) for v in analysis["labels"]].index(recipe["departure_label"])
    energies = []
    for record in derived_index["tensors"].values():
        with np.load(derived_dir / record["file"], allow_pickle=False) as data:
            architecture = str(data["architecture"].item())
            if any(v in architecture.casefold() for v in HARD_EXCLUSIONS):
                continue
            energies.append(np.asarray(data["energy"][:, departure_index], dtype=np.float64))
    threshold = _protection_threshold(
        energies, float(recipe["routing"]["protection_energy_fraction"])
    )
    (derived_dir / "candidate_recipes" / f"{recipe['candidate_id']}.json").write_text(
        json.dumps(recipe, indent=2, sort_keys=True), encoding="utf-8"
    )
    _build_preplan(
        derived_dir, tmp_path / "atlas", atlas_index, str(derived_index["identity_hash"]),
        recipe, analysis, threshold, lambda *args: None,
    )
    cached_summary = preview_routed_recipe(recipe)

    direct = json.loads(json.dumps(recipe))
    direct.pop("atlas_preplan")
    direct["routing_path"] = str(tmp_path / "direct-routing-blend")
    direct["output_path"] = str(tmp_path / "direct-output-blend")
    direct_summary = preview_routed_recipe(direct)
    assert cached_summary["c4_beta"] == pytest.approx(
        direct_summary["c4_beta"], rel=1e-9, abs=1e-11
    )

    cached_root = Path(recipe["routing_path"])
    direct_root = Path(direct["routing_path"])
    cached_index = json.loads((cached_root / "route_index.json").read_text(encoding="utf-8"))
    direct_index = json.loads((direct_root / "route_index.json").read_text(encoding="utf-8"))
    blended = 0
    for name, cached_record in cached_index["tensors"].items():
        with np.load(cached_root / cached_record["plan"], allow_pickle=False) as cached:
            with np.load(
                direct_root / direct_index["tensors"][name]["plan"], allow_pickle=False
            ) as direct_plan:
                np.testing.assert_array_equal(cached["winner"], direct_plan["winner"])
                np.testing.assert_array_equal(cached["second"], direct_plan["second"])
                np.testing.assert_allclose(cached["mix"], direct_plan["mix"], rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(cached["scale"], direct_plan["scale"], rtol=1e-9, atol=1e-11)
                blended += int(np.count_nonzero(cached["second"] >= 0))
    assert blended > 0
