import importlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from safetensors import safe_open

import llmtraits.support as support
import llmtraits.svd as svd_module
import llmtraits.svd_job as svd_job
from llmtraits.compose import DeltaSpec
from llmtraits.manifest import load_manifest, save_manifest
from llmtraits.recipes import (
    compose_svd,
    compute_svd_weights,
    load_analysis,
)
from llmtraits.steering import (
    coefficient_table,
    resolve_steering,
    steering_recipe_adjustment,
)
from llmtraits.support import (
    MAX_TOP_BLOCKS,
    compute_block_diagnostics,
    departure_overlap_table,
    load_block_diagnostics,
    overlap_csv_path,
    plot_departure_energy,
    plot_novel_support,
    save_overlap_csv,
)
from llmtraits.svd import SVDAnalyzer

from tests.test_pipeline import (
    _write_sharded_model,
    create_model,
    load_delta_tensors,
    load_model_tensors,
    run_delta,
    tensor_names,
    tensor_shape,
)


def _build_analysis(tmp_path, offsets=((0.1,), (0.2,), (0.3,)), seeds=(51, 52, 53)):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    create_model(base_dir, seed=500, dtype=torch.float32)
    delta_dirs = []
    target_dirs = []
    manifests = []
    for idx, (seed, off) in enumerate(zip(seeds, [o[0] for o in offsets])):
        target_dir = tmp_path / f"target_{idx}"
        target_dir.mkdir()
        create_model(target_dir, seed=seed, dtype=torch.float32, offset=off)
        delta_dir = tmp_path / f"delta_{idx}"
        manifests.append(run_delta(base_dir, target_dir, delta_dir, dtype="float32"))
        delta_dirs.append(delta_dir)
        target_dirs.append(target_dir)
    analyzer = SVDAnalyzer()
    for idx, delta_dir in enumerate(delta_dirs):
        analyzer.add_model(str(delta_dir), label=f"m{idx}")
    analyzer.compute()
    analysis_dir = tmp_path / "analysis"
    analyzer.save(analysis_dir)
    return analyzer, analysis_dir, delta_dirs, target_dirs, manifests


def _build_uncomputed_analyzer(tmp_path, n_models=2):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    create_model(base_dir, seed=700, dtype=torch.float32)
    analyzer = SVDAnalyzer()
    for idx in range(n_models):
        target_dir = tmp_path / f"target_{idx}"
        target_dir.mkdir()
        create_model(
            target_dir,
            seed=701 + idx,
            dtype=torch.float32,
            offset=0.1 * (idx + 1),
        )
        delta_dir = tmp_path / f"delta_{idx}"
        run_delta(base_dir, target_dir, delta_dir, dtype="float32")
        analyzer.add_model(str(delta_dir), label=f"m{idx}")
    return analyzer


# ─────────────────────────────────────────────────────────────────────────────
# 1. coefficient alignment / energy share
# ─────────────────────────────────────────────────────────────────────────────
def test_coefficient_alignment_energy_share(tmp_path):
    analyzer, analysis_dir, *_ = _build_analysis(tmp_path)
    analysis = load_analysis(str(analysis_dir))
    diag = load_block_diagnostics(str(analysis_dir))
    rows = departure_overlap_table(analysis, diag, "m0")
    coeffs = analysis["model_coefficients"][0]
    norm_sq = float(analysis["norms_sq"][0])
    norm = np.sqrt(norm_sq)
    for k, row in enumerate(rows):
        a = float(coeffs[k])
        assert row["signed_alignment"] == pytest.approx(a / max(norm, 1e-12))
        assert row["departure_energy_share"] == pytest.approx(a * a / max(norm_sq, 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# 2. known block pattern
# ─────────────────────────────────────────────────────────────────────────────
def _patch_loader(monkeypatch, table):
    @contextmanager
    def fake_open(models, name):
        handles = [model["delta_dir"] for model in models]
        shape = (len(table[(handles[0], name)]),)
        yield handles, shape

    def fake_read(handle, name, shape, start, end):
        return np.asarray(table[(handle, name)][start:end], dtype=np.float32)

    monkeypatch.setattr(support, "open_model_tensor_slices", fake_open)
    monkeypatch.setattr(support, "read_flat_slice", fake_read)


def test_block_diagnostics_known_pattern(monkeypatch):
    name = "model.layers.0.mlp.gate_proj.weight"
    table = {
        ("d0", name): [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ("d1", name): [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    }
    _patch_loader(monkeypatch, table)
    models = [
        {"delta_dir": "d0", "label": "m0", "manifest": None},
        {"delta_dir": "d1", "label": "m1", "manifest": None},
    ]
    left = np.array([[1.0], [0.0]])  # component 0 loads on model 0 only
    sigma = np.array([1.0])
    arch_labels = list(support._ARCH_COMPONENTS) + ["other"]
    arch_lookup = {n: i for i, n in enumerate(arch_labels)}
    diag = compute_block_diagnostics(
        shared_names=[name],
        models=models,
        left_vectors=left,
        singular_values=sigma,
        layer_labels=["layer_0", "global"],
        arch_labels=arch_labels,
        layer_lookup={0: 0},
        arch_lookup=arch_lookup,
        block_size=4,
    )
    # p=1 (m1) departure, component 0: block0 novel (comp 4, dep 0), block1 shared.
    assert diag["novel_support_frac"][1, 0] == pytest.approx(4.0)
    assert diag["shared_support_frac"][1, 0] == pytest.approx(4.0)
    assert diag["material_total"][1, 0] == pytest.approx(8.0)
    assert diag["sign_conflict_frac"][1, 0] == pytest.approx(0.0)
    # layer/arch novel energy sums equal novel support
    assert diag["novel_layer_energy"][1, 0].sum() == pytest.approx(4.0)
    assert diag["novel_arch_energy"][1, 0].sum() == pytest.approx(4.0)
    gate_pos = arch_lookup["gate_proj"]
    assert diag["novel_arch_energy"][1, 0, gate_pos] == pytest.approx(4.0)
    # model 0 has strong dep everywhere -> not novel for itself
    assert diag["novel_support_frac"][0, 0] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. save/load no pickle
# ─────────────────────────────────────────────────────────────────────────────
def test_block_diagnostics_save_load_no_pickle(tmp_path):
    analyzer, analysis_dir, *_ = _build_analysis(tmp_path)
    npz_path = analysis_dir / "svd_block_diagnostics.npz"
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=False) as data:
        assert data["novel_support_frac"].shape == (3, analyzer.results["n_components"])
        assert data["sign_conflict_frac"].shape == (3, analyzer.results["n_components"])
    diag = load_block_diagnostics(str(analysis_dir))
    assert diag is not None
    assert diag["novel_support_frac"].shape[0] == 3
    assert diag["schema_version"] == "1.0"


# ─────────────────────────────────────────────────────────────────────────────
# 4. top blocks bounded + deterministic
# ─────────────────────────────────────────────────────────────────────────────
def test_top_blocks_bounded_deterministic(monkeypatch):
    name = "model.layers.0.mlp.gate_proj.weight"
    n = 200
    table = {
        ("d0", name): [1.0] * n,
        ("d1", name): [0.0] * n,
    }
    _patch_loader(monkeypatch, table)
    models = [
        {"delta_dir": "d0", "label": "m0", "manifest": None},
        {"delta_dir": "d1", "label": "m1", "manifest": None},
    ]
    arch_labels = list(support._ARCH_COMPONENTS) + ["other"]
    kwargs = dict(
        shared_names=[name],
        models=models,
        left_vectors=np.array([[1.0], [0.0]]),
        singular_values=np.array([1.0]),
        layer_labels=["layer_0", "global"],
        arch_labels=arch_labels,
        layer_lookup={0: 0},
        arch_lookup={n2: i for i, n2 in enumerate(arch_labels)},
        block_size=1,
    )
    diag_a = compute_block_diagnostics(**kwargs)
    diag_b = compute_block_diagnostics(**kwargs)
    tops_a = [r for r in diag_a["top_blocks"] if r["departure_label"] == "m1"]
    assert len(tops_a) == MAX_TOP_BLOCKS
    tops_b = [r for r in diag_b["top_blocks"] if r["departure_label"] == "m1"]
    assert [r["block_offset"] for r in tops_a] == [r["block_offset"] for r in tops_b]


def test_svd_resumes_gram_and_reuses_localization(tmp_path, monkeypatch):
    analyzer = _build_uncomputed_analyzer(tmp_path)
    progress_dir = tmp_path / "analysis"
    original = svd_module.streamed_tensor_gram
    calls = {"count": 0}

    def interrupt(models, name, chunk_size):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated interruption")
        return original(models, name, chunk_size)

    monkeypatch.setattr(svd_module, "streamed_tensor_gram", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        analyzer.compute(progress_dir=str(progress_dir))

    gram_dir = progress_dir / "svd_checkpoints" / "gram"
    assert len(list(gram_dir.glob("*.npz"))) == 1
    assert len(list(gram_dir.glob("*.json"))) == 1

    resumed_calls = {"count": 0}

    def count_remaining(models, name, chunk_size):
        resumed_calls["count"] += 1
        return original(models, name, chunk_size)

    monkeypatch.setattr(svd_module, "streamed_tensor_gram", count_remaining)
    shared_count = len(analyzer._find_shared_tensors()[0])
    analyzer.compute(progress_dir=str(progress_dir))
    assert resumed_calls["count"] == shared_count - 1

    def localization_must_not_reread(*args, **kwargs):
        raise AssertionError("completed Gram checkpoints were not reused")

    monkeypatch.setattr(
        svd_module, "streamed_tensor_gram", localization_must_not_reread
    )
    analyzer.compute(progress_dir=str(progress_dir))


def test_svd_resumes_block_diagnostics(tmp_path, monkeypatch):
    analyzer = _build_uncomputed_analyzer(tmp_path)
    progress_dir = tmp_path / "analysis"
    original = support.compute_block_diagnostics
    calls = {"count": 0}

    def interrupt(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated diagnostic interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(support, "compute_block_diagnostics", interrupt)
    with pytest.raises(RuntimeError, match="simulated diagnostic interruption"):
        analyzer.compute(progress_dir=str(progress_dir))

    diagnostic_dir = progress_dir / "svd_checkpoints" / "block_diagnostics"
    assert len(list(diagnostic_dir.glob("*.npz"))) == 1
    assert len(list(diagnostic_dir.glob("*.json"))) == 1

    resumed_calls = {"count": 0}

    def count_remaining(*args, **kwargs):
        resumed_calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(support, "compute_block_diagnostics", count_remaining)
    shared_count = len(analyzer._find_shared_tensors()[0])
    analyzer.compute(progress_dir=str(progress_dir))
    assert resumed_calls["count"] == shared_count - 1


def test_svd_checkpoint_identity_is_strict(tmp_path):
    analyzer = _build_uncomputed_analyzer(tmp_path)
    progress_dir = tmp_path / "analysis"
    analyzer.compute(progress_dir=str(progress_dir))

    changed = SVDAnalyzer()
    for model in analyzer.models:
        changed.add_model(model["delta_dir"], label=f"changed-{model['label']}")
    with pytest.raises(ValueError, match="checkpoint identity"):
        changed.compute(progress_dir=str(progress_dir))


def test_svd_launcher_is_detached_and_persistent(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text('{"models": []}', encoding="utf-8")
    output_dir = tmp_path / "analysis"
    captured = {}

    class FakeProcess:
        pid = 4242

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(svd_job.subprocess, "Popen", fake_popen)
    pid, log_path = svd_job.launch_svd(str(registry), str(output_dir))
    assert pid == 4242
    assert Path(log_path).exists()
    if svd_job.sys.platform == "win32":
        assert (
            captured["kwargs"]["creationflags"]
            & svd_job.subprocess.DETACHED_PROCESS
        )
    else:
        assert captured["kwargs"]["start_new_session"] is True
    state = json.loads((output_dir / "svd_job.json").read_text(encoding="utf-8"))
    assert state["pid"] == 4242


# ─────────────────────────────────────────────────────────────────────────────
# 5. donor global recipe unchanged
# ─────────────────────────────────────────────────────────────────────────────
def test_donor_global_recipe_unchanged(tmp_path):
    _, analysis_dir, *_ = _build_analysis(tmp_path)
    analysis = load_analysis(str(analysis_dir))
    base_recipe = {
        "departure_label": "m0",
        "adjustments": [{"component": 1, "donor_label": "m1", "gamma": 0.5}],
    }
    scoped_recipe = {
        "departure_label": "m0",
        "adjustments": [
            {"component": 1, "donor_label": "m1", "gamma": 0.5, "scope": "global"}
        ],
    }
    w1 = compute_svd_weights(analysis, base_recipe)
    w2 = compute_svd_weights(analysis, scoped_recipe)
    assert np.allclose(w1, w2)


# ─────────────────────────────────────────────────────────────────────────────
# 6. steering delta/target same result
# ─────────────────────────────────────────────────────────────────────────────
def test_steering_delta_target_same_result(tmp_path):
    _, analysis_dir, *_ = _build_analysis(tmp_path)
    analysis = load_analysis(str(analysis_dir))
    current = float(analysis["model_coefficients"][0, 0])
    delta_val = 0.37
    target_val = current + delta_val
    assert resolve_steering(current, "delta", delta_val) == pytest.approx(
        resolve_steering(current, "target", target_val)
    )
    adj_d = steering_recipe_adjustment(1, current, "delta", delta_val)
    adj_t = steering_recipe_adjustment(1, current, "target", target_val)
    assert adj_d["target_coefficient"] == pytest.approx(adj_t["target_coefficient"])
    # multiplier mode
    assert resolve_steering(2.0, "multiplier", 3.0) == pytest.approx(6.0)
    # no-change returns None
    assert steering_recipe_adjustment(1, current, "delta", 0.0) is None
    table = coefficient_table(analysis, "m0", {1: target_val})
    assert table[0]["proposed_coefficient"] == pytest.approx(target_val)
    assert table[0]["changed"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 7. novel support composition
# ─────────────────────────────────────────────────────────────────────────────
def _expected_novel(analysis, delta_dirs, manifests, target_dirs, departure_idx, k, delta_a, block_sz, novelty_ratio, min_support):
    """Reference implementation using float32 arithmetic to match the streaming composer."""
    sigma_k = float(analysis["singular_values"][k])
    u_k = analysis["left_vectors"][:, k]
    dep_tensors = load_delta_tensors(delta_dirs[departure_idx], manifests[departure_idx])
    all_deltas = [load_delta_tensors(d, m) for d, m in zip(delta_dirs, manifests)]
    departure_full = load_model_tensors(target_dirs[departure_idx])
    expected = {}
    for name in sorted(manifests[departure_idx].tensors):
        # Keep base in float64 for high-precision reference; accumulation uses float32
        # to match the streaming composer's precision and block-selection decisions.
        base = departure_full[name].to(torch.float64).reshape(-1).numpy()
        dep_delta = dep_tensors[name].to(torch.float32).reshape(-1).numpy()
        v = np.zeros(dep_delta.shape[0], dtype=np.float32)
        for i, dt in enumerate(all_deltas):
            v += np.float32(float(u_k[i])) * dt[name].to(torch.float32).reshape(-1).numpy()
        v /= np.float32(sigma_k)
        add = np.zeros_like(v)
        off = 0
        while off < len(v):
            end = min(off + block_sz, len(v))
            vb = v[off:end]
            db = dep_delta[off:end]
            comp_e = float(vb @ vb)
            dep_e = float(db @ db)
            if comp_e > min_support and dep_e <= novelty_ratio ** 2 * comp_e * sigma_k ** 2:
                add[off:end] = np.float32(delta_a) * vb
            off = end
        expected[name] = base + add
    return expected


def test_novel_support_composition(tmp_path):
    analyzer, analysis_dir, delta_dirs, target_dirs, manifests = _build_analysis(tmp_path)
    analysis = load_analysis(str(analysis_dir))
    departure_idx = 0
    k = 0
    gamma = 1.0
    donor_idx = 1
    sigma_k = float(analysis["singular_values"][k])
    donor_coeff = float(analysis["model_coefficients"][donor_idx, k])
    dep_coeff = float(analysis["model_coefficients"][departure_idx, k])
    delta_a = gamma * (donor_coeff - dep_coeff)
    block_sz = 5
    novelty_ratio = 0.5
    min_support = 1e-9
    output_dir = tmp_path / "novel_out"
    recipe = {
        "schema": "svd_recipe",
        "version": "1.0",
        "analysis_dir": str(analysis_dir),
        "departure_label": "m0",
        "departure_path": str(target_dirs[departure_idx]),
        "output_path": str(output_dir),
        "dtype": "float32",
        "missing_policy": "error",
        "adjustments": [
            {
                "component": 1,
                "donor_label": "m1",
                "gamma": gamma,
                "scope": "novel_support",
                "novelty_ratio": novelty_ratio,
                "block_size": block_sz,
                "min_candidate_support": min_support,
            }
        ],
    }
    recipe_path = tmp_path / "novel_recipe.json"
    with open(recipe_path, "w", encoding="utf-8") as handle:
        json.dump(recipe, handle)
    compose_svd(str(recipe_path))
    composed = load_model_tensors(output_dir)
    expected = _expected_novel(
        analysis, delta_dirs, manifests, target_dirs, departure_idx, k,
        delta_a, block_sz, novelty_ratio, min_support,
    )
    for name, exp in expected.items():
        got = composed[name].to(torch.float64).reshape(-1).numpy()
        assert np.allclose(got, exp, atol=1e-4), name
    prov = json.loads((output_dir / "provenance.json").read_text())
    assert prov["novel_support"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 8. novel composer resume after corrupt shard
# ─────────────────────────────────────────────────────────────────────────────
def test_novel_composer_resume_corrupt_shard(tmp_path):
    analyzer, analysis_dir, delta_dirs, target_dirs, manifests = _build_analysis(tmp_path)
    _write_sharded_model(target_dirs[0])
    output_dir = tmp_path / "novel_resume_out"
    recipe = {
        "schema": "svd_recipe",
        "version": "1.0",
        "analysis_dir": str(analysis_dir),
        "departure_label": "m0",
        "departure_path": str(target_dirs[0]),
        "output_path": str(output_dir),
        "dtype": "float32",
        "missing_policy": "error",
        "adjustments": [
            {
                "component": 1,
                "donor_label": "m1",
                "gamma": 1.0,
                "scope": "novel_support",
                "block_size": 5,
                "novelty_ratio": 0.5,
                "min_candidate_support": 1e-9,
            }
        ],
    }
    recipe_path = tmp_path / "novel_resume.json"
    with open(recipe_path, "w", encoding="utf-8") as handle:
        json.dump(recipe, handle)
    compose_svd(str(recipe_path))
    from llmtraits.utils import sha256_file

    shard_paths = sorted(output_dir.glob("model-*.safetensors"))
    assert len(shard_paths) == 2
    original_shas = {p.name: sha256_file(str(p)) for p in shard_paths}
    original_times = {p.name: p.stat().st_mtime_ns for p in shard_paths}
    corrupt_path, good_path = shard_paths[0], shard_paths[1]
    time.sleep(0.02)
    with open(corrupt_path, "wb") as handle:
        handle.write(b"garbage")
    compose_svd(str(recipe_path))
    assert sha256_file(str(corrupt_path)) == original_shas[corrupt_path.name]
    assert corrupt_path.stat().st_mtime_ns != original_times[corrupt_path.name]
    assert good_path.stat().st_mtime_ns == original_times[good_path.name]


# ─────────────────────────────────────────────────────────────────────────────
# 9. fingerprint validation strict
# ─────────────────────────────────────────────────────────────────────────────
def test_analysis_fingerprint_validation(tmp_path):
    analyzer, analysis_dir, delta_dirs, target_dirs, manifests = _build_analysis(tmp_path)
    manifest = load_manifest(str(delta_dirs[0]))
    manifest.base_fingerprint = "tampered"
    save_manifest(str(delta_dirs[0]), manifest)
    recipe = {
        "schema": "svd_recipe",
        "version": "1.0",
        "analysis_dir": str(analysis_dir),
        "departure_label": "m0",
        "departure_path": str(target_dirs[0]),
        "output_path": str(tmp_path / "fp_out"),
        "dtype": "float32",
        "missing_policy": "error",
        "adjustments": [{"component": 1, "donor_label": "m1", "gamma": 0.5}],
    }
    recipe_path = tmp_path / "fp_recipe.json"
    with open(recipe_path, "w", encoding="utf-8") as handle:
        json.dump(recipe, handle)
    with pytest.raises(ValueError, match="BASE fingerprints differ"):
        compose_svd(str(recipe_path), dry_run=True)


# ─────────────────────────────────────────────────────────────────────────────
# 10. block config validation
# ─────────────────────────────────────────────────────────────────────────────
def test_block_config_validation(tmp_path):
    from llmtraits.recipes import validate_recipe

    _, analysis_dir, _, target_dirs, _ = _build_analysis(tmp_path)
    recipe = {
        "schema": "svd_recipe",
        "version": "1.0",
        "analysis_dir": str(analysis_dir),
        "departure_label": "m0",
        "departure_path": str(target_dirs[0]),
        "output_path": str(tmp_path / "bad_block_out"),
        "dtype": "float32",
        "missing_policy": "error",
        "adjustments": [
            {
                "component": 1,
                "donor_label": "m1",
                "gamma": 1.0,
                "scope": "novel_support",
                "block_size": 12345,
            },
            {
                "component": 2,
                "donor_label": "m1",
                "gamma": 1.0,
                "scope": "novel_support",
                "block_size": 6789,
            },
        ],
    }
    with pytest.raises(ValueError, match="block_size must match"):
        validate_recipe(recipe)


# ─────────────────────────────────────────────────────────────────────────────
# 11. gui tabs import
# ─────────────────────────────────────────────────────────────────────────────
def test_gui_tabs_import():
    try:
        import streamlit  # noqa: F401
    except ImportError:
        pytest.skip("streamlit not installed")
    module = importlib.import_module("llmtraits.gui")
    for fn in ("_render_donor_tab", "_render_steering_tab", "_render_overlap_tab", "_render_build_tab", "main"):
        assert hasattr(module, fn)


@pytest.mark.parametrize(
    "argv",
    [
        ["gui.py", "--analysis", r"D:\work\analysis", "--port", "8502"],
        ["gui.py", "--", "--analysis", r"D:\work\analysis", "--port", "8502"],
    ],
)
def test_gui_parses_streamlit_arguments(monkeypatch, argv):
    module = importlib.import_module("llmtraits.gui")
    monkeypatch.setattr(module.sys, "argv", argv)
    args = module._parse_args()
    assert args.analysis == r"D:\work\analysis"
    assert args.port == 8502


# ─────────────────────────────────────────────────────────────────────────────
# 12. new plots under 3000px
# ─────────────────────────────────────────────────────────────────────────────
def test_plots_under_3000px_new(tmp_path):
    _, analysis_dir, *_ = _build_analysis(tmp_path)
    analysis = load_analysis(str(analysis_dir))
    diag = load_block_diagnostics(str(analysis_dir))
    p1 = tmp_path / "dep_energy.png"
    p2 = tmp_path / "novel_support.png"
    plot_departure_energy(analysis, diag, "m0", str(p1))
    plot_novel_support(analysis, diag, str(p2))
    for path in (p1, p2):
        image = plt.imread(str(path))
        assert image.shape[0] <= 3000
        assert image.shape[1] <= 3000


# ─────────────────────────────────────────────────────────────────────────────
# 13. overlap csv columns
# ─────────────────────────────────────────────────────────────────────────────
def test_overlap_csv_columns(tmp_path):
    _, analysis_dir, *_ = _build_analysis(tmp_path)
    csv_path = Path(overlap_csv_path(str(analysis_dir)))
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    required = {
        "model",
        "component",
        "coefficient",
        "signed_alignment",
        "departure_energy_share",
        "explained_ensemble_energy",
        "shared_support_fraction",
        "novel_support_fraction",
        "sign_conflict_fraction",
        "dominant_layers",
        "dominant_arch",
    }
    assert required.issubset(set(header))


# ─────────────────────────────────────────────────────────────────────────────
# 14. load_analysis exposes target_paths
# ─────────────────────────────────────────────────────────────────────────────
def test_load_analysis_target_paths(tmp_path):
    _, analysis_dir, _, target_dirs, _ = _build_analysis(tmp_path)
    analysis = load_analysis(str(analysis_dir))
    assert "target_paths" in analysis
    assert len(analysis["target_paths"]) == 3
    assert "base_paths" in analysis


# ─────────────────────────────────────────────────────────────────────────────
# 15. combined global + novel_support recipe
# ─────────────────────────────────────────────────────────────────────────────
def test_combined_global_novel_recipe(tmp_path):
    """Mixed recipe: global component 1 + novel_support component 2.

    Verifies the exact expected sum and that the combined output differs from
    both the global-only and the novel-only outputs.
    """
    analyzer, analysis_dir, delta_dirs, target_dirs, manifests = _build_analysis(tmp_path)
    analysis = load_analysis(str(analysis_dir))
    departure_idx = 0
    block_sz = 5
    novelty_ratio = 0.5
    min_support = 1e-9

    # --- recipe parameters ---
    k_global = 0          # component 1 (0-indexed)
    gamma_global = 0.5
    k_novel = 1           # component 2 (0-indexed)
    gamma_novel = 1.0
    donor_novel_idx = 2   # m2

    sigma_novel = float(analysis["singular_values"][k_novel])
    donor_coeff = float(analysis["model_coefficients"][donor_novel_idx, k_novel])
    dep_coeff = float(analysis["model_coefficients"][departure_idx, k_novel])
    delta_a_novel = gamma_novel * (donor_coeff - dep_coeff)

    # Compute global weights for the global-only sub-recipe
    global_only_recipe = {
        "departure_label": "m0",
        "adjustments": [
            {"component": k_global + 1, "donor_label": "m1", "gamma": gamma_global,
             "scope": "global"},
        ],
    }
    global_weights = compute_svd_weights(analysis, global_only_recipe)

    # Expected: combined = (departure + novel_masked) + global_deltas
    #   _expected_novel gives departure + novel_masked for the novel-only component
    expected_novel_part = _expected_novel(
        analysis, delta_dirs, manifests, target_dirs,
        departure_idx, k_novel, delta_a_novel,
        block_sz, novelty_ratio, min_support,
    )

    all_delta_tensors = [
        load_delta_tensors(d, m) for d, m in zip(delta_dirs, manifests)
    ]
    global_deltas = {}
    for name in expected_novel_part:
        g = np.zeros(
            all_delta_tensors[0][name].reshape(-1).shape[0], dtype=np.float32
        )
        for i, (w, dt) in enumerate(zip(global_weights, all_delta_tensors)):
            if abs(float(w)) > 1e-12 and name in dt:
                g += np.float32(float(w)) * dt[name].to(torch.float32).reshape(-1).numpy()
        global_deltas[name] = g

    expected_combined = {
        name: exp_nov + global_deltas[name]
        for name, exp_nov in expected_novel_part.items()
    }

    # Verify the test has meaningful non-zero global and novel contributions
    assert any(np.any(np.abs(g) > 1e-8) for g in global_deltas.values()), (
        "Test setup produced zero global deltas – recipe has no effect"
    )

    # --- run the combined recipe ---
    combined_dir = tmp_path / "combined_out"
    combined_recipe = {
        "schema": "svd_recipe",
        "version": "1.0",
        "analysis_dir": str(analysis_dir),
        "departure_label": "m0",
        "departure_path": str(target_dirs[departure_idx]),
        "output_path": str(combined_dir),
        "dtype": "float32",
        "missing_policy": "error",
        "adjustments": [
            {
                "component": k_global + 1,
                "donor_label": "m1",
                "gamma": gamma_global,
                "scope": "global",
            },
            {
                "component": k_novel + 1,
                "donor_label": "m2",
                "gamma": gamma_novel,
                "scope": "novel_support",
                "block_size": block_sz,
                "novelty_ratio": novelty_ratio,
                "min_candidate_support": min_support,
            },
        ],
    }
    combined_recipe_path = tmp_path / "combined_recipe.json"
    with open(combined_recipe_path, "w", encoding="utf-8") as handle:
        json.dump(combined_recipe, handle)
    compose_svd(str(combined_recipe_path))
    combined_tensors = load_model_tensors(combined_dir)

    # --- run global-only ---
    global_only_dir = tmp_path / "global_only_out"
    global_only = {
        "schema": "svd_recipe", "version": "1.0",
        "analysis_dir": str(analysis_dir),
        "departure_label": "m0",
        "departure_path": str(target_dirs[departure_idx]),
        "output_path": str(global_only_dir),
        "dtype": "float32", "missing_policy": "error",
        "adjustments": [
            {"component": k_global + 1, "donor_label": "m1",
             "gamma": gamma_global, "scope": "global"},
        ],
    }
    global_only_path = tmp_path / "global_only.json"
    with open(global_only_path, "w", encoding="utf-8") as handle:
        json.dump(global_only, handle)
    compose_svd(str(global_only_path))
    global_only_tensors = load_model_tensors(global_only_dir)

    # --- run novel-only ---
    novel_only_dir = tmp_path / "novel_only_out"
    novel_only = {
        "schema": "svd_recipe", "version": "1.0",
        "analysis_dir": str(analysis_dir),
        "departure_label": "m0",
        "departure_path": str(target_dirs[departure_idx]),
        "output_path": str(novel_only_dir),
        "dtype": "float32", "missing_policy": "error",
        "adjustments": [
            {
                "component": k_novel + 1,
                "donor_label": "m2",
                "gamma": gamma_novel,
                "scope": "novel_support",
                "block_size": block_sz,
                "novelty_ratio": novelty_ratio,
                "min_candidate_support": min_support,
            },
        ],
    }
    novel_only_path = tmp_path / "novel_only.json"
    with open(novel_only_path, "w", encoding="utf-8") as handle:
        json.dump(novel_only, handle)
    compose_svd(str(novel_only_path))
    novel_only_tensors = load_model_tensors(novel_only_dir)

    # --- assertions ---
    for name, exp in expected_combined.items():
        got = combined_tensors[name].to(torch.float64).reshape(-1).numpy()
        assert np.allclose(got, exp, atol=1e-4), f"combined exact sum mismatch: {name}"

    # Combined must differ from global-only (novel part adds something)
    combined_differs_global = any(
        not torch.equal(combined_tensors[n].to(torch.float32),
                        global_only_tensors[n].to(torch.float32))
        for n in combined_tensors
    )
    assert combined_differs_global, "combined output is identical to global-only"

    # Combined must differ from novel-only (global part adds something)
    combined_differs_novel = any(
        not torch.equal(combined_tensors[n].to(torch.float32),
                        novel_only_tensors[n].to(torch.float32))
        for n in combined_tensors
    )
    assert combined_differs_novel, "combined output is identical to novel-only"

    prov = json.loads((combined_dir / "provenance.json").read_text())
    assert prov["novel_support"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 16. streaming memory-structure regression
# ─────────────────────────────────────────────────────────────────────────────
def test_novel_streaming_memory_structure(tmp_path, monkeypatch):
    """Structural regression: with K novel components the streaming implementation
    must re-read each source model's delta K times (once per component pass), not
    bulk-load all N source tensors into a list and reuse them.

    With K=2 novel components and N=3 source models the streaming code issues
    K*N = 6 reads per output tensor for the novel section plus 1 for dep_delta,
    giving total >= K*N*n_shared reads.  The old bulk-load code issued only
    N*n_shared + n_shared reads regardless of K, so its total is strictly lower.
    No RSS/memory assertions are used; only deterministic I/O counts.
    """
    from safetensors import safe_open as _orig_safe_open
    import llmtraits.recipes as _recipes

    read_count = [0]

    class _CountingHandle:
        def __init__(self, path, **kw):
            self._h = _orig_safe_open(path, **kw)
            self._is_delta = "delta_" in str(path)

        def __enter__(self):
            self._h.__enter__()
            return self

        def __exit__(self, *a):
            return self._h.__exit__(*a)

        def keys(self):
            return self._h.keys()

        def get_tensor(self, name):
            if self._is_delta:
                read_count[0] += 1
            return self._h.get_tensor(name)

    monkeypatch.setattr(_recipes, "safe_open", _CountingHandle)

    analyzer, analysis_dir, delta_dirs, target_dirs, manifests = _build_analysis(tmp_path)
    analysis = load_analysis(str(analysis_dir))

    n_models = len(analysis["delta_dirs"])  # 3
    # Tensors present in every manifest (those actually processed by the novel loop)
    shared_names = set(manifests[0].tensors)
    for m in manifests[1:]:
        shared_names &= set(m.tensors)
    n_shared = len(shared_names)

    k_novel = 2  # two novel_support adjustments

    output_dir = tmp_path / "stream_mem_out"
    recipe = {
        "schema": "svd_recipe",
        "version": "1.0",
        "analysis_dir": str(analysis_dir),
        "departure_label": "m0",
        "departure_path": str(target_dirs[0]),
        "output_path": str(output_dir),
        "dtype": "float32",
        "missing_policy": "error",
        # Pure novel recipe (no global) so the global loop issues zero delta reads
        "adjustments": [
            {
                "component": 1,
                "donor_label": "m1",
                "gamma": 1.0,
                "scope": "novel_support",
                "block_size": 5,
                "novelty_ratio": 0.5,
                "min_candidate_support": 1e-9,
            },
            {
                "component": 2,
                "donor_label": "m2",
                "gamma": 1.0,
                "scope": "novel_support",
                "block_size": 5,
                "novelty_ratio": 0.5,
                "min_candidate_support": 1e-9,
            },
        ],
    }
    recipe_path = tmp_path / "stream_mem_recipe.json"
    with open(recipe_path, "w", encoding="utf-8") as handle:
        json.dump(recipe, handle)
    compose_svd(str(recipe_path))

    # Streaming lower-bound: K passes × N models × n_shared tensors.
    # (dep_delta adds 1×n_shared more; total >= (K*N+1)*n_shared.)
    # Old bulk-load code would give only (N+1)*n_shared reads (load once, reuse K times).
    streaming_lower_bound = k_novel * n_models * n_shared
    assert read_count[0] >= streaming_lower_bound, (
        f"Expected >= {streaming_lower_bound} delta get_tensor calls for "
        f"K={k_novel} novel components × N={n_models} models × {n_shared} tensors, "
        f"but got {read_count[0]}. "
        f"Old bulk-load code would give ~{(n_models + 1) * n_shared} reads."
    )
