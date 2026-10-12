import json
import os
import shutil
import time
import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from llmtraits.compose import DeltaSpec, compose, compose_command, compose_resumable
from llmtraits.cross import CrossAnalyzer
from llmtraits.delta import DeltaComputer
from llmtraits.manifest import load_manifest, plan_shards, save_manifest
from llmtraits.recipes import compose_svd, compute_svd_weights, load_analysis
from llmtraits.scan import ModelInfo, dedup_models
from llmtraits.stats import compute_stats
from llmtraits.svd import SVDAnalyzer
from llmtraits.utils import load_tensor_index, sha256_file
from llmtraits.visualize import compute_landscapes, render_overview

HIDDEN_SIZE = 8
INTERMEDIATE_SIZE = 16
VOCAB_SIZE = 64
NUM_LAYERS = 2


def tensor_names():
    names = ["model.embed_tokens.weight"]
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}"
        names.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
            ]
        )
    names.extend(["model.norm.weight", "lm_head.weight"])
    return names


def tensor_shape(name: str, vocab_size: int):
    if name == "model.embed_tokens.weight" or name == "lm_head.weight":
        return (vocab_size, HIDDEN_SIZE)
    if name == "model.norm.weight" or "layernorm" in name:
        return (HIDDEN_SIZE,)
    if "q_proj" in name or "o_proj" in name:
        return (HIDDEN_SIZE, HIDDEN_SIZE)
    if "k_proj" in name or "v_proj" in name:
        return (4, HIDDEN_SIZE)
    if "gate_proj" in name or "up_proj" in name:
        return (INTERMEDIATE_SIZE, HIDDEN_SIZE)
    if "down_proj" in name:
        return (HIDDEN_SIZE, INTERMEDIATE_SIZE)
    raise KeyError(name)


def create_model(model_dir: Path, seed: int, dtype=torch.float32, vocab_size: int = VOCAB_SIZE, offset: float = 0.0, override=None):
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name in tensor_names():
        shape = tensor_shape(name, vocab_size)
        tensor = torch.randn(shape, generator=generator, dtype=torch.float32) + offset
        if override and name in override:
            tensor = override[name].clone().to(torch.float32)
        tensors[name] = tensor.to(dtype)
    save_file(tensors, str(model_dir / "model.safetensors"))
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": vocab_size,
    }
    with open(model_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return tensors


def load_model_tensors(model_dir: Path):
    index = load_tensor_index(str(model_dir))
    files = {os.path.basename(path): [] for path in index.values()}
    for name, path in index.items():
        files[os.path.basename(path)].append(name)
    tensors = {}
    for filename, names in files.items():
        with safe_open(str(model_dir / filename), framework="pt", device="cpu") as handle:
            for name in names:
                tensors[name] = handle.get_tensor(name)
    return tensors


def load_delta_tensors(delta_dir: Path, manifest):
    tensors = {}
    for name, record in manifest.tensors.items():
        with safe_open(str(delta_dir / record.shard_file), framework="pt", device="cpu") as handle:
            tensors[name] = handle.get_tensor(name)
    return tensors


@pytest.fixture
def synthetic_dirs(tmp_path):
    base_dir = tmp_path / "base"
    target_dir = tmp_path / "target"
    base_dir.mkdir()
    target_dir.mkdir()
    return base_dir, target_dir


def run_delta(base_dir: Path, target_dir: Path, output_dir: Path, dtype="float32", force=True, skip_landscapes=True):
    computer = DeltaComputer(str(base_dir), str(target_dir), str(output_dir), dtype=dtype, force=force, skip_landscapes=skip_landscapes)
    return computer.run()


def reconstruct_from_delta(base_dir: Path, delta_dir: Path, manifest):
    base = load_model_tensors(base_dir)
    delta = load_delta_tensors(delta_dir, manifest)
    out = {}
    for name, tensor in base.items():
        if name in delta:
            out[name] = base[name].to(torch.float32) + delta[name].to(torch.float32)
        else:
            out[name] = tensor.to(torch.float32)
    return out


def test_float32_exact_reconstruction(synthetic_dirs):
    base_dir, target_dir = synthetic_dirs
    base_override = {
        name: torch.arange(int(np.prod(tensor_shape(name, VOCAB_SIZE))), dtype=torch.float32).reshape(tensor_shape(name, VOCAB_SIZE)) / 64
        for name in tensor_names()
    }
    target_override = {name: tensor + 0.5 for name, tensor in base_override.items()}
    create_model(base_dir, seed=1, dtype=torch.float32, override=base_override)
    create_model(target_dir, seed=2, dtype=torch.float32, override=target_override)
    output_dir = base_dir.parent / "delta_exact"
    manifest = run_delta(base_dir, target_dir, output_dir, dtype="float32")
    reconstructed = reconstruct_from_delta(base_dir, output_dir, manifest)
    target = {name: tensor.to(torch.float32) for name, tensor in load_model_tensors(target_dir).items()}
    for name in target:
        assert torch.equal(reconstructed[name], target[name]), name


def test_float16_bounded_reconstruction(synthetic_dirs):
    base_dir, target_dir = synthetic_dirs
    base_override = {name: torch.arange(int(np.prod(tensor_shape(name, VOCAB_SIZE))), dtype=torch.float32).reshape(tensor_shape(name, VOCAB_SIZE)) / 64 for name in tensor_names()}
    target_override = {name: tensor + 0.125 for name, tensor in base_override.items()}
    create_model(base_dir, seed=3, dtype=torch.float16, override=base_override)
    create_model(target_dir, seed=4, dtype=torch.float16, override=target_override)
    output_dir = base_dir.parent / "delta_f16"
    manifest = run_delta(base_dir, target_dir, output_dir, dtype="float16")
    reconstructed = reconstruct_from_delta(base_dir, output_dir, manifest)
    target = {name: tensor.to(torch.float32) for name, tensor in load_model_tensors(target_dir).items()}
    errors = [float((reconstructed[name] - target[name]).abs().max().item()) for name in target]
    assert max(errors) < 1e-2


def test_resume_corrupt_shard(synthetic_dirs):
    base_dir, target_dir = synthetic_dirs
    base_override = {
        name: torch.arange(int(np.prod(tensor_shape(name, VOCAB_SIZE))), dtype=torch.float32).reshape(tensor_shape(name, VOCAB_SIZE)) / 64
        for name in tensor_names()
    }
    target_override = {name: tensor + 0.5 for name, tensor in base_override.items()}
    create_model(base_dir, seed=5, dtype=torch.float32, override=base_override)
    create_model(target_dir, seed=6, dtype=torch.float32, override=target_override)
    output_dir = base_dir.parent / "delta_resume"
    manifest = run_delta(base_dir, target_dir, output_dir, dtype="float32")
    good_shas = {name: shard.file_sha256 for name, shard in manifest.shards.items()}
    shard_names = [name for name, record in manifest.shards.items() if record.file_sha256]
    corrupt_name = sorted(shard_names)[0]
    good_name = sorted(shard_names)[-1]
    good_mtime = os.path.getmtime(output_dir / good_name)
    corrupt_mtime = os.path.getmtime(output_dir / corrupt_name)
    with open(output_dir / corrupt_name, "wb") as handle:
        handle.write(b"garbage")
    run_delta(base_dir, target_dir, output_dir, dtype="float32", force=False)
    updated = load_manifest(str(output_dir))
    assert updated.shards[corrupt_name].file_sha256 == good_shas[corrupt_name]
    assert os.path.getmtime(output_dir / corrupt_name) != corrupt_mtime
    assert os.path.getsize(output_dir / corrupt_name) > len(b"garbage")
    assert sha256_file(str(output_dir / good_name)) == good_shas[good_name]
    assert os.path.getmtime(output_dir / good_name) == pytest.approx(good_mtime)
    reconstructed = reconstruct_from_delta(base_dir, output_dir, updated)
    target = {name: tensor.to(torch.float32) for name, tensor in load_model_tensors(target_dir).items()}
    for name in target:
        assert torch.equal(reconstructed[name], target[name]), name


def test_resume_recovers_stats_missing_after_completed_shard(synthetic_dirs):
    base_dir, target_dir = synthetic_dirs
    create_model(base_dir, seed=51, dtype=torch.float32)
    create_model(target_dir, seed=52, dtype=torch.float32)
    output_dir = base_dir.parent / "delta_stats_resume"
    manifest = run_delta(base_dir, target_dir, output_dir, dtype="float32")
    expected_names = set(manifest.tensors)

    os.remove(output_dir / manifest.stats_json)
    os.remove(output_dir / manifest.stats_csv)
    manifest.stats_json = None
    manifest.stats_csv = None
    from llmtraits.manifest import save_manifest

    save_manifest(str(output_dir), manifest)
    resumed = run_delta(base_dir, target_dir, output_dir, dtype="float32", force=False)
    with open(output_dir / resumed.stats_json, "r", encoding="utf-8") as handle:
        stats_names = {item["name"] for item in json.load(handle)}
    assert stats_names == expected_names


def test_partial_shape_incompatibility(synthetic_dirs):
    base_dir, target_dir = synthetic_dirs
    create_model(base_dir, seed=7, dtype=torch.float32, vocab_size=VOCAB_SIZE)
    create_model(target_dir, seed=8, dtype=torch.float32, vocab_size=96)
    output_dir = base_dir.parent / "delta_incompat"
    manifest = run_delta(base_dir, target_dir, output_dir, dtype="float32")
    assert "model.embed_tokens.weight" in manifest.incompatible_tensors
    assert "lm_head.weight" in manifest.incompatible_tensors
    assert f"model.layers.0.self_attn.q_proj.weight" in manifest.tensors
    assert f"model.layers.1.mlp.down_proj.weight" in manifest.tensors


def test_strong_dedup(tmp_path):
    model_a = tmp_path / "model_a"
    model_b = tmp_path / "renamed_copy"
    model_a.mkdir()
    model_b.mkdir()
    create_model(model_a, seed=9, dtype=torch.float32)
    shutil.copytree(model_a, model_b, dirs_exist_ok=True)
    models = [
        ModelInfo(path=str(model_a), name="model_a", is_valid=True, n_shards=1, vocab_size=VOCAB_SIZE),
        ModelInfo(path=str(model_b), name="renamed_copy", is_valid=True, n_shards=1, vocab_size=VOCAB_SIZE),
    ]
    unique, duplicates = dedup_models(models)
    assert len(unique) == 1
    assert len(duplicates) == 1
    kept, dup = duplicates[0]
    assert {kept.path, dup.path} == {str(model_a), str(model_b)}


def test_fingerprint_cache_invalidates_in_place_rewrite(tmp_path, monkeypatch):
    from llmtraits import scan

    monkeypatch.setattr(scan, "DEFAULT_OUTPUT", str(tmp_path / "fingerprints"))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    create_model(model_dir, seed=61, dtype=torch.float32)
    model = ModelInfo(path=str(model_dir), name="model", is_valid=True, n_shards=1, vocab_size=VOCAB_SIZE)
    cache = {}
    first, size = scan._fingerprint_for(model, cache)

    checkpoint = model_dir / "model.safetensors"
    original_dir_mtime = model_dir.stat().st_mtime_ns
    create_model(model_dir, seed=62, dtype=torch.float32)
    os.utime(model_dir, ns=(original_dir_mtime, original_dir_mtime))
    assert checkpoint.stat().st_size == size
    second, _ = scan._fingerprint_for(model, cache)
    assert second != first


def test_stats_correct():
    base = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    target = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    delta = target - base
    stat = compute_stats("toy", base, target, delta)
    assert stat.delta_l2 == pytest.approx(np.sqrt(2.0))
    assert stat.cosine_sim == pytest.approx(0.0)
    assert stat.sparsity_exact == pytest.approx(0.5)


def test_plots_run_without_error(synthetic_dirs):
    base_dir, target_dir = synthetic_dirs
    create_model(base_dir, seed=10, dtype=torch.float32)
    create_model(target_dir, seed=11, dtype=torch.float32)
    output_dir = base_dir.parent / "delta_plot"
    manifest = run_delta(base_dir, target_dir, output_dir, dtype="float32", skip_landscapes=False)
    landscapes = compute_landscapes(str(output_dir), manifest)
    overview = render_overview(str(output_dir), manifest)
    assert Path(landscapes).exists()
    assert Path(overview).exists()


def test_gram_pca(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    create_model(base_dir, seed=12, dtype=torch.float32)
    delta_dirs = []
    for idx, offset in enumerate([0.1, -0.2, 0.3]):
        target_dir = tmp_path / f"target_{idx}"
        target_dir.mkdir()
        create_model(target_dir, seed=12, dtype=torch.float32, offset=offset)
        delta_dir = tmp_path / f"delta_{idx}"
        run_delta(base_dir, target_dir, delta_dir, dtype="float32")
        delta_dirs.append(delta_dir)
    analyzer = CrossAnalyzer()
    for idx, delta_dir in enumerate(delta_dirs):
        analyzer.add_model(str(delta_dir), label=f"m{idx}")
    result = analyzer.compute()
    gram = result["gram"]
    eigvals = np.linalg.eigvalsh(gram)
    assert gram.shape == (3, 3)
    assert np.all(eigvals >= -1e-6)
    assert float(np.sum(result["explained_variance_ratio"][:3])) >= 0.95
    cosine = result["cosine_similarity"]
    assert np.allclose(cosine, cosine.T, atol=1e-6)
    assert np.allclose(np.diag(cosine), 1.0, atol=1e-6)


def test_cross_skips_same_named_shape_mismatches(tmp_path):
    delta_dirs = []
    for idx, vocab_size in enumerate([VOCAB_SIZE, VOCAB_SIZE + 8]):
        base_dir = tmp_path / f"base_{idx}"
        target_dir = tmp_path / f"target_{idx}"
        delta_dir = tmp_path / f"delta_{idx}"
        base_dir.mkdir()
        target_dir.mkdir()
        create_model(base_dir, seed=120 + idx, dtype=torch.float32, vocab_size=vocab_size)
        create_model(target_dir, seed=122 + idx, dtype=torch.float32, vocab_size=vocab_size, offset=0.1)
        run_delta(base_dir, target_dir, delta_dir, dtype="float32")
        delta_dirs.append(delta_dir)
    analyzer = CrossAnalyzer()
    analyzer.add_model(str(delta_dirs[0]), "a")
    analyzer.add_model(str(delta_dirs[1]), "b")
    result = analyzer.compute()
    assert result["gram"].shape == (2, 2)
    assert np.all(np.isfinite(result["gram"]))


def test_compose(synthetic_dirs):
    base_dir, target_dir = synthetic_dirs
    base_override = {
        name: torch.arange(int(np.prod(tensor_shape(name, VOCAB_SIZE))), dtype=torch.float32).reshape(tensor_shape(name, VOCAB_SIZE)) / 64
        for name in tensor_names()
    }
    target_override = {name: tensor + 0.5 for name, tensor in base_override.items()}
    create_model(base_dir, seed=13, dtype=torch.float32, override=base_override)
    create_model(target_dir, seed=14, dtype=torch.float32, override=target_override)
    delta_dir = base_dir.parent / "delta_compose"
    run_delta(base_dir, target_dir, delta_dir, dtype="float32")
    output_dir = base_dir.parent / "composed"
    compose(str(base_dir), [DeltaSpec(delta_dir=str(delta_dir), weight=0.5)], str(output_dir), dtype="float32")
    composed = load_model_tensors(output_dir)
    base = load_model_tensors(base_dir)
    target = load_model_tensors(target_dir)
    for name in base:
        expected = (base[name].to(torch.float32) + target[name].to(torch.float32)) / 2
        assert torch.allclose(composed[name].to(torch.float32), expected, atol=0, rtol=0), name


def test_compose_missing_strict(synthetic_dirs):
    base_dir, target_dir = synthetic_dirs
    create_model(base_dir, seed=15, dtype=torch.float32)
    create_model(target_dir, seed=16, dtype=torch.float32)
    delta_dir = base_dir.parent / "delta_missing"
    manifest = run_delta(base_dir, target_dir, delta_dir, dtype="float32")
    missing_name = sorted(manifest.tensors)[0]
    record = manifest.tensors.pop(missing_name)
    with safe_open(str(delta_dir / record.shard_file), framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys() if name != missing_name}
    save_file(tensors, str(delta_dir / record.shard_file))
    from llmtraits.manifest import save_manifest

    save_manifest(str(delta_dir), manifest)
    with pytest.raises(ValueError):
        compose(str(base_dir), [DeltaSpec(delta_dir=str(delta_dir), weight=1.0)], str(base_dir.parent / "strict_out"), dtype="float32", missing_policy="error")


def test_compose_command_parses_windows_absolute_path(monkeypatch):
    captured = {}

    def fake_compose(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("llmtraits.compose.compose", fake_compose)
    args = type(
        "Args",
        (),
        {
            "delta": [r"D:\work\deltas\ModelA:0.5:ModelA"],
            "base": r"D:\work\base",
            "output": r"D:\work\output",
            "delta_dtype": "float16",
            "missing_policy": "error",
        },
    )()
    compose_command(args)
    assert captured["delta_specs"][0].delta_dir == r"D:\work\deltas\ModelA"
    assert captured["delta_specs"][0].weight == pytest.approx(0.5)
    assert captured["delta_specs"][0].label == "ModelA"


def _full_override_from_base(base_dir: Path):
    return {name: tensor.to(torch.float32).clone() for name, tensor in load_model_tensors(base_dir).items()}


def _write_sharded_model(model_dir: Path):
    tensors = load_model_tensors(model_dir)
    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        single_file.unlink()
    names = sorted(tensors)
    split = max(1, len(names) // 2)
    shard_specs = [
        ("model-00001-of-00002.safetensors", names[:split]),
        ("model-00002-of-00002.safetensors", names[split:]),
    ]
    weight_map = {}
    total_size = 0
    for filename, shard_names in shard_specs:
        shard_tensors = {name: tensors[name] for name in shard_names}
        path = model_dir / filename
        save_file(shard_tensors, str(path))
        total_size += path.stat().st_size
        for name in shard_names:
            weight_map[name] = filename
    with open(model_dir / "model.safetensors.index.json", "w", encoding="utf-8") as handle:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, handle, indent=2, sort_keys=True)


def test_svd_gram_reconstruction(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    create_model(base_dir, seed=20, dtype=torch.float32)
    delta_dirs = []
    manifests = []
    for idx, (seed, offset) in enumerate([(21, 0.1), (22, -0.2), (23, 0.3)]):
        target_dir = tmp_path / f"target_{idx}"
        target_dir.mkdir()
        create_model(target_dir, seed=seed, dtype=torch.float32, offset=offset)
        delta_dir = tmp_path / f"delta_{idx}"
        manifests.append(run_delta(base_dir, target_dir, delta_dir, dtype="float32"))
        delta_dirs.append(delta_dir)

    analyzer = SVDAnalyzer()
    for idx, delta_dir in enumerate(delta_dirs):
        analyzer.add_model(str(delta_dir), label=f"m{idx}")
    result = analyzer.compute()
    shared = set(manifests[0].tensors)
    for manifest in manifests[1:]:
        shared &= set(manifest.tensors)
    shared = sorted(shared)

    manual = np.zeros((3, 3), dtype=np.float64)
    delta_tensors = [load_delta_tensors(delta_dir, manifest) for delta_dir, manifest in zip(delta_dirs, manifests)]
    for i in range(3):
        for j in range(3):
            total = 0.0
            for name in shared:
                total += float(torch.sum(delta_tensors[i][name].to(torch.float32) * delta_tensors[j][name].to(torch.float32)).item())
            manual[i, j] = total
    assert np.allclose(result["gram"], manual, atol=1e-5)
    assert np.allclose(
        result["model_coefficients"],
        result["left_vectors"] * result["singular_values"][np.newaxis, :],
        atol=1e-8,
    )


def test_svd_coefficient_reconstruction(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    create_model(base_dir, seed=24, dtype=torch.float32)
    delta_dirs = []
    for idx, (seed, offset) in enumerate([(25, 0.1), (26, -0.2), (27, 0.3)]):
        target_dir = tmp_path / f"target_{idx}"
        target_dir.mkdir()
        create_model(target_dir, seed=seed, dtype=torch.float32, offset=offset)
        delta_dir = tmp_path / f"delta_{idx}"
        run_delta(base_dir, target_dir, delta_dir, dtype="float32")
        delta_dirs.append(delta_dir)
    analyzer = SVDAnalyzer()
    for idx, delta_dir in enumerate(delta_dirs):
        analyzer.add_model(str(delta_dir), label=f"m{idx}")
    result = analyzer.compute()
    coeff_energy = np.sum(result["model_coefficients"] ** 2, axis=1)
    assert np.allclose(result["norms_sq"], coeff_energy, atol=1e-5)


def test_svd_localization_fracs_sum_to_one(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    create_model(base_dir, seed=28, dtype=torch.float32)
    analyzer = SVDAnalyzer()
    for idx, (seed, offset) in enumerate([(29, 0.1), (30, -0.2), (31, 0.3)]):
        target_dir = tmp_path / f"target_{idx}"
        target_dir.mkdir()
        create_model(target_dir, seed=seed, dtype=torch.float32, offset=offset)
        delta_dir = tmp_path / f"delta_{idx}"
        run_delta(base_dir, target_dir, delta_dir, dtype="float32")
        analyzer.add_model(str(delta_dir), label=f"m{idx}")
    result = analyzer.compute()
    for row in result["component_layer_fracs"]:
        assert np.all(row >= -1e-12)
        assert np.isclose(np.sum(row), 1.0, atol=1e-5)
    for row in result["component_arch_fracs"]:
        assert np.all(row >= -1e-12)
        assert np.isclose(np.sum(row), 1.0, atol=1e-5)


def test_svd_unique_residual_known_cases(tmp_path):
    parallel_base = tmp_path / "parallel_base"
    parallel_base.mkdir()
    create_model(parallel_base, seed=30, dtype=torch.float32)
    base_override = _full_override_from_base(parallel_base)
    probe_name = "model.layers.0.self_attn.q_proj.weight"
    template = torch.zeros_like(base_override[probe_name])
    template[0, 0] = 1.0
    targets = []
    for idx, scale in enumerate([0.1, 0.2]):
        target_dir = tmp_path / f"parallel_target_{idx}"
        target_dir.mkdir()
        override = {name: tensor.clone() for name, tensor in base_override.items()}
        override[probe_name] = override[probe_name] + scale * template
        create_model(target_dir, seed=31 + idx, dtype=torch.float32, override=override)
        delta_dir = tmp_path / f"parallel_delta_{idx}"
        run_delta(parallel_base, target_dir, delta_dir, dtype="float32")
        targets.append(delta_dir)
    parallel = SVDAnalyzer()
    for idx, delta_dir in enumerate(targets):
        parallel.add_model(str(delta_dir), label=f"p{idx}")
    parallel_result = parallel.compute()
    assert np.allclose(parallel_result["unique_residual"], 0.0, atol=1e-6)

    ortho_base = tmp_path / "ortho_base"
    ortho_base.mkdir()
    create_model(ortho_base, seed=32, dtype=torch.float32)
    ortho_override = _full_override_from_base(ortho_base)
    ortho_targets = []
    for idx, (row, col) in enumerate([(0, 0), (0, 1)]):
        target_dir = tmp_path / f"ortho_target_{idx}"
        target_dir.mkdir()
        override = {name: tensor.clone() for name, tensor in ortho_override.items()}
        delta = torch.zeros_like(override[probe_name])
        delta[row, col] = 1.0
        override[probe_name] = override[probe_name] + delta
        create_model(target_dir, seed=33 + idx, dtype=torch.float32, override=override)
        delta_dir = tmp_path / f"ortho_delta_{idx}"
        run_delta(ortho_base, target_dir, delta_dir, dtype="float32")
        ortho_targets.append(delta_dir)
    orthogonal = SVDAnalyzer()
    for idx, delta_dir in enumerate(ortho_targets):
        orthogonal.add_model(str(delta_dir), label=f"o{idx}")
    orthogonal_result = orthogonal.compute()
    assert np.allclose(orthogonal_result["unique_residual"], 1.0, atol=1e-6)


def test_svd_support_jaccard(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    create_model(base_dir, seed=34, dtype=torch.float32)
    base_override = _full_override_from_base(base_dir)

    same_target = tmp_path / "same_target"
    same_target.mkdir()
    same_override = {name: tensor.clone() for name, tensor in base_override.items()}
    same_override["model.layers.0.self_attn.q_proj.weight"][0, 0] += 1.0
    create_model(same_target, seed=35, dtype=torch.float32, override=same_override)
    same_delta_a = tmp_path / "same_delta_a"
    same_delta_b = tmp_path / "same_delta_b"
    run_delta(base_dir, same_target, same_delta_a, dtype="float32")
    run_delta(base_dir, same_target, same_delta_b, dtype="float32")
    same_analyzer = SVDAnalyzer()
    same_analyzer.add_model(str(same_delta_a), "a")
    same_analyzer.add_model(str(same_delta_b), "b")
    same_result = same_analyzer.compute()
    assert same_result["support_jaccard"][0, 1] == pytest.approx(1.0)

    diff_target_a = tmp_path / "diff_target_a"
    diff_target_b = tmp_path / "diff_target_b"
    diff_target_a.mkdir()
    diff_target_b.mkdir()
    override_a = {name: tensor.clone() for name, tensor in base_override.items()}
    override_b = {name: tensor.clone() for name, tensor in base_override.items()}
    override_a["model.layers.0.self_attn.q_proj.weight"][0, 0] += 1.0
    override_b["lm_head.weight"][0, 0] += 1.0
    create_model(diff_target_a, seed=36, dtype=torch.float32, override=override_a)
    create_model(diff_target_b, seed=37, dtype=torch.float32, override=override_b)
    diff_delta_a = tmp_path / "diff_delta_a"
    diff_delta_b = tmp_path / "diff_delta_b"
    run_delta(base_dir, diff_target_a, diff_delta_a, dtype="float32")
    run_delta(base_dir, diff_target_b, diff_delta_b, dtype="float32")
    diff_analyzer = SVDAnalyzer()
    diff_analyzer.add_model(str(diff_delta_a), "a")
    diff_analyzer.add_model(str(diff_delta_b), "b")
    diff_result = diff_analyzer.compute()
    assert diff_result["support_jaccard"][0, 1] < 1.0


def test_recipe_gamma_zero(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    create_model(base_dir, seed=40, dtype=torch.float32)
    analyzer = SVDAnalyzer()
    target_dirs = []
    for idx, (seed, offset) in enumerate([(41, 0.1), (42, 0.2), (43, 0.3)]):
        target_dir = tmp_path / f"target_{idx}"
        target_dir.mkdir()
        create_model(target_dir, seed=seed, dtype=torch.float32, offset=offset)
        delta_dir = tmp_path / f"delta_{idx}"
        run_delta(base_dir, target_dir, delta_dir, dtype="float32")
        analyzer.add_model(str(delta_dir), label=f"m{idx}")
        target_dirs.append(target_dir)
    analysis_dir = tmp_path / "analysis"
    result = analyzer.compute()
    analyzer.save(analysis_dir)
    output_dir = tmp_path / "recipe_zero_out"
    recipe_path = tmp_path / "recipe_zero.json"
    with open(recipe_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": "svd_recipe",
                "version": "1.0",
                "analysis_dir": str(analysis_dir),
                "departure_label": "m0",
                "departure_path": str(target_dirs[0]),
                "output_path": str(output_dir),
                "dtype": "float32",
                "missing_policy": "error",
                "adjustments": [{"component": 1, "donor_label": "m1", "gamma": 0.0}],
            },
            handle,
            indent=2,
        )
    compose_svd(str(recipe_path))
    departure = load_model_tensors(target_dirs[0])
    composed = load_model_tensors(output_dir)
    for name in departure:
        assert torch.equal(composed[name].to(torch.float32), departure[name].to(torch.float32)), name
    with open(output_dir / "provenance.json", "r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    assert provenance["identity_copy"] is True
    assert provenance["deltas"] == []
    assert np.allclose(result["model_coefficients"][0], load_analysis(str(analysis_dir))["model_coefficients"][0])


def test_recipe_gamma_one_component(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    create_model(base_dir, seed=50, dtype=torch.float32)
    analyzer = SVDAnalyzer()
    delta_dirs = []
    target_dirs = []
    for idx, (seed, offset) in enumerate([(51, 0.1), (52, 0.2), (53, 0.3)]):
        target_dir = tmp_path / f"target_{idx}"
        target_dir.mkdir()
        create_model(target_dir, seed=seed, dtype=torch.float32, offset=offset)
        delta_dir = tmp_path / f"delta_{idx}"
        manifest = run_delta(base_dir, target_dir, delta_dir, dtype="float32")
        analyzer.add_model(str(delta_dir), label=f"m{idx}")
        delta_dirs.append((delta_dir, manifest))
        target_dirs.append(target_dir)
    analysis_dir = tmp_path / "analysis"
    analyzer.compute()
    analyzer.save(analysis_dir)
    recipe = {
        "schema": "svd_recipe",
        "version": "1.0",
        "analysis_dir": str(analysis_dir),
        "departure_label": "m0",
        "departure_path": str(target_dirs[0]),
        "output_path": str(tmp_path / "recipe_one_out"),
        "dtype": "float32",
        "missing_policy": "error",
        "adjustments": [{"component": 1, "donor_label": "m1", "gamma": 1.0}],
    }
    recipe_path = tmp_path / "recipe_one.json"
    with open(recipe_path, "w", encoding="utf-8") as handle:
        json.dump(recipe, handle, indent=2)
    analysis = load_analysis(str(analysis_dir))
    weights = compute_svd_weights(analysis, recipe)
    compose_svd(str(recipe_path))
    departure = load_model_tensors(target_dirs[0])
    composed = load_model_tensors(Path(recipe["output_path"]))
    delta_tensors = [load_delta_tensors(delta_dir, manifest) for delta_dir, manifest in delta_dirs]
    for name in departure:
        expected_diff = torch.zeros_like(departure[name].to(torch.float32))
        for idx, (delta_dir, manifest) in enumerate(delta_dirs):
            if name in manifest.tensors:
                expected_diff = expected_diff + float(weights[idx]) * delta_tensors[idx][name].to(torch.float32)
        actual_diff = composed[name].to(torch.float32) - departure[name].to(torch.float32)
        err = torch.linalg.vector_norm(actual_diff - expected_diff).item()
        assert err < 1e-4, name


def test_recipe_invalid_component(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    create_model(base_dir, seed=60, dtype=torch.float32)
    analyzer = SVDAnalyzer()
    target_dirs = []
    delta_dirs = []
    for idx, (seed, offset) in enumerate([(61, 0.1), (62, 0.2), (63, 0.3)]):
        target_dir = tmp_path / f"target_{idx}"
        target_dir.mkdir()
        create_model(target_dir, seed=seed, dtype=torch.float32, offset=offset)
        delta_dir = tmp_path / f"delta_{idx}"
        run_delta(base_dir, target_dir, delta_dir, dtype="float32")
        analyzer.add_model(str(delta_dir), label=f"m{idx}")
        target_dirs.append(target_dir)
        delta_dirs.append(delta_dir)
    analysis_dir = tmp_path / "analysis"
    analyzer.compute()
    analyzer.save(analysis_dir)

    bad_component = tmp_path / "bad_component.json"
    with open(bad_component, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": "svd_recipe",
                "version": "1.0",
                "analysis_dir": str(analysis_dir),
                "departure_label": "m0",
                "departure_path": str(target_dirs[0]),
                "output_path": str(tmp_path / "bad_component_out"),
                "dtype": "float32",
                "missing_policy": "error",
                "adjustments": [{"component": 99, "donor_label": "m1", "gamma": 1.0}],
            },
            handle,
            indent=2,
        )
    with pytest.raises(ValueError):
        compose_svd(str(bad_component))

    bad_departure = tmp_path / "bad_departure.json"
    with open(bad_departure, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": "svd_recipe",
                "version": "1.0",
                "analysis_dir": str(analysis_dir),
                "departure_label": "missing",
                "departure_path": str(target_dirs[0]),
                "output_path": str(tmp_path / "bad_departure_out"),
                "dtype": "float32",
                "missing_policy": "error",
                "adjustments": [{"component": 1, "donor_label": "m1", "gamma": 1.0}],
            },
            handle,
            indent=2,
        )
    with pytest.raises(ValueError):
        compose_svd(str(bad_departure))

    duplicate_component = tmp_path / "duplicate_component.json"
    duplicate_payload = {
        "schema": "svd_recipe",
        "version": "1.0",
        "analysis_dir": str(analysis_dir),
        "departure_label": "m0",
        "departure_path": str(target_dirs[0]),
        "output_path": str(tmp_path / "duplicate_component_out"),
        "dtype": "float32",
        "missing_policy": "error",
        "adjustments": [
            {"component": 1, "donor_label": "m1", "gamma": 0.5},
            {"component": 1, "donor_label": "m2", "gamma": 0.5},
        ],
    }
    with open(duplicate_component, "w", encoding="utf-8") as handle:
        json.dump(duplicate_payload, handle, indent=2)
    with pytest.raises(ValueError, match="appears more than once"):
        compose_svd(str(duplicate_component), dry_run=True)

    stale_manifest = load_manifest(str(delta_dirs[0]))
    stale_manifest.base_fingerprint = "different-base"
    save_manifest(str(delta_dirs[0]), stale_manifest)
    valid_recipe = dict(duplicate_payload)
    valid_recipe["adjustments"] = [{"component": 1, "donor_label": "m1", "gamma": 0.5}]
    stale_analysis = tmp_path / "stale_analysis.json"
    with open(stale_analysis, "w", encoding="utf-8") as handle:
        json.dump(valid_recipe, handle, indent=2)
    with pytest.raises(ValueError, match="BASE fingerprints differ"):
        compose_svd(str(stale_analysis), dry_run=True)


def test_compose_resumable(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    create_model(base_dir, seed=70, dtype=torch.float32)
    _write_sharded_model(base_dir)

    delta_specs = []
    for idx, offset in enumerate([0.1, 0.2]):
        target_dir = tmp_path / f"target_{idx}"
        target_dir.mkdir()
        create_model(target_dir, seed=71 + idx, dtype=torch.float32, offset=offset)
        delta_dir = tmp_path / f"delta_{idx}"
        run_delta(base_dir, target_dir, delta_dir, dtype="float32")
        delta_specs.append(DeltaSpec(delta_dir=str(delta_dir), weight=0.5, label=f"m{idx}"))

    output_dir = tmp_path / "compose_resumable_out"
    compose_resumable(str(base_dir), delta_specs, str(output_dir), dtype="float32")
    shard_paths = sorted(output_dir.glob("model-*.safetensors"))
    assert len(shard_paths) == 2
    original_shas = {path.name: sha256_file(str(path)) for path in shard_paths}
    original_times = {path.name: path.stat().st_mtime_ns for path in shard_paths}
    corrupt_path = shard_paths[0]
    good_path = shard_paths[1]
    time.sleep(0.02)
    with open(corrupt_path, "wb") as handle:
        handle.write(b"garbage")
    compose_resumable(str(base_dir), delta_specs, str(output_dir), dtype="float32")
    assert sha256_file(str(corrupt_path)) == original_shas[corrupt_path.name]
    assert corrupt_path.stat().st_mtime_ns != original_times[corrupt_path.name]
    assert good_path.stat().st_mtime_ns == original_times[good_path.name]


def test_plots_under_3000px(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    create_model(base_dir, seed=80, dtype=torch.float32)
    analyzer = SVDAnalyzer()
    for idx, (seed, offset) in enumerate([(81, 0.1), (82, 0.2), (83, 0.3)]):
        target_dir = tmp_path / f"target_{idx}"
        target_dir.mkdir()
        create_model(target_dir, seed=seed, dtype=torch.float32, offset=offset)
        delta_dir = tmp_path / f"delta_{idx}"
        run_delta(base_dir, target_dir, delta_dir, dtype="float32")
        analyzer.add_model(str(delta_dir), label=f"m{idx}")
    analyzer.compute()
    output_dir = tmp_path / "plots"
    analyzer.plot(output_dir)
    for path in output_dir.glob("*.png"):
        image = plt.imread(str(path))
        assert image.shape[0] <= 3000
        assert image.shape[1] <= 3000


def test_svd_save_load_no_pickle(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    create_model(base_dir, seed=90, dtype=torch.float32)
    analyzer = SVDAnalyzer()
    for idx, (seed, offset) in enumerate([(91, 0.1), (92, 0.2), (93, 0.3)]):
        target_dir = tmp_path / f"target_{idx}"
        target_dir.mkdir()
        create_model(target_dir, seed=seed, dtype=torch.float32, offset=offset)
        delta_dir = tmp_path / f"delta_{idx}"
        run_delta(base_dir, target_dir, delta_dir, dtype="float32")
        analyzer.add_model(str(delta_dir), label=f"m{idx}")
    result = analyzer.compute()
    output_dir = tmp_path / "analysis"
    analyzer.save(output_dir)
    with np.load(output_dir / "svd_analysis.npz", allow_pickle=False) as data:
        assert data["gram"].shape == (3, 3)
        assert data["model_coefficients"].shape == result["model_coefficients"].shape
        assert data["labels"].shape == (3,)
    with open(output_dir / "svd_analysis.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["schema_version"] == "1.0"


def test_gui_import_no_server():
    try:
        import streamlit  # noqa: F401
    except ImportError:
        pytest.skip("streamlit not installed")
    module = importlib.import_module("llmtraits.gui")
    assert hasattr(module, "main")
