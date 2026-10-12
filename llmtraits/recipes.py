import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .compose import DeltaSpec, compose_resumable
from .manifest import load_manifest
from .paths import relocated_path
from .svd import SVD_SCHEMA_VERSION
from .support import BLOCK_SIZE, MATERIAL_BLOCK_FRACTION, NOVELTY_RATIO
from .utils import ensure_dir

# Module-level alias so tests can monkeypatch it without touching safetensors itself.
try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover – optional at import time
    safe_open = None  # type: ignore[assignment]


def _read_analysis_json(analysis_dir: str) -> dict:
    path = relocated_path(analysis_dir) / "svd_analysis.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_analysis(analysis_dir: str) -> dict:
    npz_path = relocated_path(analysis_dir) / "svd_analysis.npz"
    json_payload = _read_analysis_json(analysis_dir)
    with np.load(npz_path, allow_pickle=False) as data:
        analysis = {
            "schema_version": str(data["schema_version"].item()),
            "labels": [str(item) for item in data["labels"].tolist()],
            "delta_dirs": [str(item) for item in data["delta_dirs"].tolist()],
            "n_models": int(data["n_models"].item()),
            "n_components": int(data["n_components"].item()),
            "shared_tensor_count": int(data["shared_tensor_count"].item()),
            "excluded_tensor_count": int(data["excluded_tensor_count"].item()),
            "excluded_tensors": [str(item) for item in data["excluded_tensors"].tolist()],
            "total_shared_params": int(data["total_shared_params"].item()),
            "support_threshold": float(data["support_threshold"].item()),
            "singular_values": data["singular_values"],
            "left_vectors": data["left_vectors"],
            "model_coefficients": data["model_coefficients"],
            "all_singular_values": data["all_singular_values"],
            "all_eigenvalues": data["all_eigenvalues"],
            "explained_energy_ratio": data["explained_energy_ratio"],
            "all_explained_energy_ratio": data["all_explained_energy_ratio"],
            "gram": data["gram"],
            "norms_sq": data["norms_sq"],
            "cosine_similarity": data["cosine_similarity"],
            "support_jaccard": data["support_jaccard"],
            "shared_energy_frac": data["shared_energy_frac"],
            "unique_residual": data["unique_residual"],
            "sign_conflict_count": int(data["sign_conflict_count"].item()),
            "sign_conflict_energy": float(data["sign_conflict_energy"].item()),
            "sign_conflicts_top": json_payload.get("sign_conflicts_top", []),
            "layer_labels": [str(item) for item in data["layer_labels"].tolist()],
            "arch_labels": [str(item) for item in data["arch_labels"].tolist()],
            "component_layer_fracs": data["component_layer_fracs"],
            "component_arch_fracs": data["component_arch_fracs"],
        }
    analysis["target_paths"] = json_payload.get("target_paths", [])
    analysis["base_paths"] = json_payload.get("base_paths", [])
    return analysis


def validate_recipe(recipe: dict) -> None:
    if recipe.get("schema") == "routed_recipe":
        from .routing import validate_routed_recipe

        analysis = recipe.get("_analysis_cache")
        if analysis is None:
            analysis = load_analysis(recipe["analysis_dir"])
            recipe["_analysis_cache"] = analysis
        validate_routed_recipe(recipe, analysis)
        return
    if recipe.get("schema") != "svd_recipe":
        raise ValueError("Recipe schema must be 'svd_recipe' or 'routed_recipe'")
    if recipe.get("version") != "1.0":
        raise ValueError("Recipe version must be '1.0'")
    missing_policy = recipe.get("missing_policy", "error")
    if missing_policy not in {"error", "skip", "zero"}:
        raise ValueError("missing_policy must be one of: error, skip, zero")

    analysis = recipe.get("_analysis_cache")
    if analysis is None:
        analysis = load_analysis(recipe["analysis_dir"])
        recipe["_analysis_cache"] = analysis
    if analysis.get("schema_version") != SVD_SCHEMA_VERSION:
        raise ValueError(f"Analysis schema_version must be {SVD_SCHEMA_VERSION}")

    labels = analysis["labels"]
    if recipe.get("departure_label") not in labels:
        raise ValueError(f"Unknown departure_label: {recipe.get('departure_label')}")
    if "departure_path" not in recipe:
        raise ValueError("Recipe must include departure_path")
    if "output_path" not in recipe:
        raise ValueError("Recipe must include output_path")

    adjustments = recipe.get("adjustments")
    if not isinstance(adjustments, list):
        raise ValueError("Recipe adjustments must be a list")
    seen_components = set()
    novel_block_size = None
    for adjustment in adjustments:
        component = int(adjustment.get("component", 0))
        if component < 1 or component > analysis["n_components"]:
            raise ValueError(f"Component {component} is out of range 1..{analysis['n_components']}")
        if component in seen_components:
            raise ValueError(f"Component {component} appears more than once in recipe adjustments")
        seen_components.add(component)
        if float(analysis["singular_values"][component - 1]) == 0.0:
            raise ValueError(f"Component {component} has zero singular value")
        scope = adjustment.get("scope", "global")
        if scope not in ("global", "novel_support"):
            raise ValueError("Adjustment scope must be 'global' or 'novel_support'")
        if scope == "novel_support":
            float(adjustment.get("novelty_ratio", NOVELTY_RATIO))
            bs = int(adjustment.get("block_size", BLOCK_SIZE))
            float(adjustment.get("min_candidate_support", MATERIAL_BLOCK_FRACTION))
            if bs <= 0:
                raise ValueError("novel_support block_size must be positive")
            if novel_block_size is None:
                novel_block_size = bs
            elif bs != novel_block_size:
                raise ValueError(
                    "novel_support block_size must match across all adjustments "
                    f"({bs} != {novel_block_size})"
                )
        has_target = "target_coefficient" in adjustment
        has_gamma_mode = "gamma" in adjustment or "donor_label" in adjustment
        if has_target and has_gamma_mode:
            raise ValueError("Adjustment must use either target_coefficient or donor_label+gamma, not both")
        if not has_target and not has_gamma_mode:
            raise ValueError("Adjustment must provide target_coefficient or donor_label+gamma")
        if has_target:
            float(adjustment["target_coefficient"])
        else:
            donor_label = adjustment.get("donor_label")
            if donor_label not in labels:
                raise ValueError(f"Unknown donor_label: {donor_label}")
            if "gamma" not in adjustment:
                raise ValueError("Gamma-mode adjustment requires gamma")
            float(adjustment["gamma"])

    meta = _read_analysis_json(recipe["analysis_dir"])
    expected_fps = meta.get("base_fingerprints", [])
    current_fps = []
    for delta_dir in analysis["delta_dirs"]:
        manifest = load_manifest(delta_dir)
        if manifest is None:
            raise ValueError(f"Delta manifest referenced by analysis is missing: {delta_dir}")
        current_fps.append(manifest.base_fingerprint)
    if not expected_fps:
        raise ValueError("SVD analysis does not contain BASE fingerprints")
    if expected_fps != current_fps:
        raise ValueError("Analysis BASE fingerprints differ from current delta manifests")


def compute_svd_weights(analysis: dict, recipe: dict) -> np.ndarray:
    labels = analysis["labels"]
    departure_idx = labels.index(recipe["departure_label"])
    departure_coeffs = analysis["model_coefficients"][departure_idx]
    delta_a = np.zeros(analysis["n_components"], dtype=np.float64)
    for adjustment in recipe.get("adjustments", []):
        if adjustment.get("scope", "global") != "global":
            continue
        k = int(adjustment["component"]) - 1
        if "target_coefficient" in adjustment:
            delta_a[k] += float(adjustment["target_coefficient"]) - float(departure_coeffs[k])
        else:
            donor_idx = labels.index(adjustment["donor_label"])
            donor_coeff = float(analysis["model_coefficients"][donor_idx, k])
            departure_coeff = float(departure_coeffs[k])
            delta_a[k] += float(adjustment["gamma"]) * (donor_coeff - departure_coeff)
    weights = np.zeros(analysis["n_models"], dtype=np.float64)
    for k in range(analysis["n_components"]):
        sigma = float(analysis["singular_values"][k])
        if sigma <= 0:
            continue
        weights += delta_a[k] * analysis["left_vectors"][:, k] / sigma
    return weights


def _recipe_target_coefficients(analysis: dict, recipe: dict) -> tuple[np.ndarray, np.ndarray]:
    departure_idx = analysis["labels"].index(recipe["departure_label"])
    start = np.array(analysis["model_coefficients"][departure_idx], dtype=np.float64, copy=True)
    target = np.array(start, copy=True)
    for adjustment in recipe.get("adjustments", []):
        k = int(adjustment["component"]) - 1
        if "target_coefficient" in adjustment:
            target[k] = float(adjustment["target_coefficient"])
        else:
            donor_idx = analysis["labels"].index(adjustment["donor_label"])
            donor_coeff = float(analysis["model_coefficients"][donor_idx, k])
            target[k] = float(start[k]) + float(adjustment["gamma"]) * (donor_coeff - float(start[k]))
    return start, target


def _copy_tree(src: str, dst: str) -> None:
    ensure_dir(dst)
    for root, dirs, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        out_root = dst if rel_root == "." else os.path.join(dst, rel_root)
        ensure_dir(out_root)
        for dirname in dirs:
            ensure_dir(os.path.join(out_root, dirname))
        for filename in files:
            shutil.copy2(os.path.join(root, filename), os.path.join(out_root, filename))


def compose_svd(recipe_path: str, dry_run=False) -> None:
    with open(recipe_path, "r", encoding="utf-8") as handle:
        recipe = json.load(handle)
    if recipe.get("schema") == "routed_recipe":
        from .routing import compose_routed_recipe

        compose_routed_recipe(recipe_path, dry_run=dry_run)
        return
    validate_recipe(recipe)
    has_novel = any(
        adjustment.get("scope", "global") == "novel_support"
        for adjustment in recipe.get("adjustments", [])
    )
    if has_novel:
        recipe.pop("_analysis_cache", None)
        compose_svd_novel_support(recipe_path, dry_run=dry_run)
        return
    analysis = recipe.pop("_analysis_cache", None) or load_analysis(recipe["analysis_dir"])
    weights = compute_svd_weights(analysis, recipe)
    start_coeffs, target_coeffs = _recipe_target_coefficients(analysis, recipe)

    output_path = str(relocated_path(recipe["output_path"]))
    ensure_dir(output_path)
    resolved = {
        "original_recipe": recipe,
        "source_delta_weights": [
            {"delta_dir": delta_dir, "label": label, "weight": float(weight)}
            for delta_dir, label, weight in zip(analysis["delta_dirs"], analysis["labels"], weights)
        ],
        "starting_coefficients": [float(value) for value in start_coeffs],
        "target_coefficients": [float(value) for value in target_coeffs],
        "analysis_fingerprint": {
            "schema_version": analysis["schema_version"],
            "n_models": analysis["n_models"],
            "n_components": analysis["n_components"],
            "shared_tensor_count": analysis["shared_tensor_count"],
        },
    }
    with open(Path(output_path) / "resolved_recipe.json", "w", encoding="utf-8") as handle:
        json.dump(resolved, handle, indent=2)
    if dry_run:
        return

    active_specs = [
        DeltaSpec(delta_dir=delta_dir, weight=float(weight), label=label)
        for delta_dir, label, weight in zip(analysis["delta_dirs"], analysis["labels"], weights)
        if abs(float(weight)) >= 1e-12
    ]
    if not active_specs:
        _copy_tree(str(relocated_path(recipe["departure_path"])), output_path)
        provenance = {
            "base_dir": os.path.abspath(relocated_path(recipe["departure_path"])),
            "deltas": [],
            "dtype": recipe.get("dtype", "float16"),
            "missing_policy": recipe.get("missing_policy", "error"),
            "identity_copy": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        provenance_path = Path(output_path) / "provenance.json"
        provenance_tmp = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
        with open(provenance_tmp, "w", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2)
        os.replace(provenance_tmp, provenance_path)
        return

    compose_resumable(
        base_dir=str(relocated_path(recipe["departure_path"])),
        delta_specs=active_specs,
        output_dir=output_path,
        dtype=recipe.get("dtype", "float16"),
        missing_policy=recipe.get("missing_policy", "error"),
    )


def _write_resolved_recipe(recipe, analysis, global_weights, novel_params, output_path):
    resolved = {
        "original_recipe": {k: v for k, v in recipe.items() if k != "_analysis_cache"},
        "source_delta_weights": [
            {"delta_dir": delta_dir, "label": label, "weight": float(weight)}
            for delta_dir, label, weight in zip(
                analysis["delta_dirs"], analysis["labels"], global_weights
            )
        ],
        "novel_support_adjustments": [
            {
                "component": int(k + 1),
                "delta_a": float(delta_a_k),
                "novelty_ratio": float(nr),
                "block_size": int(bs),
                "min_candidate_support": float(ms),
            }
            for (k, delta_a_k, nr, bs, ms) in novel_params
        ],
        "analysis_fingerprint": {
            "schema_version": analysis["schema_version"],
            "n_models": analysis["n_models"],
            "n_components": analysis["n_components"],
            "shared_tensor_count": analysis["shared_tensor_count"],
        },
    }
    path = Path(output_path) / "resolved_recipe.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(resolved, handle, indent=2)
    os.replace(str(tmp), str(path))


def compose_svd_novel_support(recipe_path: str, dry_run: bool = False) -> None:
    """Novel-support scope composer. Applies masked component deltas per block."""
    import torch
    from safetensors.torch import save_file

    from .compose import _copy_metadata_files, _load_compose_state, _save_compose_state
    from .utils import load_tensor_index, sha256_file

    with open(recipe_path, "r", encoding="utf-8") as handle:
        recipe = json.load(handle)
    validate_recipe(recipe)
    analysis = recipe.pop("_analysis_cache", None) or load_analysis(recipe["analysis_dir"])

    global_adjustments = [
        a for a in recipe.get("adjustments", []) if a.get("scope", "global") == "global"
    ]
    novel_adjustments = [a for a in recipe.get("adjustments", []) if a.get("scope") == "novel_support"]

    global_recipe = dict(recipe)
    global_recipe["adjustments"] = global_adjustments
    global_weights = compute_svd_weights(analysis, global_recipe)

    labels = analysis["labels"]
    departure_idx = labels.index(recipe["departure_label"])
    departure_coeffs = analysis["model_coefficients"][departure_idx]

    novel_params = []
    for adj in novel_adjustments:
        k = int(adj["component"]) - 1
        if "target_coefficient" in adj:
            delta_a_k = float(adj["target_coefficient"]) - float(departure_coeffs[k])
        else:
            donor_idx = labels.index(adj["donor_label"])
            donor_coeff = float(analysis["model_coefficients"][donor_idx, k])
            delta_a_k = float(adj["gamma"]) * (donor_coeff - float(departure_coeffs[k]))
        novelty_ratio = float(adj.get("novelty_ratio", NOVELTY_RATIO))
        block_sz = int(adj.get("block_size", BLOCK_SIZE))
        min_support = float(adj.get("min_candidate_support", MATERIAL_BLOCK_FRACTION))
        novel_params.append((k, delta_a_k, novelty_ratio, block_sz, min_support))

    output_path = str(relocated_path(recipe["output_path"]))
    ensure_dir(output_path)
    _write_resolved_recipe(recipe, analysis, global_weights, novel_params, output_path)
    if dry_run:
        return

    departure_path = str(relocated_path(recipe["departure_path"]))
    missing_policy = recipe.get("missing_policy", "error")
    out_dtype = torch.float16 if recipe.get("dtype", "float16") == "float16" else torch.float32

    base_index = load_tensor_index(departure_path)
    shard_groups = {}
    for name, path in base_index.items():
        shard_groups.setdefault(os.path.basename(path), []).append(name)
    shard_groups = {key: sorted(value) for key, value in sorted(shard_groups.items())}

    all_manifests = [load_manifest(delta_dir) for delta_dir in analysis["delta_dirs"]]
    dep_manifest = all_manifests[departure_idx]
    left_vectors = np.asarray(analysis["left_vectors"], dtype=np.float64)
    singular_values = np.asarray(analysis["singular_values"], dtype=np.float64)

    state = _load_compose_state(output_path)
    weight_map = {}

    for shard_name, tensor_names in shard_groups.items():
        shard_path = Path(output_path) / shard_name
        if shard_path.exists():
            expected_sha = state["shards"].get(shard_name)
            if expected_sha and sha256_file(str(shard_path)) == expected_sha:
                for name in tensor_names:
                    weight_map[name] = shard_name
                continue

        # Group base tensors by their source file; process one tensor at a time so that
        # peak working memory stays O(tensor_size) rather than O(N_shard * tensor_size).
        source_files = {}
        for name in tensor_names:
            source_files.setdefault(str(base_index[name]), []).append(name)

        out_tensors = {}
        for src_path, src_names in source_files.items():
            with safe_open(src_path, framework="pt", device="cpu") as base_handle:
                for name in src_names:
                    base_t = base_handle.get_tensor(name).to(torch.float32)
                    orig_shape = base_t.shape
                    flat = base_t.reshape(-1)
                    del base_t

                    # Global weight accumulation: one source model at a time.
                    for i, (delta_dir, manifest) in enumerate(
                        zip(analysis["delta_dirs"], all_manifests)
                    ):
                        w = float(global_weights[i])
                        if abs(w) < 1e-12 or manifest is None:
                            continue
                        record = manifest.tensors.get(name)
                        if record is None:
                            if missing_policy == "error":
                                raise ValueError(
                                    f"Missing delta tensor: {name} in {delta_dir}"
                                )
                            continue
                        shard_delta_path = relocated_path(delta_dir) / record.shard_file
                        with safe_open(
                            str(shard_delta_path), framework="pt", device="cpu"
                        ) as handle:
                            delta_t = handle.get_tensor(name).to(torch.float32).reshape(-1)
                        if delta_t.shape != flat.shape:
                            if missing_policy == "error":
                                raise ValueError(f"Shape mismatch for {name}")
                            continue
                        flat = flat + w * delta_t
                        del delta_t

                    # Novel-support: one component at a time, sources re-streamed each pass.
                    # Peak memory = O(tensor_size); sources are never all held concurrently.
                    if novel_params:
                        dep_record = (
                            dep_manifest.tensors.get(name) if dep_manifest else None
                        )
                        if dep_record is not None:
                            dep_shard = (
                                relocated_path(analysis["delta_dirs"][departure_idx])
                                / dep_record.shard_file
                            )
                            with safe_open(
                                str(dep_shard), framework="pt", device="cpu"
                            ) as handle:
                                dep_delta = (
                                    handle.get_tensor(name)
                                    .to(torch.float32)
                                    .reshape(-1)
                                    .numpy()
                                )
                        else:
                            dep_delta = np.zeros(flat.shape[0], dtype=np.float32)

                        n_total = flat.shape[0]

                        for k, delta_a_k, novelty_ratio, block_sz, min_support in novel_params:
                            sigma_k = float(singular_values[k])
                            if sigma_k <= 0 or abs(delta_a_k) < 1e-12:
                                continue
                            u_k = left_vectors[:, k].astype(np.float32)

                            # Stream each source model once; release before loading the next.
                            v_k_flat = np.zeros(n_total, dtype=np.float32)
                            for i, (delta_dir, manifest) in enumerate(
                                zip(analysis["delta_dirs"], all_manifests)
                            ):
                                u_ki = float(u_k[i])
                                if abs(u_ki) < 1e-15 or manifest is None:
                                    continue
                                record = manifest.tensors.get(name)
                                if record is None:
                                    continue
                                shard_p = relocated_path(delta_dir) / record.shard_file
                                with safe_open(
                                    str(shard_p), framework="pt", device="cpu"
                                ) as handle:
                                    md = (
                                        handle.get_tensor(name)
                                        .to(torch.float32)
                                        .reshape(-1)
                                        .numpy()
                                    )
                                v_k_flat += np.float32(u_ki) * md
                                del md
                            v_k_flat /= np.float32(sigma_k)

                            # Add masked novel contribution block-by-block directly into flat.
                            offset = 0
                            while offset < n_total:
                                end = min(offset + block_sz, n_total)
                                v_b = v_k_flat[offset:end]
                                d_b = dep_delta[offset:end]
                                comp_e = float(np.dot(v_b, v_b))
                                dep_e = float(np.dot(d_b, d_b))
                                is_material = comp_e > min_support
                                is_novel = (
                                    dep_e <= novelty_ratio ** 2 * comp_e * sigma_k ** 2
                                ) and is_material
                                if is_novel:
                                    flat[offset:end].add_(
                                        torch.from_numpy(v_b), alpha=float(delta_a_k)
                                    )
                                offset = end
                            del v_k_flat

                        del dep_delta

                    out_tensors[name] = flat.reshape(orig_shape).to(out_dtype).contiguous()
                    del flat
                    weight_map[name] = shard_name

        tmp_path = Path(str(shard_path) + ".tmp")
        save_file(out_tensors, str(tmp_path))
        os.replace(str(tmp_path), str(shard_path))
        state["shards"][shard_name] = sha256_file(str(shard_path))
        _save_compose_state(output_path, state)

    if len(shard_groups) > 1:
        index_payload = {
            "metadata": {
                "total_size": sum((Path(output_path) / sn).stat().st_size for sn in shard_groups)
            },
            "weight_map": weight_map,
        }
        with open(Path(output_path) / "model.safetensors.index.json", "w", encoding="utf-8") as handle:
            json.dump(index_payload, handle, indent=2, sort_keys=True)
    _copy_metadata_files(departure_path, output_path)
    provenance = {
        "base_dir": os.path.abspath(departure_path),
        "novel_support": True,
        "dtype": recipe.get("dtype", "float16"),
        "missing_policy": missing_policy,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    provenance_path = Path(output_path) / "provenance.json"
    provenance_tmp = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    with open(provenance_tmp, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    os.replace(str(provenance_tmp), str(provenance_path))


def compose_svd_command(args) -> None:
    compose_svd(args.recipe, dry_run=args.dry_run)
