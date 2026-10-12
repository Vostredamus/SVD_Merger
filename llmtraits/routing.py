"""Deterministic block-routed composition.

The routed format deliberately keeps its intermediate files simple: JSON for
indexes and audit data, and non-pickled NPZ files for numerical checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from .manifest import load_manifest
from .paths import relocated_path
from .tensor_stream import iter_flat_ranges, read_flat_slice, tensor_numel
from .utils import ensure_dir, load_tensor_index, sha256_file


ROUTED_SCHEMA_VERSION = "1.0"
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_ARCHITECTURES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "embed_tokens",
    "lm_head",
    "input_layernorm",
    "post_attention_layernorm",
    "layernorm",
    "norm",
)
_HARD_EXCLUSIONS = ("embed_tokens", "lm_head", "norm", "layernorm")
_TRANSIENT_RECIPE_KEYS = {
    "_analysis_cache",
    "_routing_cache",
    "analysis_cache",
    "routing_cache",
    "transient_cache",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if key not in _TRANSIENT_RECIPE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def routed_identity_hash(identity: dict) -> str:
    payload = {key: value for key, value in identity.items() if key != "identity_hash"}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    ensure_dir(str(path.parent))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
    os.replace(str(tmp), str(path))


def _atomic_npz(path: Path, **arrays: Any) -> None:
    ensure_dir(str(path.parent))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(tmp), str(path))


def _settings(recipe: dict) -> dict:
    settings = dict(recipe.get("routing", {}))
    aliases = {
        "fine_block_size": ("fine_block_size", "block_size"),
        "coarse_block_size": ("coarse_block_size",),
        "protection_energy_fraction": (
            "protection_energy_fraction",
            "protected_energy_fraction",
        ),
        "hard_exclusions": ("hard_exclusions", "excluded_patterns"),
        "conflict_threshold": ("conflict_threshold",),
        "winner_only": ("winner_only",),
        "blend_cosine_threshold": ("blend_cosine_threshold",),
        "blend_score_ratio": ("blend_score_ratio",),
        "require_coarse_stability": (
            "require_coarse_stability",
            "coarse_stability",
        ),
        "max_rms_ratio": ("max_rms_ratio", "rms_cap"),
        "novelty_ratio": ("novelty_ratio",),
        "protected_component": ("protected_component", "c4_component"),
        "projection_iterations": (
            "projection_iterations",
            "c4_projection_iterations",
        ),
    }
    for destination, names in aliases.items():
        if destination in settings:
            continue
        for name in names:
            if name in recipe:
                settings[destination] = recipe[name]
                break
    settings.setdefault("fine_block_size", 262_144)
    settings.setdefault("coarse_block_size", 1_048_576)
    settings.setdefault("protection_energy_fraction", 0.90)
    settings.setdefault("hard_exclusions", list(_HARD_EXCLUSIONS))
    settings.setdefault("conflict_threshold", -0.10)
    settings.setdefault("winner_only", True)
    settings.setdefault("blend_cosine_threshold", 0.25)
    settings.setdefault("blend_score_ratio", 0.80)
    settings.setdefault("require_coarse_stability", True)
    settings.setdefault("max_rms_ratio", 0.005)
    settings.setdefault("novelty_ratio", 0.10)
    settings.setdefault("protected_component", 4)
    settings.setdefault("projection_iterations", 8)
    return settings


def _donor_label(donor: dict) -> str:
    return str(donor.get("donor_label", donor.get("label", "")))


def _donor_alpha(donor: dict) -> float:
    return float(donor.get("alpha", donor.get("weight", donor.get("gamma", 0.0))))


def _donor_components(donor: dict) -> list[int]:
    values = donor.get(
        "components",
        donor.get("svd_components", donor.get("affinity_components", [])),
    )
    if isinstance(values, (int, str)):
        values = [values]
    result = []
    for value in values or []:
        text = str(value).strip().lower()
        if text.startswith("c"):
            text = text[1:]
        result.append(int(text))
    return result


def _find_label(labels: list[str], needle: str) -> str | None:
    folded = needle.casefold()
    exact = [label for label in labels if label.casefold() == folded]
    if exact:
        return exact[0]
    matches = [
        label
        for label in labels
        if folded in label.casefold() or label.casefold() in folded
    ]
    return matches[0] if matches else None


def default_routed_recipe(
    analysis: dict,
    departure_label: str,
    departure_path: str = "",
    output_path: str = "",
) -> dict:
    """Return an empty routed-recipe template; add donors in the GUI or JSON."""
    labels = [str(label) for label in analysis.get("labels", [])]
    actual_departure = _find_label(labels, departure_label) or departure_label
    donors: list[dict] = []
    routing_path = f"{output_path}.routing" if output_path else ""
    return {
        "schema": "routed_recipe",
        "version": ROUTED_SCHEMA_VERSION,
        "analysis_dir": str(analysis.get("analysis_dir", "")),
        "departure_label": actual_departure,
        "departure_path": departure_path,
        "output_path": output_path,
        "routing_path": routing_path,
        "dtype": "source",
        "missing_policy": "error",
        "donors": donors,
        "routing": {
            "fine_block_size": 262_144,
            "coarse_block_size": 1_048_576,
            "protection_energy_fraction": 0.90,
            "hard_exclusions": list(_HARD_EXCLUSIONS),
            "conflict_threshold": -0.10,
            "winner_only": True,
            "blend_cosine_threshold": 0.25,
            "blend_score_ratio": 0.80,
            "require_coarse_stability": True,
            "max_rms_ratio": 0.005,
            "novelty_ratio": 0.10,
            "protected_component": 4,
            "projection_iterations": 8,
        },
    }


def _analysis_for(recipe: dict) -> dict:
    analysis = recipe.get("_analysis_cache")
    if analysis is None:
        analysis_dir = recipe.get("analysis_dir")
        if not analysis_dir:
            raise ValueError("Recipe must include analysis_dir")
        from .recipes import load_analysis

        analysis = load_analysis(str(analysis_dir))
    return analysis


def validate_routed_recipe(recipe: dict, analysis: dict) -> None:
    if recipe.get("schema") != "routed_recipe":
        raise ValueError("Recipe schema must be 'routed_recipe'")
    if recipe.get("version") != ROUTED_SCHEMA_VERSION:
        raise ValueError(f"Routed recipe version must be {ROUTED_SCHEMA_VERSION}")
    if str(analysis.get("schema_version")) != "1.0":
        raise ValueError("Analysis schema_version must be 1.0")
    labels = [str(label) for label in analysis.get("labels", [])]
    if recipe.get("departure_label") not in labels:
        raise ValueError(f"Unknown departure_label: {recipe.get('departure_label')}")
    for key in ("departure_path", "output_path"):
        if not recipe.get(key):
            raise ValueError(f"Recipe must include {key}")
    if recipe.get("dtype", "source") not in {"source", "float16", "float32"}:
        raise ValueError("dtype must be source, float16, or float32")
    if recipe.get("missing_policy", "error") != "error":
        raise ValueError("Routed composition requires missing_policy='error'")
    atlas_preplan = recipe.get("atlas_preplan")
    if atlas_preplan is not None:
        if not isinstance(atlas_preplan, dict):
            raise ValueError("atlas_preplan must be an object")
        if atlas_preplan.get("schema") != "atlas_route_preplan":
            raise ValueError("atlas_preplan schema must be 'atlas_route_preplan'")
        if not atlas_preplan.get("path"):
            raise ValueError("atlas_preplan must include path")
        if atlas_preplan.get("requires_weight_rms_finalization") is not True:
            raise ValueError(
                "atlas_preplan must require weight RMS finalization"
            )

    donors = recipe.get("donors", recipe.get("routes"))
    if not isinstance(donors, list):
        raise ValueError("Recipe donors must be a list")
    seen = set()
    for donor in donors:
        if not isinstance(donor, dict):
            raise ValueError("Each donor must be an object")
        label = _donor_label(donor)
        if label not in labels:
            raise ValueError(f"Unknown donor label: {label}")
        if label == recipe["departure_label"]:
            raise ValueError("Departure model cannot also be a donor")
        if label in seen:
            raise ValueError(f"Duplicate donor label: {label}")
        seen.add(label)
        alpha = _donor_alpha(donor)
        if not math.isfinite(alpha) or alpha < 0.0 or alpha > 2.0:
            raise ValueError(f"Alpha for donor {label} must be finite and in [0, 2]")
        for component in _donor_components(donor):
            if component <= 0 or component > int(analysis.get("n_components", 0)):
                raise ValueError(
                    f"SVD component {component} for {label} is outside "
                    f"1..{int(analysis.get('n_components', 0))}"
                )
        if "layers" in donor:
            [int(value) for value in donor["layers"]]
        for key in ("route_prior", "layer_boost", "architecture_boost"):
            if key in donor and not math.isfinite(float(donor[key])):
                raise ValueError(f"{key} for donor {label} must be finite")
        for band in donor.get("bands", []) or []:
            if not isinstance(band, dict):
                raise ValueError(f"Each band for donor {label} must be an object")
            factor = float(band.get("factor", 1.0))
            if not math.isfinite(factor) or factor < 0.0:
                raise ValueError(f"Band factor for donor {label} must be finite and >= 0")
            layer_range = band.get("layer_range")
            if layer_range is not None and (
                len(layer_range) != 2 or int(layer_range[0]) > int(layer_range[1])
            ):
                raise ValueError(f"Band layer_range for donor {label} must be [low, high]")

    settings = _settings(recipe)
    fine = int(settings["fine_block_size"])
    coarse = int(settings["coarse_block_size"])
    if fine <= 0 or coarse < fine or coarse % fine:
        raise ValueError("coarse_block_size must be a positive multiple of fine_block_size")
    for key in (
        "protection_energy_fraction",
        "blend_cosine_threshold",
        "blend_score_ratio",
        "max_rms_ratio",
        "novelty_ratio",
    ):
        value = float(settings[key])
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if not 0.0 <= float(settings["protection_energy_fraction"]) <= 1.0:
        raise ValueError("protection_energy_fraction must be in [0, 1]")
    if float(settings["max_rms_ratio"]) < 0 or float(settings["novelty_ratio"]) < 0:
        raise ValueError("RMS and novelty ratios must be non-negative")
    if int(settings["protected_component"]) <= 0:
        raise ValueError("protected_component must be positive")
    if int(settings["protected_component"]) > int(analysis.get("n_components", 0)):
        raise ValueError("protected_component exceeds the available SVD components")
    if int(settings["projection_iterations"]) <= 0:
        raise ValueError("projection_iterations must be positive")
    exclusions = settings["hard_exclusions"]
    if not isinstance(exclusions, list) or not all(isinstance(item, str) for item in exclusions):
        raise ValueError("hard_exclusions must be a list of strings")

    delta_dirs = list(analysis.get("delta_dirs", []))
    n_models = int(analysis.get("n_models", -1))
    n_components = int(analysis.get("n_components", -1))
    if len(delta_dirs) != len(labels) or n_models != len(labels):
        raise ValueError("Analysis model dimensions do not match labels/delta_dirs")
    gram = np.asarray(analysis.get("gram"))
    left = np.asarray(analysis.get("left_vectors"))
    singular = np.asarray(analysis.get("singular_values"))
    if gram.shape != (n_models, n_models) or not np.all(np.isfinite(gram)):
        raise ValueError("Analysis Gram matrix is invalid")
    if (
        n_components < 0
        or left.shape != (n_models, n_components)
        or singular.shape != (n_components,)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(singular))
        or np.any(singular < 0)
    ):
        raise ValueError("Analysis SVD dimensions or values are invalid")
    manifests = []
    for delta_dir in delta_dirs:
        manifest = load_manifest(delta_dir)
        if manifest is None:
            raise ValueError(f"Missing delta manifest: {delta_dir}")
        manifests.append(manifest)
    bases = {manifest.base_fingerprint for manifest in manifests}
    if len(bases) != 1:
        raise ValueError("Routed deltas do not share one BASE fingerprint")

    analysis_dir = recipe.get("analysis_dir")
    if analysis_dir:
        meta_path = relocated_path(analysis_dir) / "svd_analysis.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            expected = metadata.get("base_fingerprints", [])
            current = [manifest.base_fingerprint for manifest in manifests]
            if expected and expected != current:
                raise ValueError("Analysis BASE fingerprints differ from current delta manifests")


def _stable_pinv(matrix: np.ndarray, regularization: float) -> np.ndarray:
    symmetric = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(
        float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0,
        np.finfo(np.float64).tiny,
    )
    cutoff = max(float(regularization), np.finfo(np.float64).eps) * scale
    inverse = np.zeros_like(eigenvalues)
    good = eigenvalues > cutoff
    inverse[good] = 1.0 / eigenvalues[good]
    return (eigenvectors * inverse[np.newaxis, :]) @ eigenvectors.T


def compute_residual_coefficients(
    analysis: dict,
    departure_label: str,
    donor_labels: list[str],
    regularization: float = 1e-8,
) -> np.ndarray:
    """Express sequential donor-exclusive residuals in the model-delta basis."""

    labels = [str(label) for label in analysis["labels"]]
    if departure_label not in labels:
        raise ValueError(f"Unknown departure label: {departure_label}")
    if len(set(donor_labels)) != len(donor_labels):
        raise ValueError("Donor labels must be unique")
    if departure_label in donor_labels:
        raise ValueError("Departure cannot be a donor")
    unknown = [label for label in donor_labels if label not in labels]
    if unknown:
        raise ValueError(f"Unknown donor label: {unknown[0]}")
    gram = np.asarray(analysis["gram"], dtype=np.float64)
    n_models = len(labels)
    if gram.shape != (n_models, n_models) or not np.all(np.isfinite(gram)):
        raise ValueError("Analysis Gram matrix is invalid")
    if regularization < 0 or not math.isfinite(float(regularization)):
        raise ValueError("regularization must be finite and non-negative")

    departure = np.zeros(n_models, dtype=np.float64)
    departure[labels.index(departure_label)] = 1.0
    basis = [departure]
    residuals = []
    for donor_label in donor_labels:
        donor = np.zeros(n_models, dtype=np.float64)
        donor[labels.index(donor_label)] = 1.0
        basis_matrix = np.stack(basis, axis=0)
        basis_gram = basis_matrix @ gram @ basis_matrix.T
        rhs = basis_matrix @ gram @ donor
        projection = _stable_pinv(basis_gram, float(regularization)) @ rhs
        residual = donor - projection @ basis_matrix
        residuals.append(residual)
        basis.append(residual)
    result = (
        np.stack(residuals, axis=0)
        if residuals
        else np.empty((0, n_models), dtype=np.float64)
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("Residual coefficient computation produced non-finite values")
    return result


def _source_shards(model_dir: str) -> list[dict]:
    """Return a cheap change-sensitive signature for full-model source shards."""
    index = load_tensor_index(model_dir)
    result = []
    for path in sorted(set(index.values())):
        file_path = Path(path)
        stat = file_path.stat()
        result.append(
            {
                "file": os.path.basename(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        )
    return result


def routed_recipe_identity(recipe: dict, analysis: dict) -> dict:
    """Build a content identity for a recipe, its analysis, and all input shards."""

    manifests = []
    for label, delta_dir in zip(analysis["labels"], analysis["delta_dirs"]):
        manifest = load_manifest(delta_dir)
        if manifest is None:
            raise ValueError(f"Missing delta manifest: {delta_dir}")
        shard_items = []
        for name, shard in sorted(manifest.shards.items()):
            path = relocated_path(delta_dir) / name
            if not path.exists():
                raise ValueError(f"Missing delta shard: {path}")
            stat = path.stat()
            recorded_hash = shard.file_sha256
            if not recorded_hash:
                recorded_hash = sha256_file(str(path))
            shard_items.append(
                {
                    "name": name,
                    "sha256": recorded_hash,
                    "status": shard.status,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                }
            )
        tensor_digest = hashlib.sha256(
            _canonical_bytes(
                [
                    {
                        "name": name,
                        "shape": record.shape,
                        "base_sha256": record.base_sha256,
                        "delta_sha256": record.delta_sha256,
                        "shard_file": record.shard_file,
                    }
                    for name, record in sorted(manifest.tensors.items())
                ]
            )
        ).hexdigest()
        manifests.append(
            {
                "label": str(label),
                "delta_dir": os.path.abspath(str(delta_dir)),
                "base_fingerprint": manifest.base_fingerprint,
                "target_fingerprint": manifest.target_fingerprint,
                "tensor_manifest_sha256": tensor_digest,
                "shards": shard_items,
            }
        )
    canonical_recipe = _jsonable(recipe)
    for key in ("analysis_dir", "departure_path", "output_path", "routing_path"):
        if canonical_recipe.get(key):
            canonical_recipe[key] = os.path.abspath(str(canonical_recipe[key]))
    identity = {
        "routed_schema_version": ROUTED_SCHEMA_VERSION,
        "recipe": canonical_recipe,
        "analysis": {
            "schema_version": str(analysis.get("schema_version")),
            "n_models": int(analysis.get("n_models", 0)),
            "n_components": int(analysis.get("n_components", 0)),
            "shared_tensor_count": int(analysis.get("shared_tensor_count", 0)),
            "labels": [str(value) for value in analysis.get("labels", [])],
            "delta_dirs": [
                os.path.abspath(str(value)) for value in analysis.get("delta_dirs", [])
            ],
        },
        "manifests": manifests,
        "departure_shards": _source_shards(recipe["departure_path"]),
    }
    identity["identity_hash"] = routed_identity_hash(identity)
    return identity


def _routing_path(recipe: dict) -> Path:
    value = recipe.get("routing_path")
    if not value:
        value = f"{recipe['output_path']}.routing"
    return Path(value)


def _progress(path: Path, identity_hash: str, phase: str, **extra: Any) -> None:
    _atomic_json(
        path / "routing_progress.json",
        {
            "schema_version": ROUTED_SCHEMA_VERSION,
            "identity_hash": identity_hash,
            "phase": phase,
            "updated_at": _utc_now(),
            **extra,
        },
    )


def _notify(callback, current: int, total: int, name: str, phase: str) -> None:
    if callback is None:
        return
    payload = {
        "phase": phase,
        "current": current,
        "total": total,
        "tensor_name": name,
    }
    try:
        callback(payload)
    except TypeError:
        callback(current, total, name)


def _tensor_key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def _arch(name: str) -> str:
    lower = name.casefold()
    for value in _ARCHITECTURES:
        if value in lower:
            return value
    return "other"


def _layer(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _is_hard_excluded(tensor_name: str, settings: dict) -> bool:
    lower = tensor_name.casefold()
    return any(
        str(pattern).casefold() in lower
        for pattern in settings["hard_exclusions"]
    )


def _route_prior(donor: dict, tensor_name: str) -> float:
    architecture = _arch(tensor_name)
    layer = _layer(tensor_name)
    architectures = donor.get(
        "architectures", donor.get("tensor_types", donor.get("arch", []))
    )
    if isinstance(architectures, str):
        architectures = [architectures]
    if architectures and architecture not in [str(item) for item in architectures]:
        return 0.0
    layers = donor.get("layers", [])
    if isinstance(layers, (int, str)):
        layers = [layers]
    if layers and layer not in [int(item) for item in layers]:
        return 0.0
    prior = float(donor.get("route_prior", donor.get("prior", 1.0)))
    boost_arch = donor.get("boost_architectures", [])
    if isinstance(boost_arch, str):
        boost_arch = [boost_arch]
    if architecture in boost_arch:
        prior *= float(donor.get("architecture_boost", donor.get("boost", 1.5)))
    boost_layers = donor.get("boost_layers", [])
    if isinstance(boost_layers, (int, str)):
        boost_layers = [boost_layers]
    if layer is not None and layer in [int(item) for item in boost_layers]:
        prior *= float(donor.get("layer_boost", donor.get("boost", 1.5)))
    for band in donor.get("bands", []) or []:
        if _band_matches(band, architecture, layer):
            prior *= float(band.get("factor", 1.0))
    return max(prior, 0.0)


def _band_matches(band: dict, architecture: str, layer: int | None) -> bool:
    """A band is an architecture set x inclusive layer range; empty means any."""
    band_arch = band.get("architectures", [])
    if isinstance(band_arch, str):
        band_arch = [band_arch]
    if band_arch and architecture not in [str(item) for item in band_arch]:
        return False
    layer_range = band.get("layer_range")
    if layer_range:
        if layer is None:
            return False
        low, high = int(layer_range[0]), int(layer_range[1])
        if not low <= layer <= high:
            return False
    return True


def _shared_context(recipe: dict, analysis: dict) -> tuple[list[dict], list[str]]:
    models = []
    shared = None
    shapes: dict[str, tuple[int, ...]] = {}
    for label, delta_dir in zip(analysis["labels"], analysis["delta_dirs"]):
        manifest = load_manifest(delta_dir)
        if manifest is None:
            raise ValueError(f"Missing delta manifest: {delta_dir}")
        names = set(manifest.tensors)
        shared = names if shared is None else shared & names
        models.append({"label": label, "delta_dir": delta_dir, "manifest": manifest})
    shared_names = sorted(shared or set())
    for name in list(shared_names):
        current = {tuple(model["manifest"].tensors[name].shape) for model in models}
        if len(current) != 1:
            raise ValueError(f"Shape mismatch for shared tensor {name}")
        shapes[name] = next(iter(current))
    source_index = load_tensor_index(recipe["departure_path"])
    for name in shared_names:
        if name not in source_index:
            raise ValueError(f"Departure checkpoint is missing tensor {name}")
        source_path = source_index[name]
        if not source_path.casefold().endswith(".safetensors"):
            raise ValueError("Routed mode requires a safetensors departure checkpoint")
        with safe_open(source_path, framework="pt", device="cpu") as handle:
            if tuple(handle.get_slice(name).get_shape()) != shapes[name]:
                raise ValueError(f"Departure checkpoint shape mismatch for {name}")
    if not shared_names:
        raise ValueError("No shared tensors are available for routed composition")
    return models, shared_names


def _load_metric(path: Path, identity_hash: str, tensor_name: str) -> dict | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["identity_hash"].item()) != identity_hash:
                raise ValueError(f"Metrics identity mismatch for {tensor_name}")
            if str(data["tensor_name"].item()) != tensor_name:
                raise ValueError(f"Metrics tensor mismatch for {tensor_name}")
            return {key: np.array(data[key], copy=True) for key in data.files}
    except ValueError as error:
        if "identity mismatch" in str(error) or "tensor mismatch" in str(error):
            raise
        return None
    except Exception:
        return None


def _compute_metric(
    recipe: dict,
    analysis: dict,
    models: list[dict],
    name: str,
    coefficients: np.ndarray,
    identity_hash: str,
) -> dict:
    settings = _settings(recipe)
    fine = int(settings["fine_block_size"])
    donors = recipe.get("donors", recipe.get("routes", []))
    selected = sorted(
        {
            component
            for donor in donors
            for component in _donor_components(donor)
            if component <= int(analysis.get("n_components", 0))
        }
        | (
            {int(settings["protected_component"])}
            if int(settings["protected_component"]) <= int(analysis.get("n_components", 0))
            else set()
        )
    )
    selected_pos = {component: position for position, component in enumerate(selected)}
    departure_idx = list(analysis["labels"]).index(recipe["departure_label"])
    source_index = load_tensor_index(recipe["departure_path"])
    shape = tuple(models[0]["manifest"].tensors[name].shape)
    ranges = list(iter_flat_ranges(shape, fine))
    block_count = len(ranges)
    donor_count = len(donors)
    component_count = len(selected)
    dep_energy = np.zeros(block_count, dtype=np.float64)
    weight_energy = np.zeros(block_count, dtype=np.float64)
    residual_energy = np.zeros((block_count, donor_count), dtype=np.float64)
    residual_dot_departure = np.zeros((block_count, donor_count), dtype=np.float64)
    pair_dots = np.zeros((block_count, donor_count, donor_count), dtype=np.float64)
    component_dot = np.zeros(
        (block_count, donor_count, component_count), dtype=np.float64
    )
    component_energy = np.zeros((block_count, component_count), dtype=np.float64)
    route_prior = np.array(
        [_route_prior(donor, name) for donor in donors], dtype=np.float64
    )
    left = np.asarray(analysis["left_vectors"], dtype=np.float64)
    singular = np.asarray(analysis["singular_values"], dtype=np.float64)

    with ExitStack() as stack:
        handles = [
            stack.enter_context(
                safe_open(
                    str(
                        relocated_path(model["delta_dir"])
                        / model["manifest"].tensors[name].shard_file
                    ),
                    framework="pt",
                    device="cpu",
                )
            )
            for model in models
        ]
        source = stack.enter_context(
            safe_open(source_index[name], framework="pt", device="cpu")
        )
        for block_index, (start, end) in enumerate(ranges):
            deltas = np.empty((len(models), end - start), dtype=np.float64)
            for model_index, handle in enumerate(handles):
                deltas[model_index] = read_flat_slice(
                    handle, name, shape, start, end
                )
            departure = deltas[departure_idx]
            residuals = coefficients @ deltas
            weight = read_flat_slice(source, name, shape, start, end).astype(
                np.float64, copy=False
            )
            dep_energy[block_index] = departure @ departure
            weight_energy[block_index] = weight @ weight
            if donor_count:
                residual_energy[block_index] = np.einsum(
                    "ij,ij->i", residuals, residuals
                )
                residual_dot_departure[block_index] = residuals @ departure
                pair_dots[block_index] = residuals @ residuals.T
            for component_position, component in enumerate(selected):
                k = component - 1
                sigma = float(singular[k])
                if sigma <= 0:
                    continue
                vector = (left[:, k] @ deltas) / sigma
                component_energy[block_index, component_position] = vector @ vector
                if donor_count:
                    component_dot[block_index, :, component_position] = (
                        residuals @ vector
                    )
    protected = int(settings["protected_component"])
    if protected in selected_pos:
        c4_position = selected_pos[protected]
        c4_dot = component_dot[:, :, c4_position]
        c4_energy = component_energy[:, c4_position]
    else:
        c4_dot = np.zeros((block_count, donor_count), dtype=np.float64)
        c4_energy = np.zeros(block_count, dtype=np.float64)
    return {
        "schema_version": np.array(ROUTED_SCHEMA_VERSION, dtype="U16"),
        "identity_hash": np.array(identity_hash, dtype="U64"),
        "tensor_name": np.array(name, dtype="U2048"),
        "shape": np.asarray(shape, dtype=np.int64),
        "starts": np.asarray([item[0] for item in ranges], dtype=np.int64),
        "ends": np.asarray([item[1] for item in ranges], dtype=np.int64),
        "dep_energy": dep_energy,
        "weight_energy": weight_energy,
        "residual_energy": residual_energy,
        "residual_dot_departure": residual_dot_departure,
        "pair_dots": pair_dots,
        "selected_components": np.asarray(selected, dtype=np.int32),
        "component_dot": component_dot,
        "component_energy": component_energy,
        "c4_dot": c4_dot,
        "c4_energy": c4_energy,
        "route_prior": route_prior,
    }


def _scores(metric: dict, donors: list[dict], settings: dict) -> tuple[np.ndarray, np.ndarray]:
    energies = np.asarray(metric["residual_energy"], dtype=np.float64)
    dep_energy = np.asarray(metric["dep_energy"], dtype=np.float64)
    dep_dot = np.asarray(metric["residual_dot_departure"], dtype=np.float64)
    priors = np.asarray(metric["route_prior"], dtype=np.float64)
    scores = np.zeros_like(energies)
    eligible = np.ones_like(energies, dtype=bool)
    selected = [int(value) for value in metric["selected_components"]]
    component_dot = np.asarray(metric["component_dot"], dtype=np.float64)
    component_energy = np.asarray(metric["component_energy"], dtype=np.float64)
    eps = np.finfo(np.float64).tiny
    conflict = np.zeros_like(energies)
    denominator = np.sqrt(np.maximum(dep_energy[:, None] * energies, 0.0))
    np.divide(dep_dot, denominator, out=conflict, where=denominator > eps)
    eligible &= energies > eps
    eligible &= conflict >= float(settings["conflict_threshold"])
    novelty_ratio = float(settings["novelty_ratio"])

    for donor_index, donor in enumerate(donors):
        if bool(donor.get("require_novel", donor.get("novel_only", False))):
            ratio = float(donor.get("novelty_ratio", novelty_ratio))
            eligible[:, donor_index] &= (
                dep_energy <= ratio * ratio * energies[:, donor_index]
            )
        components = [
            component
            for component in _donor_components(donor)
            if component in selected
        ]
        affinity = np.zeros(energies.shape[0], dtype=np.float64)
        for component in components:
            position = selected.index(component)
            denom = np.sqrt(
                np.maximum(
                    energies[:, donor_index] * component_energy[:, position], 0.0
                )
            )
            local = np.zeros_like(affinity)
            np.divide(
                np.abs(component_dot[:, donor_index, position]),
                denom,
                out=local,
                where=denom > eps,
            )
            affinity = np.maximum(affinity, local)
        component_factor = 1.0 + affinity
        scores[:, donor_index] = (
            abs(_donor_alpha(donor))
            * np.sqrt(np.maximum(energies[:, donor_index], 0.0))
            * priors[donor_index]
            * component_factor
        )
    scores[~eligible] = 0.0
    return scores, conflict


def _winner_rows(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = scores.shape
    winner = np.full(rows, -1, dtype=np.int16)
    second = np.full(rows, -1, dtype=np.int16)
    for row in range(rows):
        order = sorted(range(columns), key=lambda col: (-scores[row, col], col))
        if order and scores[row, order[0]] > 0:
            winner[row] = order[0]
        if len(order) > 1 and scores[row, order[1]] > 0:
            second[row] = order[1]
    return winner, second


def _initial_plan(
    metric: dict,
    donors: list[dict],
    settings: dict,
    protected_threshold: float,
    tensor_name: str,
) -> tuple[dict, dict]:
    scores, conflicts = _scores(metric, donors, settings)
    winner, second = _winner_rows(scores)
    starts = np.asarray(metric["starts"], dtype=np.int64)
    coarse_size = int(settings["coarse_block_size"])
    if bool(settings["require_coarse_stability"]) and len(donors):
        coarse_ids = starts // coarse_size
        for coarse_id in np.unique(coarse_ids):
            positions = np.flatnonzero(coarse_ids == coarse_id)
            coarse_metric = {
                "residual_energy": np.sum(
                    np.asarray(metric["residual_energy"])[positions],
                    axis=0,
                    keepdims=True,
                ),
                "dep_energy": np.array(
                    [np.sum(np.asarray(metric["dep_energy"])[positions])]
                ),
                "residual_dot_departure": np.sum(
                    np.asarray(metric["residual_dot_departure"])[positions],
                    axis=0,
                    keepdims=True,
                ),
                "route_prior": metric["route_prior"],
                "selected_components": metric["selected_components"],
                "component_dot": np.sum(
                    np.asarray(metric["component_dot"])[positions],
                    axis=0,
                    keepdims=True,
                ),
                "component_energy": np.sum(
                    np.asarray(metric["component_energy"])[positions],
                    axis=0,
                    keepdims=True,
                ),
            }
            coarse_scores, _ = _scores(coarse_metric, donors, settings)
            coarse_winner = _winner_rows(coarse_scores)[0][0]
            for position in positions:
                if winner[position] != coarse_winner:
                    winner[position] = -1
                    second[position] = -1

    hard_excluded = _is_hard_excluded(tensor_name, settings)
    dep_energy = np.asarray(metric["dep_energy"], dtype=np.float64)
    if float(settings["protection_energy_fraction"]) <= 0:
        protected = np.zeros_like(dep_energy, dtype=bool)
    elif math.isinf(protected_threshold):
        protected = np.zeros_like(dep_energy, dtype=bool)
    else:
        protected = dep_energy >= protected_threshold
    if hard_excluded:
        winner.fill(-1)
        second.fill(-1)
    winner[protected] = -1
    second[winner < 0] = -1

    mix = np.zeros(len(winner), dtype=np.float32)
    if not bool(settings["winner_only"]):
        pair = np.asarray(metric["pair_dots"], dtype=np.float64)
        energy = np.asarray(metric["residual_energy"], dtype=np.float64)
        for block, (first, other) in enumerate(zip(winner, second)):
            if first < 0 or other < 0:
                continue
            denominator = math.sqrt(
                max(float(energy[block, first] * energy[block, other]), 0.0)
            )
            cosine = float(pair[block, first, other]) / denominator if denominator else 0.0
            ratio = scores[block, other] / max(scores[block, first], 1e-300)
            if (
                cosine >= float(settings["blend_cosine_threshold"])
                and ratio >= float(settings["blend_score_ratio"])
            ):
                mix[block] = float(
                    scores[block, other]
                    / (scores[block, first] + scores[block, other])
                )
            else:
                second[block] = -1
    else:
        second.fill(-1)
    plan = {
        "winner": winner,
        "second": second,
        "mix": mix,
        "scale": np.ones(len(winner), dtype=np.float32),
        "c4_beta": 0.0,
    }
    stats = {
        "blocks": len(winner),
        "accepted": int(np.count_nonzero(winner >= 0)),
        "protected": int(np.count_nonzero(protected)),
        "excluded": int(hard_excluded) * len(winner),
        "conflicts": int(np.count_nonzero(conflicts < float(settings["conflict_threshold"]))),
    }
    return plan, stats


def _raw_terms(metric: dict, plan: dict, donors: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if "raw_update_energy" in metric and "raw_c4_dot" in metric:
        return (
            np.asarray(metric["raw_update_energy"], dtype=np.float64),
            np.asarray(metric["raw_c4_dot"], dtype=np.float64),
        )
    winner = plan["winner"]
    second = plan["second"]
    mix = plan["mix"].astype(np.float64)
    energy = np.asarray(metric["residual_energy"], dtype=np.float64)
    pair = np.asarray(metric["pair_dots"], dtype=np.float64)
    c4_dot = np.asarray(metric["c4_dot"], dtype=np.float64)
    raw_energy = np.zeros(len(winner), dtype=np.float64)
    raw_dot = np.zeros(len(winner), dtype=np.float64)
    for block, first in enumerate(winner):
        if first < 0:
            continue
        a1 = _donor_alpha(donors[first]) * (1.0 - mix[block])
        raw_energy[block] = a1 * a1 * energy[block, first]
        raw_dot[block] = a1 * c4_dot[block, first]
        other = second[block]
        if other >= 0 and mix[block] > 0:
            a2 = _donor_alpha(donors[other]) * mix[block]
            raw_energy[block] += (
                a2 * a2 * energy[block, other]
                + 2.0 * a1 * a2 * pair[block, first, other]
            )
            raw_dot[block] += a2 * c4_dot[block, other]
    return raw_energy, raw_dot


def _finalize_projection(
    plans: dict[str, dict],
    metrics: dict[str, dict],
    donors: list[dict],
    settings: dict,
) -> float:
    cap = float(settings["max_rms_ratio"])
    accepted_c4_energy = 0.0
    for name, plan in plans.items():
        accepted = np.asarray(plan["winner"]) >= 0
        accepted_c4_energy += float(
            np.sum(np.asarray(metrics[name]["c4_energy"], dtype=np.float64)[accepted])
        )

    def evaluate(beta: float, *, save_scales: bool = False) -> float:
        leakage = 0.0
        for name, plan in plans.items():
            metric = metrics[name]
            raw_energy, raw_dot = _raw_terms(metric, plan, donors)
            c4_energy = np.asarray(metric["c4_energy"], dtype=np.float64)
            corrected = np.maximum(
                raw_energy - 2.0 * beta * raw_dot + beta * beta * c4_energy,
                0.0,
            )
            weight_energy = np.asarray(metric["weight_energy"], dtype=np.float64)
            scales = np.ones(len(corrected), dtype=np.float64)
            positive = corrected > 0
            scales[positive] = np.minimum(
                1.0,
                cap
                * np.sqrt(
                    np.maximum(weight_energy[positive], 0.0) / corrected[positive]
                ),
            )
            scales[plan["winner"] < 0] = 1.0
            if save_scales:
                plan["scale"] = scales
            accepted = plan["winner"] >= 0
            leakage += float(
                np.sum(
                    scales[accepted]
                    * (raw_dot[accepted] - beta * c4_energy[accepted])
                )
            )
        return leakage

    beta = 0.0
    if accepted_c4_energy > 0 and cap > 0:
        at_zero = evaluate(0.0)
        if at_zero != 0.0:
            low, high = (0.0, 1.0) if at_zero > 0 else (-1.0, 0.0)
            for _ in range(256):
                edge = evaluate(high if at_zero > 0 else low)
                if (at_zero > 0 and edge <= 0) or (at_zero < 0 and edge >= 0):
                    break
                if at_zero > 0:
                    high *= 2.0
                else:
                    low *= 2.0
            else:
                raise ValueError("Could not bracket the protected-component projection")
            for _ in range(max(64, int(settings["projection_iterations"]))):
                beta = (low + high) * 0.5
                value = evaluate(beta)
                if value > 0:
                    low = beta
                else:
                    high = beta
            beta = (low + high) * 0.5
    if not math.isfinite(beta):
        raise ValueError("C4 projection produced a non-finite coefficient")
    evaluate(beta, save_scales=True)
    for plan in plans.values():
        plan["c4_beta"] = float(beta)
    return float(beta)


def _read_required_json(path: Path, description: str) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as error:
        raise ValueError(f"Cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _bound_artifact(root: Path, relative: Any, description: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Missing {description} path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{description} escapes its artifact directory: {relative}") from error
    if not candidate.is_file():
        raise ValueError(f"Missing {description}: {candidate}")
    return candidate


def _normalized_path(value: Any) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(value))))


def _preplan_recipe_semantics(recipe: dict) -> dict:
    preplan = recipe.get("atlas_preplan", {})
    result = {
        "schema": recipe.get("schema"),
        "version": recipe.get("version"),
        "candidate_id": recipe.get("candidate_id"),
        "analysis_dir": _normalized_path(recipe.get("analysis_dir", "")),
        "departure_label": recipe.get("departure_label"),
        "departure_path": _normalized_path(recipe.get("departure_path", "")),
        "dtype": recipe.get("dtype", "source"),
        "missing_policy": recipe.get("missing_policy", "error"),
        "donors": _jsonable(recipe.get("donors", recipe.get("routes", []))),
        "routing": _jsonable(_settings(recipe)),
        "residual_regularization": float(recipe.get("residual_regularization", 1e-8)),
        "atlas_preplan": {
            "schema": preplan.get("schema"),
            "path": _normalized_path(preplan.get("path", "")),
            "requires_weight_rms_finalization": preplan.get(
                "requires_weight_rms_finalization"
            ),
        },
    }
    return result


def _find_atlas_root(
    preplan_root: Path, descriptor: dict, preplan_index: dict
) -> Path:
    candidates: list[Path] = []
    for value in (descriptor.get("atlas_path"), preplan_index.get("atlas_path")):
        if value:
            candidates.append(Path(value))
    analysis_output = preplan_root.parent.parent
    candidates.extend((analysis_output, preplan_root.parents[2]))
    search_parent = preplan_root.parents[2]
    try:
        candidates.extend(
            child
            for child in search_parent.iterdir()
            if child.is_dir() and (child / "atlas_index.json").is_file()
        )
    except OSError:
        pass
    matches = []
    seen = set()
    expected = preplan_index.get("atlas_identity_hash")
    for candidate in candidates:
        key = _normalized_path(candidate)
        if key in seen:
            continue
        seen.add(key)
        index_path = candidate / "atlas_index.json"
        identity_path = candidate / "atlas_identity.json"
        if not index_path.is_file() or not identity_path.is_file():
            continue
        try:
            index = _read_required_json(index_path, "atlas index")
        except ValueError:
            continue
        if index.get("identity_hash") == expected:
            matches.append(candidate.resolve())
    if len(matches) != 1:
        raise ValueError(
            "Could not uniquely locate the atlas bound to the cached preplan"
        )
    return matches[0]


def _validate_current_delta_bindings(analysis: dict, atlas_identity: dict) -> None:
    expected_models = atlas_identity.get("models")
    if not isinstance(expected_models, list) or len(expected_models) != len(
        analysis["labels"]
    ):
        raise ValueError("Atlas model identity is invalid")
    for label, delta_dir, expected in zip(
        analysis["labels"], analysis["delta_dirs"], expected_models
    ):
        manifest = load_manifest(delta_dir)
        if manifest is None:
            raise ValueError(f"Missing delta manifest: {delta_dir}")
        tensor_digest = hashlib.sha256(
            _canonical_bytes(
                {
                    name: {
                        "shape": record.shape,
                        "shard": record.shard_file,
                        "delta_sha256": record.delta_sha256,
                    }
                    for name, record in sorted(manifest.tensors.items())
                }
            )
        ).hexdigest()
        if (
            str(expected.get("label")) != str(label)
            or _normalized_path(expected.get("delta_dir", ""))
            != _normalized_path(delta_dir)
            or expected.get("base_fingerprint") != manifest.base_fingerprint
            or expected.get("target_fingerprint") != manifest.target_fingerprint
            or expected.get("tensor_manifest_sha256") != tensor_digest
        ):
            raise ValueError(f"Current delta manifest differs from atlas for {label}")
        expected_shards = expected.get("shards")
        if not isinstance(expected_shards, list):
            raise ValueError(f"Atlas shard identity is invalid for {label}")
        current_shards = []
        for filename, record in sorted(manifest.shards.items()):
            path = relocated_path(delta_dir) / filename
            if not path.is_file():
                raise ValueError(f"Missing delta shard: {path}")
            if not record.file_sha256:
                raise ValueError(
                    f"Delta manifest lacks a recorded shard hash for {path}"
                )
            current_shards.append(
                {
                    "name": filename,
                    "sha256": record.file_sha256,
                    "size": path.stat().st_size,
                }
            )
        if current_shards != expected_shards:
            raise ValueError(f"Current delta shards differ from atlas for {label}")


def _gram_raw_terms(
    gram: np.ndarray,
    plan: dict,
    coefficients: np.ndarray,
    donor_alphas: np.ndarray,
    component_coefficient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    block_count, model_count, _ = gram.shape
    residual_energy = np.maximum(
        np.einsum(
            "dm,bmn,dn->bd", coefficients, gram, coefficients, optimize=True
        ),
        0.0,
    )
    update_coefficients = np.zeros((block_count, model_count), dtype=np.float64)
    winner = np.asarray(plan["winner"], dtype=np.int16)
    second = np.asarray(plan["second"], dtype=np.int16)
    mix = np.asarray(plan["mix"], dtype=np.float64)
    accepted = winner >= 0
    positions = np.flatnonzero(accepted)
    if positions.size:
        first = winner[positions]
        update_coefficients[positions] = (
            donor_alphas[first, None]
            * (1.0 - mix[positions, None])
            * coefficients[first]
        )
        blended = second[positions] >= 0
        if np.any(blended):
            blend_positions = positions[blended]
            other = second[blend_positions]
            update_coefficients[blend_positions] += (
                donor_alphas[other, None]
                * mix[blend_positions, None]
                * coefficients[other]
            )
    raw_energy = np.maximum(
        np.einsum(
            "bi,bij,bj->b",
            update_coefficients,
            gram,
            update_coefficients,
            optimize=True,
        ),
        0.0,
    )
    raw_dot = np.einsum(
        "bi,bij,j->b",
        update_coefficients,
        gram,
        component_coefficient,
        optimize=True,
    )
    c4_energy = np.maximum(
        np.einsum(
            "i,bij,j->b",
            component_coefficient,
            gram,
            component_coefficient,
            optimize=True,
        ),
        0.0,
    )
    return raw_energy, raw_dot, c4_energy, residual_energy


def _validate_atlas_preplan(recipe: dict, analysis: dict) -> dict:
    descriptor = recipe["atlas_preplan"]
    preplan_root = Path(descriptor["path"]).resolve()
    preplan_index_path = preplan_root / "route_index.json"
    summary_path = preplan_root / "route_summary.json"
    preplan_index = _read_required_json(preplan_index_path, "atlas preplan index")
    source_summary = _read_required_json(summary_path, "atlas preplan summary")
    if (
        preplan_index.get("schema") != "atlas_route_preplan"
        or preplan_index.get("schema_version") != ROUTED_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported atlas preplan schema")
    if (
        source_summary.get("schema") != "atlas_route_preplan_summary"
        or source_summary.get("schema_version") != ROUTED_SCHEMA_VERSION
        or source_summary.get("status") != "complete"
    ):
        raise ValueError("Atlas preplan summary is not complete")
    identity = preplan_index.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Atlas preplan identity is missing")
    identity_hash = routed_identity_hash(identity)
    if (
        identity.get("identity_hash") != identity_hash
        or preplan_index.get("identity_hash") != identity_hash
        or source_summary.get("identity_hash") != identity_hash
    ):
        raise ValueError("Atlas preplan identity hash mismatch")
    if (
        preplan_index.get("scale_status")
        != "nominal_one_not_full_weight_rms_capped"
        or preplan_index.get("requires_finalization", {}).get(
            "departure_weight_rms_scale"
        )
        is not True
        or preplan_index.get("requires_finalization", {}).get(
            "projection_beta_after_scaling"
        )
        is not True
    ):
        raise ValueError("Atlas preplan is not marked for exact RMS finalization")

    candidate_id = str(recipe.get("candidate_id", ""))
    if not candidate_id or preplan_index.get("candidate_id") != candidate_id:
        raise ValueError("Atlas preplan candidate identity mismatch")
    candidate_path = (
        preplan_root.parent.parent / "candidate_recipes" / f"{candidate_id}.json"
    )
    candidate_recipe = _read_required_json(candidate_path, "source candidate recipe")
    candidate_hash = hashlib.sha256(_canonical_bytes(candidate_recipe)).hexdigest()
    if (
        candidate_hash != preplan_index.get("recipe_hash")
        or identity.get("recipe_hash") != candidate_hash
    ):
        raise ValueError("Atlas preplan candidate recipe hash mismatch")
    if _preplan_recipe_semantics(recipe) != _preplan_recipe_semantics(candidate_recipe):
        raise ValueError("Runtime recipe does not match the cached candidate recipe")

    analysis_output = preplan_root.parent.parent
    analysis_identity_path = analysis_output / "analysis_identity.json"
    analysis_identity = _read_required_json(
        analysis_identity_path, "atlas analysis identity"
    )
    analysis_identity_hash = routed_identity_hash(analysis_identity)
    if (
        analysis_identity.get("identity_hash") != analysis_identity_hash
        or preplan_index.get("analysis_identity_hash") != analysis_identity_hash
        or identity.get("analysis_identity_hash") != analysis_identity_hash
    ):
        raise ValueError("Atlas analysis identity hash mismatch")
    svd_path = relocated_path(recipe["analysis_dir"]) / "svd_analysis.npz"
    # The departure is bound through the candidate recipe hash and the
    # re-verified residual coefficients below; the analysis identity's
    # departure_label only describes the derived reports, so a preplan built
    # for another departure from the same atlas Grams remains valid.
    if (
        not svd_path.is_file()
        or sha256_file(str(svd_path))
        != analysis_identity.get("svd_analysis_sha256")
        or analysis_identity.get("labels")
        != [str(value) for value in analysis["labels"]]
    ):
        raise ValueError("Current SVD analysis does not match the atlas analysis")

    atlas_root = _find_atlas_root(preplan_root, descriptor, preplan_index)
    atlas_index = _read_required_json(atlas_root / "atlas_index.json", "atlas index")
    atlas_identity = _read_required_json(
        atlas_root / "atlas_identity.json", "atlas identity"
    )
    atlas_identity_hash = routed_identity_hash(atlas_identity)
    if (
        atlas_identity.get("identity_hash") != atlas_identity_hash
        or atlas_index.get("identity_hash") != atlas_identity_hash
        or preplan_index.get("atlas_identity_hash") != atlas_identity_hash
        or identity.get("atlas_identity_hash") != atlas_identity_hash
    ):
        raise ValueError("Atlas identity hash mismatch")
    if (
        atlas_index.get("schema_version") != ROUTED_SCHEMA_VERSION
        or atlas_identity.get("schema_version") != ROUTED_SCHEMA_VERSION
        or int(atlas_index.get("block_size", -1))
        != int(preplan_index.get("fine_block_size", -2))
        or int(analysis_identity.get("atlas_block_size", -1))
        != int(atlas_index.get("block_size", -2))
        or int(analysis_identity.get("atlas_tensor_count", -1))
        != int(atlas_index.get("tensor_count", -2))
        or int(analysis_identity.get("atlas_block_count", -1))
        != int(atlas_index.get("block_count", -2))
    ):
        raise ValueError("Atlas dimensions do not match the cached preplan")
    labels = [str(value) for value in analysis["labels"]]
    if atlas_index.get("labels") != labels or atlas_identity.get("labels") != labels:
        raise ValueError("Atlas and analysis label order differ")
    _validate_current_delta_bindings(analysis, atlas_identity)

    donors = recipe.get("donors", recipe.get("routes", []))
    donor_labels = [_donor_label(donor) for donor in donors]
    if preplan_index.get("donor_labels") != donor_labels:
        raise ValueError("Atlas preplan donor order mismatch")
    coefficients = compute_residual_coefficients(
        analysis,
        recipe["departure_label"],
        donor_labels,
        float(recipe.get("residual_regularization", 1e-8)),
    )
    cached_coefficients = np.asarray(
        preplan_index.get("residual_coefficients"), dtype=np.float64
    )
    if (
        cached_coefficients.shape != coefficients.shape
        or not np.allclose(cached_coefficients, coefficients, rtol=1e-12, atol=1e-14)
    ):
        raise ValueError("Atlas preplan residual coefficients mismatch")
    settings = _settings(recipe)
    if (
        int(preplan_index.get("coarse_block_size", -1))
        != int(settings["coarse_block_size"])
        or int(preplan_index.get("protected_component", -1))
        != int(settings["protected_component"])
    ):
        raise ValueError("Atlas preplan routing settings mismatch")

    atlas_records = atlas_index.get("tensors")
    preplan_records = preplan_index.get("tensors")
    shared_tensors = atlas_identity.get("shared_tensors")
    if (
        not isinstance(atlas_records, dict)
        or not isinstance(preplan_records, dict)
        or not isinstance(shared_tensors, list)
        or set(preplan_records) != set(atlas_records)
        or set(preplan_records) != set(shared_tensors)
        or int(atlas_index.get("tensor_count", -1)) != len(preplan_records)
    ):
        raise ValueError("Atlas preplan tensor set mismatch")

    source_index = load_tensor_index(recipe["departure_path"])
    missing_source = sorted(set(preplan_records) - set(source_index))
    if missing_source:
        raise ValueError(
            f"Departure checkpoint is missing tensor {missing_source[0]}"
        )
    source_groups: dict[str, list[str]] = {}
    for name in preplan_records:
        source_path = source_index[name]
        if not source_path.casefold().endswith(".safetensors"):
            raise ValueError("Routed mode requires a safetensors departure checkpoint")
        source_groups.setdefault(source_path, []).append(name)
    source_metadata = []
    for source_path, names in sorted(source_groups.items()):
        with safe_open(source_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in sorted(names):
                if name not in available:
                    raise ValueError(f"Departure shard is missing tensor {name}")
                shape = tuple(int(value) for value in handle.get_slice(name).get_shape())
                if shape != tuple(int(value) for value in preplan_records[name]["shape"]):
                    raise ValueError(f"Departure checkpoint shape mismatch for {name}")
                source_metadata.append(
                    {"name": name, "shape": list(shape), "file": os.path.basename(source_path)}
                )

    protected_component = int(settings["protected_component"])
    k = protected_component - 1
    left = np.asarray(analysis["left_vectors"], dtype=np.float64)
    singular = np.asarray(analysis["singular_values"], dtype=np.float64)
    component_coefficient = np.zeros(len(labels), dtype=np.float64)
    if 0 <= k < left.shape[1] and singular[k] > 0:
        component_coefficient = left[:, k] / singular[k]
    donor_alphas = np.asarray([_donor_alpha(item) for item in donors], dtype=np.float64)
    plans: dict[str, dict] = {}
    terms: dict[str, dict] = {}
    total_blocks = 0
    total_accepted = 0
    total_protected = 0
    total_excluded = 0
    winner_counts = {label: 0 for label in donor_labels}
    threshold = source_summary.get("protection_threshold")
    threshold_value = math.inf if threshold is None else float(threshold)
    fraction = float(settings["protection_energy_fraction"])

    for name in sorted(preplan_records):
        record = preplan_records[name]
        atlas_record = atlas_records[name]
        shape = tuple(int(value) for value in record.get("shape", []))
        blocks = int(record.get("blocks", -1))
        if (
            shape != tuple(int(value) for value in atlas_record.get("shape", []))
            or blocks != int(atlas_record.get("blocks", -2))
            or record.get("source_atlas_file") != atlas_record.get("file")
            or record.get("source_atlas_sha256") != atlas_record.get("sha256")
        ):
            raise ValueError(f"Atlas preplan tensor metadata mismatch for {name}")
        plan_path = _bound_artifact(preplan_root, record.get("plan"), "preplan")
        atlas_path = _bound_artifact(
            atlas_root, atlas_record.get("file"), "atlas tensor"
        )
        if sha256_file(str(plan_path)) != record.get("plan_sha256"):
            raise ValueError(f"Atlas preplan SHA-256 mismatch for {name}")
        if sha256_file(str(atlas_path)) != atlas_record.get("sha256"):
            raise ValueError(f"Atlas tensor SHA-256 mismatch for {name}")
        try:
            with np.load(plan_path, allow_pickle=False) as data:
                plan = {key: np.array(data[key], copy=True) for key in data.files}
            with np.load(atlas_path, allow_pickle=False) as data:
                atlas_tensor = {
                    key: np.array(data[key], copy=True)
                    for key in (
                        "identity_hash",
                        "tensor_name",
                        "shape",
                        "starts",
                        "ends",
                        "gram",
                        "completed_blocks",
                    )
                }
        except Exception as error:
            raise ValueError(f"Cannot read cached preplan data for {name}: {error}") from error
        required_plan = {
            "schema_version",
            "identity_hash",
            "atlas_identity_hash",
            "source_sha256",
            "tensor_name",
            "shape",
            "starts",
            "ends",
            "winner",
            "second",
            "mix",
            "scale",
            "protected",
            "winner_score",
            "second_score",
            "sequential_residual_energy",
            "scale_status",
        }
        if not required_plan.issubset(plan):
            raise ValueError(f"Atlas preplan payload is incomplete for {name}")
        starts = np.asarray(plan["starts"], dtype=np.int64)
        ends = np.asarray(plan["ends"], dtype=np.int64)
        expected_ranges = list(
            iter_flat_ranges(shape, int(preplan_index["fine_block_size"]))
        )
        expected_starts = np.asarray([item[0] for item in expected_ranges], dtype=np.int64)
        expected_ends = np.asarray([item[1] for item in expected_ranges], dtype=np.int64)
        vector_keys = (
            "winner",
            "second",
            "mix",
            "scale",
            "protected",
            "winner_score",
            "second_score",
        )
        winner = np.asarray(plan["winner"], dtype=np.int16)
        second = np.asarray(plan["second"], dtype=np.int16)
        mix = np.asarray(plan["mix"], dtype=np.float64)
        protected = np.asarray(plan["protected"])
        residual_energy = np.asarray(
            plan["sequential_residual_energy"], dtype=np.float64
        )
        if (
            str(plan["schema_version"].item()) != ROUTED_SCHEMA_VERSION
            or str(plan["identity_hash"].item()) != identity_hash
            or str(plan["atlas_identity_hash"].item()) != atlas_identity_hash
            or str(plan["source_sha256"].item()) != atlas_record.get("sha256")
            or str(plan["tensor_name"].item()) != name
            or str(plan["scale_status"].item())
            != "nominal_unscaled_requires_departure_weight_rms"
            or tuple(int(value) for value in plan["shape"]) != shape
            or blocks != len(expected_ranges)
            or any(np.asarray(plan[key]).shape != (blocks,) for key in vector_keys)
            or residual_energy.shape != (blocks, len(donors))
            or not np.array_equal(starts, expected_starts)
            or not np.array_equal(ends, expected_ends)
            or not np.all(np.isfinite(mix))
            or not np.all(np.isfinite(np.asarray(plan["scale"])))
            or not np.all(np.isfinite(np.asarray(plan["winner_score"])))
            or not np.all(np.isfinite(np.asarray(plan["second_score"])))
            or not np.all(np.isfinite(residual_energy))
            or np.any(residual_energy < -1e-10)
            or np.any((winner < -1) | (winner >= len(donors)))
            or np.any((second < -1) | (second >= len(donors)))
            or np.any((mix < 0) | (mix > 1))
            or np.any((winner < 0) & (second >= 0))
            or np.any((second >= 0) & (second == winner))
            or np.any((protected != 0) & (protected != 1))
            or not np.all(np.asarray(plan["scale"]) == 1.0)
        ):
            raise ValueError(f"Atlas preplan payload is invalid for {name}")
        gram = np.asarray(atlas_tensor["gram"], dtype=np.float64)
        if (
            str(atlas_tensor["identity_hash"].item()) != atlas_identity_hash
            or str(atlas_tensor["tensor_name"].item()) != name
            or tuple(int(value) for value in atlas_tensor["shape"]) != shape
            or not np.array_equal(atlas_tensor["starts"], starts)
            or not np.array_equal(atlas_tensor["ends"], ends)
            or int(atlas_tensor["completed_blocks"].item()) != blocks
            or gram.shape != (blocks, len(labels), len(labels))
            or not np.all(np.isfinite(gram))
            or not np.allclose(gram, np.swapaxes(gram, 1, 2), rtol=1e-9, atol=1e-9)
        ):
            raise ValueError(f"Atlas tensor payload is invalid for {name}")
        raw_energy, raw_dot, c4_energy, reconstructed_residual = _gram_raw_terms(
            gram, plan, coefficients, donor_alphas, component_coefficient
        )
        if not np.allclose(
            residual_energy, reconstructed_residual, rtol=1e-8, atol=1e-10
        ):
            raise ValueError(f"Cached residual energies mismatch the atlas for {name}")
        expected_protected = (
            np.asarray(gram[:, labels.index(recipe["departure_label"]), labels.index(recipe["departure_label"])])
            >= threshold_value
            if math.isfinite(threshold_value) and fraction > 0
            else np.zeros(blocks, dtype=bool)
        )
        if not np.array_equal(protected.astype(bool), expected_protected):
            raise ValueError(f"Cached protected blocks mismatch the atlas for {name}")
        plans[name] = plan
        terms[name] = {
            "raw_update_energy": raw_energy,
            "raw_c4_dot": raw_dot,
            "c4_energy": c4_energy,
        }
        total_blocks += blocks
        total_accepted += int(np.count_nonzero(winner >= 0))
        total_protected += int(np.count_nonzero(protected))
        total_excluded += int(_is_hard_excluded(name, settings)) * blocks
        for donor_index, label in enumerate(donor_labels):
            winner_counts[label] += int(np.count_nonzero(winner == donor_index))

    expected_stats = {
        "blocks": total_blocks,
        "accepted": total_accepted,
        "protected": total_protected,
        "excluded": total_excluded,
        "winner_counts": winner_counts,
    }
    if any(source_summary.get(key) != value for key, value in expected_stats.items()):
        raise ValueError("Atlas preplan summary statistics mismatch")
    if (
        float(source_summary.get("protection_energy_fraction", -1)) != fraction
        or int(source_summary.get("conflicts", -1)) < 0
        or int(source_summary.get("conflicts", -1)) > total_blocks * max(len(donors), 1)
    ):
        raise ValueError("Atlas preplan summary settings are invalid")

    return {
        "root": preplan_root,
        "index_path": preplan_index_path,
        "summary_path": summary_path,
        "index": preplan_index,
        "summary": source_summary,
        "atlas_root": atlas_root,
        "atlas_index": atlas_index,
        "analysis_identity_path": analysis_identity_path,
        "analysis_identity": analysis_identity,
        "candidate_path": candidate_path,
        "candidate_hash": candidate_hash,
        "plans": plans,
        "terms": terms,
        "source_index": source_index,
        "source_groups": source_groups,
        "source_metadata_sha256": hashlib.sha256(
            _canonical_bytes(source_metadata)
        ).hexdigest(),
        "coefficients": coefficients,
        "donor_labels": donor_labels,
        "stats": expected_stats,
    }


def _load_finalization_metric(
    path: Path,
    identity_hash: str,
    name: str,
    shape: tuple[int, ...],
    starts: np.ndarray,
    ends: np.ndarray,
    preplan_sha256: str,
    atlas_sha256: str,
) -> dict | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            metric = {key: np.array(data[key], copy=True) for key in data.files}
        blocks = len(starts)
        if (
            str(metric["identity_hash"].item()) != identity_hash
            or str(metric["tensor_name"].item()) != name
            or str(metric["preplan_sha256"].item()) != preplan_sha256
            or str(metric["atlas_sha256"].item()) != atlas_sha256
        ):
            raise ValueError(f"Finalization metric identity mismatch for {name}")
        if (
            tuple(int(value) for value in metric["shape"]) != shape
            or not np.array_equal(metric["starts"], starts)
            or not np.array_equal(metric["ends"], ends)
            or any(
                np.asarray(metric[key]).shape != (blocks,)
                for key in (
                    "raw_update_energy",
                    "raw_c4_dot",
                    "c4_energy",
                    "weight_energy",
                )
            )
            or any(
                not np.all(np.isfinite(metric[key]))
                for key in (
                    "raw_update_energy",
                    "raw_c4_dot",
                    "c4_energy",
                    "weight_energy",
                )
            )
            or np.any(metric["raw_update_energy"] < 0)
            or np.any(metric["c4_energy"] < 0)
            or np.any(metric["weight_energy"] < 0)
        ):
            return None
        return metric
    except ValueError as error:
        if "identity mismatch" in str(error):
            raise
        return None
    except Exception:
        return None


def _read_departure_weight_energy(
    handle, name: str, shape: tuple[int, ...], starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    result = np.zeros(len(starts), dtype=np.float64)
    for block, (start, end) in enumerate(zip(starts, ends)):
        values = read_flat_slice(handle, name, shape, int(start), int(end)).astype(
            np.float64, copy=False
        )
        result[block] = float(values @ values)
    return result


def _final_plan_matches(
    path: Path,
    identity_hash: str,
    name: str,
    source_plan: dict,
    scales: np.ndarray,
    beta: float,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["identity_hash"].item()) == identity_hash
                and str(data["tensor_name"].item()) == name
                and np.array_equal(data["shape"], source_plan["shape"])
                and np.array_equal(data["starts"], source_plan["starts"])
                and np.array_equal(data["ends"], source_plan["ends"])
                and np.array_equal(data["winner"], source_plan["winner"])
                and np.array_equal(data["second"], source_plan["second"])
                and np.array_equal(data["mix"], source_plan["mix"])
                and np.array_equal(data["scale"], scales)
                and float(data["c4_beta"].item()) == beta
            )
    except Exception:
        return False


def _finalize_atlas_preplan(
    recipe: dict, analysis: dict, progress_callback=None
) -> dict:
    context = _validate_atlas_preplan(recipe, analysis)
    routing_path = _routing_path(recipe)
    ensure_dir(str(routing_path))
    identity = routed_recipe_identity(recipe, analysis)
    identity["atlas_preplan"] = {
        "identity_hash": context["index"]["identity_hash"],
        "route_index_sha256": sha256_file(str(context["index_path"])),
        "route_summary_sha256": sha256_file(str(context["summary_path"])),
        "candidate_recipe_sha256": context["candidate_hash"],
        "analysis_identity_sha256": sha256_file(
            str(context["analysis_identity_path"])
        ),
        "atlas_identity_hash": context["index"]["atlas_identity_hash"],
        "source_tensor_metadata_sha256": context["source_metadata_sha256"],
    }
    identity["identity_hash"] = routed_identity_hash(identity)
    identity_hash = identity["identity_hash"]
    _existing_identity_guard(routing_path, identity_hash)
    existing = _preview_is_valid(routing_path, identity_hash)
    if existing is not None:
        return existing

    metrics_dir = routing_path / "metrics"
    plans_dir = routing_path / "plans"
    ensure_dir(str(metrics_dir))
    ensure_dir(str(plans_dir))
    plans = context["plans"]
    metrics: dict[str, dict] = {}
    metric_paths: dict[str, Path] = {}
    pending: dict[str, list[str]] = {}
    names = sorted(plans)
    completed = 0
    try:
        _progress(
            routing_path,
            identity_hash,
            "preplan_finalization",
            completed_tensors=0,
            tensor_total=len(names),
            preplan_identity_hash=context["index"]["identity_hash"],
        )
        for name in names:
            plan = plans[name]
            record = context["index"]["tensors"][name]
            atlas_record = context["atlas_index"]["tensors"][name]
            metric_path = metrics_dir / f"{_tensor_key(name)}.npz"
            metric_paths[name] = metric_path
            metric = _load_finalization_metric(
                metric_path,
                identity_hash,
                name,
                tuple(int(value) for value in plan["shape"]),
                np.asarray(plan["starts"], dtype=np.int64),
                np.asarray(plan["ends"], dtype=np.int64),
                record["plan_sha256"],
                atlas_record["sha256"],
            )
            if metric is not None:
                metrics[name] = metric
                completed += 1
                _progress(
                    routing_path,
                    identity_hash,
                    "preplan_finalization",
                    completed_tensors=completed,
                    tensor_total=len(names),
                    tensor_name=name,
                    resumed=True,
                    preplan_identity_hash=context["index"]["identity_hash"],
                )
                _notify(
                    progress_callback, completed, len(names), name, "preplan_finalization"
                )
            else:
                pending.setdefault(context["source_index"][name], []).append(name)

        for source_path, pending_names in sorted(pending.items()):
            with safe_open(source_path, framework="pt", device="cpu") as source:
                for name in sorted(pending_names):
                    plan = plans[name]
                    record = context["index"]["tensors"][name]
                    atlas_record = context["atlas_index"]["tensors"][name]
                    shape = tuple(int(value) for value in plan["shape"])
                    starts = np.asarray(plan["starts"], dtype=np.int64)
                    ends = np.asarray(plan["ends"], dtype=np.int64)
                    weight_energy = _read_departure_weight_energy(
                        source, name, shape, starts, ends
                    )
                    metric = {
                        "schema_version": np.array(ROUTED_SCHEMA_VERSION, dtype="U16"),
                        "identity_hash": np.array(identity_hash, dtype="U64"),
                        "tensor_name": np.array(name, dtype="U2048"),
                        "preplan_identity_hash": np.array(
                            context["index"]["identity_hash"], dtype="U64"
                        ),
                        "preplan_sha256": np.array(
                            record["plan_sha256"], dtype="U64"
                        ),
                        "atlas_sha256": np.array(
                            atlas_record["sha256"], dtype="U64"
                        ),
                        "shape": np.asarray(shape, dtype=np.int64),
                        "starts": starts,
                        "ends": ends,
                        "raw_update_energy": context["terms"][name][
                            "raw_update_energy"
                        ],
                        "raw_c4_dot": context["terms"][name]["raw_c4_dot"],
                        "c4_energy": context["terms"][name]["c4_energy"],
                        "weight_energy": weight_energy,
                    }
                    _atomic_npz(metric_paths[name], **metric)
                    metrics[name] = metric
                    completed += 1
                    _progress(
                        routing_path,
                        identity_hash,
                        "preplan_finalization",
                        completed_tensors=completed,
                        tensor_total=len(names),
                        tensor_name=name,
                        resumed=False,
                        preplan_identity_hash=context["index"]["identity_hash"],
                    )
                    _notify(
                        progress_callback,
                        completed,
                        len(names),
                        name,
                        "preplan_finalization",
                    )

        settings = _settings(recipe)
        beta = _finalize_projection(
            plans,
            metrics,
            recipe.get("donors", recipe.get("routes", [])),
            settings,
        )
        cap = float(settings["max_rms_ratio"])
        total_leakage = 0.0
        total_c4_energy = 0.0
        maximum_rms_ratio = 0.0
        route_records = {}
        for index, name in enumerate(names, start=1):
            plan = plans[name]
            metric = metrics[name]
            raw_energy = np.asarray(metric["raw_update_energy"], dtype=np.float64)
            raw_dot = np.asarray(metric["raw_c4_dot"], dtype=np.float64)
            c4_energy = np.asarray(metric["c4_energy"], dtype=np.float64)
            weight_energy = np.asarray(metric["weight_energy"], dtype=np.float64)
            scales = np.asarray(plan["scale"], dtype=np.float64)
            accepted = np.asarray(plan["winner"]) >= 0
            corrected = np.maximum(
                raw_energy - 2.0 * beta * raw_dot + beta * beta * c4_energy,
                0.0,
            )
            final_energy = scales * scales * corrected
            if np.any(
                final_energy[accepted]
                > cap * cap * weight_energy[accepted]
                + 1e-12 * np.maximum(weight_energy[accepted], 1.0)
            ):
                raise ValueError(f"Final RMS cap validation failed for {name}")
            positive_weight = accepted & (weight_energy > 0)
            if np.any(positive_weight):
                maximum_rms_ratio = max(
                    maximum_rms_ratio,
                    float(
                        np.max(
                            np.sqrt(
                                final_energy[positive_weight]
                                / weight_energy[positive_weight]
                            )
                        )
                    ),
                )
            total_leakage += float(
                np.sum(scales[accepted] * (raw_dot[accepted] - beta * c4_energy[accepted]))
            )
            total_c4_energy += float(np.sum(c4_energy[accepted]))
            plan_path = plans_dir / f"{_tensor_key(name)}.npz"
            if not _final_plan_matches(
                plan_path, identity_hash, name, plan, scales, beta
            ):
                _atomic_npz(
                    plan_path,
                    schema_version=np.array(ROUTED_SCHEMA_VERSION, dtype="U16"),
                    identity_hash=np.array(identity_hash, dtype="U64"),
                    preplan_identity_hash=np.array(
                        context["index"]["identity_hash"], dtype="U64"
                    ),
                    tensor_name=np.array(name, dtype="U2048"),
                    shape=np.asarray(plan["shape"], dtype=np.int64),
                    starts=np.asarray(plan["starts"], dtype=np.int64),
                    ends=np.asarray(plan["ends"], dtype=np.int64),
                    winner=np.asarray(plan["winner"], dtype=np.int16),
                    second=np.asarray(plan["second"], dtype=np.int16),
                    mix=np.asarray(plan["mix"], dtype=np.float32),
                    scale=scales,
                    c4_beta=np.array(beta, dtype=np.float64),
                )
            route_records[name] = {
                "plan": str(plan_path.relative_to(routing_path)),
                "plan_sha256": sha256_file(str(plan_path)),
                "metrics": str(metric_paths[name].relative_to(routing_path)),
                "metrics_sha256": sha256_file(str(metric_paths[name])),
                "preplan_plan_sha256": context["index"]["tensors"][name][
                    "plan_sha256"
                ],
                "source_atlas_sha256": context["atlas_index"]["tensors"][name][
                    "sha256"
                ],
                "shape": [int(value) for value in plan["shape"]],
                "blocks": int(len(plan["winner"])),
            }
            _progress(
                routing_path,
                identity_hash,
                "planning",
                completed_tensors=index,
                tensor_total=len(names),
                tensor_name=name,
                preplan_identity_hash=context["index"]["identity_hash"],
            )
            _notify(progress_callback, index, len(names), name, "planning")

        leakage_tolerance = 1e-11 * max(
            1.0,
            sum(
                float(
                    np.sum(
                        np.abs(np.asarray(metrics[name]["raw_c4_dot"], dtype=np.float64))
                    )
                )
                for name in names
            ),
        )
        if abs(total_leakage) > leakage_tolerance:
            raise ValueError(
                "Final protected-component projection did not converge "
                f"(leakage={total_leakage})"
            )
        route_index = {
            "schema_version": ROUTED_SCHEMA_VERSION,
            "identity_hash": identity_hash,
            "identity": identity,
            "fine_block_size": int(settings["fine_block_size"]),
            "coarse_block_size": int(settings["coarse_block_size"]),
            "donor_labels": context["donor_labels"],
            "residual_coefficients": context["coefficients"].tolist(),
            "protected_component": int(settings["protected_component"]),
            "c4_beta": beta,
            "atlas_preplan_identity_hash": context["index"]["identity_hash"],
            "departure_shards": identity["departure_shards"],
            "tensors": route_records,
        }
        _atomic_json(routing_path / "route_index.json", route_index)
        source_summary = context["summary"]
        summary = {
            "schema_version": ROUTED_SCHEMA_VERSION,
            "identity_hash": identity_hash,
            "status": "complete",
            "routing_path": str(routing_path),
            "tensor_count": len(names),
            **context["stats"],
            "conflicts": int(source_summary["conflicts"]),
            "protection_energy_fraction": float(
                source_summary["protection_energy_fraction"]
            ),
            "protection_threshold": source_summary["protection_threshold"],
            "protected_component": int(settings["protected_component"]),
            "c4_beta": beta,
            "precast_c4_leakage": total_leakage,
            "maximum_update_rms_ratio": maximum_rms_ratio,
            "atlas_preplan_identity_hash": context["index"]["identity_hash"],
            "finalization_source": "departure_weights_and_cached_atlas_gram",
            "completed_at": _utc_now(),
        }
        _atomic_json(routing_path / "route_preview.json", summary)
        _progress(
            routing_path,
            identity_hash,
            "complete",
            completed_tensors=len(names),
            tensor_total=len(names),
            preplan_identity_hash=context["index"]["identity_hash"],
        )
        return summary
    except Exception as error:
        _progress(
            routing_path,
            identity_hash,
            "failed",
            error=f"{type(error).__name__}: {error}",
            preplan_identity_hash=context["index"]["identity_hash"],
        )
        raise


def _existing_identity_guard(routing_path: Path, identity_hash: str) -> None:
    for filename in ("routing_progress.json", "route_preview.json", "route_index.json"):
        path = routing_path / filename
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except Exception:
            continue
        prior = existing.get("identity_hash")
        if prior and prior != identity_hash:
            raise ValueError(
                f"Routing identity mismatch in {filename}; use a new routing_path"
            )


def _preview_is_valid(routing_path: Path, identity_hash: str) -> dict | None:
    preview_path = routing_path / "route_preview.json"
    index_path = routing_path / "route_index.json"
    if not preview_path.exists() or not index_path.exists():
        return None
    try:
        with open(preview_path, "r", encoding="utf-8") as handle:
            preview = json.load(handle)
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        if (
            preview.get("identity_hash") != identity_hash
            or index.get("identity_hash") != identity_hash
        ):
            raise ValueError("Routing preview identity mismatch")
        for record in index.get("tensors", {}).values():
            path = routing_path / record["plan"]
            if not path.exists() or sha256_file(str(path)) != record["plan_sha256"]:
                return None
            if record.get("metrics"):
                metric_path = routing_path / record["metrics"]
                if (
                    not metric_path.exists()
                    or sha256_file(str(metric_path)) != record.get("metrics_sha256")
                ):
                    return None
        return preview
    except ValueError:
        raise
    except Exception:
        return None


def preview_routed_recipe(recipe: dict, progress_callback=None) -> dict:
    analysis = _analysis_for(recipe)
    validate_routed_recipe(recipe, analysis)
    if "atlas_preplan" in recipe:
        return _finalize_atlas_preplan(recipe, analysis, progress_callback)
    routing_path = _routing_path(recipe)
    ensure_dir(str(routing_path))
    identity = routed_recipe_identity(recipe, analysis)
    identity_hash = identity["identity_hash"]
    _existing_identity_guard(routing_path, identity_hash)
    existing = _preview_is_valid(routing_path, identity_hash)
    if existing is not None:
        return existing

    donors = recipe.get("donors", recipe.get("routes", []))
    donor_labels = [_donor_label(donor) for donor in donors]
    coefficients = compute_residual_coefficients(
        analysis,
        recipe["departure_label"],
        donor_labels,
        float(recipe.get("residual_regularization", 1e-8)),
    )
    models, shared_names = _shared_context(recipe, analysis)
    metrics_dir = routing_path / "metrics"
    plans_dir = routing_path / "plans"
    ensure_dir(str(metrics_dir))
    ensure_dir(str(plans_dir))
    metrics: dict[str, dict] = {}
    metric_files: dict[str, str] = {}
    try:
        _progress(
            routing_path,
            identity_hash,
            "metrics",
            completed_tensors=0,
            tensor_total=len(shared_names),
        )
        for index, name in enumerate(shared_names, start=1):
            metric_path = metrics_dir / f"{_tensor_key(name)}.npz"
            metric = _load_metric(metric_path, identity_hash, name)
            if metric is None:
                metric = _compute_metric(
                    recipe, analysis, models, name, coefficients, identity_hash
                )
                _atomic_npz(metric_path, **metric)
            metrics[name] = metric
            metric_files[name] = str(metric_path.relative_to(routing_path))
            _progress(
                routing_path,
                identity_hash,
                "metrics",
                completed_tensors=index,
                tensor_total=len(shared_names),
                tensor_name=name,
            )
            _notify(progress_callback, index, len(shared_names), name, "metrics")

        settings = _settings(recipe)
        fraction = float(settings["protection_energy_fraction"])
        routable_names = [
            name for name in shared_names if not _is_hard_excluded(name, settings)
        ]
        all_energy = np.concatenate(
            [
                np.asarray(metrics[name]["dep_energy"], dtype=np.float64)
                for name in routable_names
            ]
        )
        positive = all_energy[all_energy > 0]
        if fraction <= 0 or positive.size == 0:
            protection_threshold = math.inf
        else:
            descending = np.sort(positive)[::-1]
            position = int(
                np.searchsorted(
                    np.cumsum(descending),
                    fraction * float(np.sum(descending)),
                    side="left",
                )
            )
            protection_threshold = float(descending[min(position, len(descending) - 1)])

        _progress(
            routing_path,
            identity_hash,
            "planning",
            completed_tensors=0,
            tensor_total=len(shared_names),
        )
        plans: dict[str, dict] = {}
        per_tensor_stats: dict[str, dict] = {}
        for name in shared_names:
            plans[name], per_tensor_stats[name] = _initial_plan(
                metrics[name], donors, settings, protection_threshold, name
            )
        beta = _finalize_projection(plans, metrics, donors, settings)
        route_records = {}
        winner_counts = {label: 0 for label in donor_labels}
        total_stats = {
            "blocks": 0,
            "accepted": 0,
            "protected": 0,
            "excluded": 0,
            "conflicts": 0,
        }
        for index, name in enumerate(shared_names, start=1):
            plan = plans[name]
            metric = metrics[name]
            for donor_index, label in enumerate(donor_labels):
                winner_counts[label] += int(np.count_nonzero(plan["winner"] == donor_index))
            for key in total_stats:
                total_stats[key] += int(per_tensor_stats[name][key])
            plan_path = plans_dir / f"{_tensor_key(name)}.npz"
            _atomic_npz(
                plan_path,
                schema_version=np.array(ROUTED_SCHEMA_VERSION, dtype="U16"),
                identity_hash=np.array(identity_hash, dtype="U64"),
                tensor_name=np.array(name, dtype="U2048"),
                shape=np.asarray(metric["shape"], dtype=np.int64),
                starts=np.asarray(metric["starts"], dtype=np.int64),
                ends=np.asarray(metric["ends"], dtype=np.int64),
                winner=np.asarray(plan["winner"], dtype=np.int16),
                second=np.asarray(plan["second"], dtype=np.int16),
                mix=np.asarray(plan["mix"], dtype=np.float32),
                scale=np.asarray(plan["scale"], dtype=np.float64),
                c4_beta=np.array(beta, dtype=np.float64),
            )
            route_records[name] = {
                "plan": str(plan_path.relative_to(routing_path)),
                "plan_sha256": sha256_file(str(plan_path)),
                "metrics": metric_files[name],
                "metrics_sha256": sha256_file(str(routing_path / metric_files[name])),
                "shape": [int(value) for value in metric["shape"]],
                "blocks": int(len(plan["winner"])),
            }
            _progress(
                routing_path,
                identity_hash,
                "planning",
                completed_tensors=index,
                tensor_total=len(shared_names),
                tensor_name=name,
            )
            _notify(progress_callback, index, len(shared_names), name, "planning")

        route_index = {
            "schema_version": ROUTED_SCHEMA_VERSION,
            "identity_hash": identity_hash,
            "identity": identity,
            "fine_block_size": int(settings["fine_block_size"]),
            "coarse_block_size": int(settings["coarse_block_size"]),
            "donor_labels": donor_labels,
            "residual_coefficients": coefficients.tolist(),
            "protected_component": int(settings["protected_component"]),
            "c4_beta": beta,
            "tensors": route_records,
        }
        _atomic_json(routing_path / "route_index.json", route_index)
        summary = {
            "schema_version": ROUTED_SCHEMA_VERSION,
            "identity_hash": identity_hash,
            "status": "complete",
            "routing_path": str(routing_path),
            "tensor_count": len(shared_names),
            **total_stats,
            "winner_counts": winner_counts,
            "protection_energy_fraction": fraction,
            "protection_threshold": (
                None if math.isinf(protection_threshold) else protection_threshold
            ),
            "protected_component": int(settings["protected_component"]),
            "c4_beta": beta,
            "completed_at": _utc_now(),
        }
        _atomic_json(routing_path / "route_preview.json", summary)
        _progress(
            routing_path,
            identity_hash,
            "complete",
            completed_tensors=len(shared_names),
            tensor_total=len(shared_names),
        )
        return summary
    except Exception as error:
        _progress(
            routing_path,
            identity_hash,
            "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise


def load_route_preview(routing_path: str) -> dict | None:
    root = relocated_path(routing_path)
    preview_path = root / "route_preview.json"
    index_path = root / "route_index.json"
    if not preview_path.exists() or not index_path.exists():
        return None
    try:
        with open(preview_path, "r", encoding="utf-8") as handle:
            preview = json.load(handle)
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        if (
            preview.get("status") != "complete"
            or preview.get("identity_hash") != index.get("identity_hash")
        ):
            return None
        return preview
    except (OSError, json.JSONDecodeError):
        return None


def _load_plan(
    routing_path: Path, record: dict, identity_hash: str, tensor_name: str
) -> dict:
    path = routing_path / record["plan"]
    if sha256_file(str(path)) != record["plan_sha256"]:
        raise ValueError(f"Route plan hash mismatch for {tensor_name}")
    with np.load(path, allow_pickle=False) as data:
        if (
            str(data["identity_hash"].item()) != identity_hash
            or str(data["tensor_name"].item()) != tensor_name
        ):
            raise ValueError(f"Route plan identity mismatch for {tensor_name}")
        return {key: np.array(data[key], copy=True) for key in data.files}


def _compose_tensor(
    source_tensor: torch.Tensor,
    output_dtype: torch.dtype,
    name: str,
    plan: dict,
    models: list[dict],
    coefficients: np.ndarray,
    donor_alphas: np.ndarray,
    left: np.ndarray,
    singular: np.ndarray,
    protected_component: int,
) -> tuple[torch.Tensor, float, float]:
    shape = tuple(source_tensor.shape)
    if tuple(int(value) for value in plan["shape"]) != shape:
        raise ValueError(f"Route plan shape mismatch for {name}")
    original_dtype = source_tensor.dtype
    if not source_tensor.is_floating_point() and np.any(plan["winner"] >= 0):
        raise ValueError(f"Cannot route into non-floating tensor {name}")
    if not np.any(plan["winner"] >= 0):
        return source_tensor.to(output_dtype), 0.0, 0.0
    accumulation_dtype = (
        output_dtype
        if output_dtype in {torch.float32, torch.float64}
        else torch.float32
    )
    output = source_tensor.to(accumulation_dtype).contiguous().view(-1)
    source_flat = source_tensor.contiguous().view(-1)
    beta = float(plan["c4_beta"].item())
    k = protected_component - 1
    leakage = 0.0
    c4_total = 0.0
    with ExitStack() as stack:
        handles = [
            stack.enter_context(
                safe_open(
                    str(
                        relocated_path(model["delta_dir"])
                        / model["manifest"].tensors[name].shard_file
                    ),
                    framework="pt",
                    device="cpu",
                )
            )
            for model in models
        ]
        for block, (start, end) in enumerate(zip(plan["starts"], plan["ends"])):
            first = int(plan["winner"][block])
            if first < 0:
                continue
            start_i, end_i = int(start), int(end)
            deltas = np.empty((len(models), end_i - start_i), dtype=np.float64)
            for model_index, handle in enumerate(handles):
                deltas[model_index] = read_flat_slice(
                    handle, name, shape, start_i, end_i
                )
            residual_first = coefficients[first] @ deltas
            mix = float(plan["mix"][block])
            update = donor_alphas[first] * (1.0 - mix) * residual_first
            other = int(plan["second"][block])
            if other >= 0 and mix > 0:
                update += donor_alphas[other] * mix * (coefficients[other] @ deltas)
            if 0 <= k < left.shape[1] and singular[k] > 0:
                c4 = (left[:, k] @ deltas) / singular[k]
                update -= beta * c4
            else:
                c4 = np.zeros(end_i - start_i, dtype=np.float64)
            update *= float(plan["scale"][block])
            if not np.all(np.isfinite(update)):
                raise ValueError(f"Non-finite routed update for {name}")
            output[start_i:end_i] += torch.from_numpy(update).to(accumulation_dtype)
            actual = (
                output[start_i:end_i].to(output_dtype).to(torch.float64)
                - source_flat[start_i:end_i].to(output_dtype).to(torch.float64)
            ).numpy()
            leakage += float(actual @ c4)
            c4_total += float(c4 @ c4)
    if not torch.isfinite(output).all():
        raise ValueError(f"Non-finite composed tensor {name}")
    return output.reshape(shape).to(output_dtype), leakage, c4_total


_DTYPE_CODES = {
    torch.bool: ("BOOL", 1),
    torch.uint8: ("U8", 1),
    torch.int8: ("I8", 1),
    torch.int16: ("I16", 2),
    torch.uint16: ("U16", 2),
    torch.int32: ("I32", 4),
    torch.uint32: ("U32", 4),
    torch.int64: ("I64", 8),
    torch.uint64: ("U64", 8),
    torch.float16: ("F16", 2),
    torch.bfloat16: ("BF16", 2),
    torch.float32: ("F32", 4),
    torch.float64: ("F64", 8),
}


def _atomic_stream_safetensors(
    output_path: Path,
    specs: list[tuple[str, tuple[int, ...], torch.dtype]],
    metadata: dict[str, str] | None,
    producer,
) -> str:
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(key): str(value) for key, value in metadata.items()}
    offset = 0
    for name, shape, dtype in specs:
        if dtype not in _DTYPE_CODES:
            raise ValueError(f"Unsupported source dtype for {name}: {dtype}")
        code, size = _DTYPE_CODES[dtype]
        end = offset + tensor_numel(shape) * size
        header[name] = {"dtype": code, "shape": list(shape), "data_offsets": [offset, end]}
        offset = end
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    leakage = 0.0
    c4_energy = 0.0
    with open(tmp, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for name, shape, dtype in specs:
            tensor, local_leakage, local_c4 = producer(name, dtype)
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"Produced tensor metadata mismatch for {name}")
            raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            handle.write(raw)
            leakage += local_leakage
            c4_energy += local_c4
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(output_path))
    return sha256_file(str(output_path)), leakage, c4_energy


def _copy_metadata(base_dir: str, output_dir: str) -> None:
    allowed = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
    }
    for name in allowed:
        source = Path(base_dir) / name
        if source.exists():
            shutil.copy2(source, Path(output_dir) / name)


def compose_routed_recipe(recipe_path: str, dry_run: bool = False) -> None:
    with open(recipe_path, "r", encoding="utf-8") as handle:
        recipe = json.load(handle)
    analysis = _analysis_for(recipe)
    validate_routed_recipe(recipe, analysis)
    summary = preview_routed_recipe(recipe)
    routing_path = _routing_path(recipe)
    with open(routing_path / "route_index.json", "r", encoding="utf-8") as handle:
        route_index = json.load(handle)
    identity_hash = route_index["identity_hash"]
    output_path = relocated_path(recipe["output_path"])
    ensure_dir(str(output_path))
    resolved = {
        "schema_version": ROUTED_SCHEMA_VERSION,
        "original_recipe": _jsonable(recipe),
        "routing_identity": identity_hash,
        "routing_summary": summary,
        "donor_labels": route_index["donor_labels"],
        "residual_coefficients": route_index["residual_coefficients"],
        "c4_beta": route_index["c4_beta"],
    }
    if dry_run:
        _atomic_json(output_path / "resolved_recipe.json", resolved)
        return

    models, _ = _shared_context(recipe, analysis)
    coefficients = np.asarray(route_index["residual_coefficients"], dtype=np.float64)
    donors = recipe.get("donors", recipe.get("routes", []))
    donor_alphas = np.asarray([_donor_alpha(item) for item in donors], dtype=np.float64)
    left = np.asarray(analysis["left_vectors"], dtype=np.float64)
    singular = np.asarray(analysis["singular_values"], dtype=np.float64)
    protected_component = int(route_index["protected_component"])
    source_index = load_tensor_index(recipe["departure_path"])
    source_groups: dict[str, list[str]] = {}
    source_paths: dict[str, str] = {}
    for name, path in source_index.items():
        filename = os.path.basename(path)
        source_groups.setdefault(filename, []).append(name)
        source_paths[filename] = path
    source_signatures = {}
    for filename, source_path in source_paths.items():
        stat = Path(source_path).stat()
        source_signatures[filename] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
    compose_identity = hashlib.sha256(
        _canonical_bytes(
            {
                "routing_identity": identity_hash,
                "source_signatures": source_signatures,
                "dtype": recipe.get("dtype", "source"),
            }
        )
    ).hexdigest()
    state_path = output_path / "compose_state.json"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("identity_hash") != compose_identity:
            raise ValueError("Compose state identity mismatch; use a new output_path")
    else:
        state = {
            "schema_version": ROUTED_SCHEMA_VERSION,
            "identity_hash": compose_identity,
            "routing_identity": identity_hash,
            "source_shards": source_signatures,
            "shards": {},
        }
    _atomic_json(output_path / "resolved_recipe.json", resolved)
    dtype_setting = recipe.get("dtype", "source")
    target_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_setting)
    weight_map = {}
    total_leakage = 0.0
    total_c4_energy = 0.0
    try:
        _progress(
            routing_path,
            identity_hash,
            "composing",
            completed_shards=0,
            shard_total=len(source_groups),
        )
        for shard_index, filename in enumerate(sorted(source_groups), start=1):
            names = sorted(source_groups[filename])
            output_shard = output_path / filename
            record = state["shards"].get(filename)
            if (
                record
                and output_shard.exists()
                and sha256_file(str(output_shard)) == record.get("sha256")
                and record.get("source_signature") == source_signatures[filename]
            ):
                for name in names:
                    weight_map[name] = filename
                total_leakage += float(record.get("c4_leakage", 0.0))
                total_c4_energy += float(record.get("c4_energy", 0.0))
                continue
            source_path = source_paths[filename]
            with safe_open(source_path, framework="pt", device="cpu") as source:
                metadata = source.metadata()
                specs = []
                source_dtypes = {}
                for name in names:
                    tensor = source.get_tensor(name)
                    source_dtypes[name] = tensor.dtype
                    specs.append(
                        (
                            name,
                            tuple(tensor.shape),
                            target_dtype if target_dtype is not None else tensor.dtype,
                        )
                    )
                    del tensor

                def produce(name: str, requested_dtype: torch.dtype):
                    source_tensor = source.get_tensor(name)
                    route_record = route_index["tensors"].get(name)
                    if route_record is None:
                        result = source_tensor
                        leakage = 0.0
                        c4_energy = 0.0
                    else:
                        plan = _load_plan(
                            routing_path, route_record, identity_hash, name
                        )
                        result, leakage, c4_energy = _compose_tensor(
                            source_tensor,
                            requested_dtype,
                            name,
                            plan,
                            models,
                            coefficients,
                            donor_alphas,
                            left,
                            singular,
                            protected_component,
                        )
                    return result.to(requested_dtype), leakage, c4_energy

                shard_sha, leakage, c4_energy = _atomic_stream_safetensors(
                    output_shard, specs, metadata, produce
                )
            state["shards"][filename] = {
                "sha256": shard_sha,
                "source_signature": source_signatures[filename],
                "c4_leakage": leakage,
                "c4_energy": c4_energy,
            }
            _atomic_json(state_path, state)
            total_leakage += leakage
            total_c4_energy += c4_energy
            for name in names:
                weight_map[name] = filename
            _progress(
                routing_path,
                identity_hash,
                "composing",
                completed_shards=shard_index,
                shard_total=len(source_groups),
                shard_name=filename,
            )

        if len(source_groups) > 1:
            index_payload = {
                "metadata": {
                    "total_size": sum(
                        (output_path / filename).stat().st_size
                        for filename in source_groups
                    )
                },
                "weight_map": weight_map,
            }
            _atomic_json(output_path / "model.safetensors.index.json", index_payload)
        _copy_metadata(str(relocated_path(recipe["departure_path"])), str(output_path))
        provenance = {
            "base_dir": os.path.abspath(relocated_path(recipe["departure_path"])),
            "dtype": dtype_setting,
            "missing_policy": "error",
            "created_at": _utc_now(),
            "routed": {
                "schema_version": ROUTED_SCHEMA_VERSION,
                "identity_hash": identity_hash,
                "routing_path": os.path.abspath(str(routing_path)),
                "donor_labels": route_index["donor_labels"],
                "c4_beta": route_index["c4_beta"],
                "achieved_c4_leakage": total_leakage,
                "c4_energy": total_c4_energy,
                "normalized_c4_leakage": (
                    total_leakage / math.sqrt(total_c4_energy)
                    if total_c4_energy > 0
                    else 0.0
                ),
                "preview": summary,
            },
        }
        if not math.isfinite(provenance["routed"]["normalized_c4_leakage"]):
            raise ValueError("Non-finite achieved C4 leakage")
        _atomic_json(output_path / "provenance.json", provenance)
        state["complete"] = True
        state["completed_at"] = _utc_now()
        _atomic_json(state_path, state)
        _progress(
            routing_path,
            identity_hash,
            "complete",
            completed_shards=len(source_groups),
            shard_total=len(source_groups),
        )
    except Exception as error:
        _progress(
            routing_path,
            identity_hash,
            "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
