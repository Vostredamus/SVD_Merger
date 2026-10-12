"""Derived analysis and atlas-only route preplanning.

This module never opens model or delta shards.  Every local measurement is
derived from the block Gram matrices in a completed local atlas.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .atlas import _pid_alive
from .recipes import load_analysis
from .paths import relocated_path
from .routing import (
    ROUTED_SCHEMA_VERSION,
    _arch,
    _initial_plan,
    _is_hard_excluded,
    _raw_terms,
    _route_prior,
    _scores,
    _settings,
    compute_residual_coefficients,
)
from .utils import sha256_file


ATLAS_ANALYSIS_SCHEMA_VERSION = "1.0"
ATLAS_PREPLAN_SCHEMA_VERSION = "1.0"
DEFAULT_DEPARTURE_LABEL: str | None = None
DEFAULT_PROTECTION_FRACTION = 0.90
DEFAULT_CONFLICT_THRESHOLD = -0.10
HARD_EXCLUSIONS = ("embed_tokens", "lm_head", "norm", "layernorm")


def _now() -> str:
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
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _atomic_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _tensor_key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def _layer(name: str) -> int:
    from .routing import _layer as routing_layer

    value = routing_layer(name)
    return -1 if value is None else int(value)


def _safe_cosine(gram: np.ndarray) -> np.ndarray:
    energy = np.maximum(np.diagonal(gram, axis1=1, axis2=2), 0.0)
    denominator = np.sqrt(energy[:, :, None] * energy[:, None, :])
    result = np.zeros_like(gram, dtype=np.float64)
    np.divide(gram, denominator, out=result, where=denominator > 0)
    return np.clip(result, -1.0, 1.0)


def derive_block_metrics(
    gram: np.ndarray,
    departure_index: int,
    left_vectors: np.ndarray,
    singular_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derive exact local metrics from a ``[blocks, models, models]`` Gram."""

    gram = np.asarray(gram, dtype=np.float64)
    left = np.asarray(left_vectors, dtype=np.float64)
    singular = np.asarray(singular_values, dtype=np.float64)
    if gram.ndim != 3 or gram.shape[1] != gram.shape[2]:
        raise ValueError("Local Gram must have shape [blocks, models, models]")
    blocks, models, _ = gram.shape
    if not 0 <= departure_index < models:
        raise ValueError("departure_index is outside the model dimension")
    if left.ndim != 2 or left.shape[0] != models or singular.shape != (left.shape[1],):
        raise ValueError("SVD coefficient dimensions do not match the local Gram")
    if not np.all(np.isfinite(gram)):
        raise ValueError("Local Gram contains non-finite values")

    energy = np.maximum(np.diagonal(gram, axis1=1, axis2=2), 0.0)
    norm = np.sqrt(energy)
    cosine = _safe_cosine(gram)
    departure_energy = energy[:, departure_index]
    departure_dot = gram[:, departure_index, :]

    projection_energy = np.zeros((blocks, models), dtype=np.float64)
    np.divide(
        departure_dot * departure_dot,
        departure_energy[:, None],
        out=projection_energy,
        where=departure_energy[:, None] > 0,
    )
    novelty_energy = np.maximum(energy - projection_energy, 0.0)
    novelty_fraction = np.zeros_like(novelty_energy)
    np.divide(
        novelty_energy,
        energy,
        out=novelty_fraction,
        where=energy > 0,
    )

    coefficients = np.zeros_like(left)
    np.divide(
        left,
        singular[None, :],
        out=coefficients,
        where=singular[None, :] > 0,
    )
    component_energy = np.einsum(
        "mk,bmn,nk->bk", coefficients, gram, coefficients, optimize=True
    )
    component_energy = np.maximum(component_energy, 0.0)
    model_component_dot = np.einsum(
        "bmn,nk->bmk", gram, coefficients, optimize=True
    )
    affinity_denominator = np.sqrt(
        energy[:, :, None] * component_energy[:, None, :]
    )
    component_affinity = np.zeros_like(model_component_dot)
    np.divide(
        model_component_dot,
        affinity_denominator,
        out=component_affinity,
        where=affinity_denominator > 0,
    )
    return {
        "energy": energy,
        "norm": norm,
        "pair_cosine": np.clip(cosine, -1.0, 1.0),
        "pair_conflict": (cosine < 0).astype(np.uint8),
        "departure_cosine": cosine[:, departure_index, :],
        "departure_conflict": (cosine[:, departure_index, :] < 0).astype(np.uint8),
        "route_conflict": (
            cosine[:, departure_index, :] < DEFAULT_CONFLICT_THRESHOLD
        ).astype(np.uint8),
        "novelty_energy_departure": novelty_energy,
        "novelty_fraction_departure": np.clip(novelty_fraction, 0.0, 1.0),
        "svd_component_energy": component_energy,
        "svd_model_component_dot": model_component_dot,
        "svd_model_component_affinity": np.clip(
            component_affinity, -1.0, 1.0
        ),
    }


def _analysis_identity(
    atlas_dir: Path,
    analysis_dir: Path,
    atlas_index: dict,
    atlas_identity: dict,
    departure_label: str,
    prior_recipe: str | None,
) -> dict:
    svd_path = analysis_dir / "svd_analysis.npz"
    if not svd_path.exists():
        raise ValueError(f"Missing SVD analysis: {svd_path}")
    prior_hash = None
    if prior_recipe:
        prior_path = Path(prior_recipe)
        if not prior_path.exists():
            raise ValueError(f"Missing prior recipe: {prior_path}")
        prior_hash = sha256_file(str(prior_path))
    identity = {
        "schema_version": ATLAS_ANALYSIS_SCHEMA_VERSION,
        "atlas_identity_hash": str(atlas_index["identity_hash"]),
        "atlas_schema_version": str(atlas_index["schema_version"]),
        "atlas_block_size": int(atlas_index["block_size"]),
        "atlas_tensor_count": int(atlas_index["tensor_count"]),
        "atlas_block_count": int(atlas_index["block_count"]),
        "labels": [str(value) for value in atlas_index["labels"]],
        "svd_analysis_sha256": sha256_file(str(svd_path)),
        "departure_label": departure_label,
        "protection_energy_fraction": DEFAULT_PROTECTION_FRACTION,
        "conflict_threshold": DEFAULT_CONFLICT_THRESHOLD,
        "hard_exclusions": list(HARD_EXCLUSIONS),
        "prior_recipe_sha256": prior_hash,
    }
    if atlas_identity.get("identity_hash") != atlas_index.get("identity_hash"):
        raise ValueError("Atlas index and identity hashes differ")
    identity["identity_hash"] = _canonical_hash(identity)
    return identity


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_preflight(
    atlas_dir: Path, analysis: dict, departure_label: str
) -> tuple[dict, dict, int]:
    index_path = atlas_dir / "atlas_index.json"
    identity_path = atlas_dir / "atlas_identity.json"
    if not index_path.exists() or not identity_path.exists():
        raise ValueError("Atlas must contain atlas_index.json and atlas_identity.json")
    atlas_index = _read_json(index_path)
    atlas_identity = _read_json(identity_path)
    labels = [str(value) for value in atlas_index.get("labels", [])]
    if labels != [str(value) for value in analysis.get("labels", [])]:
        raise ValueError("Atlas and SVD label order differ")
    if departure_label not in labels:
        raise ValueError(f"Unknown departure label: {departure_label}")
    if int(atlas_index.get("tensor_count", -1)) != len(
        atlas_index.get("tensors", {})
    ):
        raise ValueError("Atlas tensor count is inconsistent")
    return atlas_index, atlas_identity, labels.index(departure_label)


def _load_atlas_tensor(
    atlas_dir: Path,
    analysis_identity_hash: str,
    atlas_identity_hash: str,
    name: str,
    record: dict,
    model_count: int,
) -> tuple[dict[str, np.ndarray], str]:
    path = atlas_dir / record["file"]
    if not path.exists():
        raise ValueError(f"Missing atlas tensor: {path}")
    digest = sha256_file(str(path))
    if digest != record.get("sha256"):
        raise ValueError(f"Atlas tensor SHA-256 mismatch for {name}")
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["identity_hash"].item()) != atlas_identity_hash:
                raise ValueError(f"Atlas tensor identity mismatch for {name}")
            if str(data["tensor_name"].item()) != name:
                raise ValueError(f"Atlas tensor name mismatch for {name}")
            result = {key: np.array(data[key], copy=True) for key in data.files}
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Cannot read atlas tensor {name}: {error}") from error
    gram = np.asarray(result.get("gram"), dtype=np.float64)
    starts = np.asarray(result.get("starts"), dtype=np.int64)
    ends = np.asarray(result.get("ends"), dtype=np.int64)
    if (
        gram.shape != (int(record["blocks"]), model_count, model_count)
        or starts.shape != (int(record["blocks"]),)
        or ends.shape != starts.shape
        or int(result.get("completed_blocks", -1)) != len(starts)
        or not np.all(np.isfinite(gram))
        or not np.allclose(gram, np.swapaxes(gram, 1, 2), rtol=1e-9, atol=1e-9)
    ):
        raise ValueError(f"Atlas tensor payload is invalid for {name}")
    result["analysis_identity_hash"] = np.array(
        analysis_identity_hash, dtype="U64"
    )
    return result, digest


_DERIVED_ARRAYS = {
    "energy",
    "norm",
    "pair_cosine",
    "pair_conflict",
    "departure_cosine",
    "departure_conflict",
    "route_conflict",
    "novelty_energy_departure",
    "novelty_fraction_departure",
    "svd_component_energy",
    "svd_model_component_dot",
    "svd_model_component_affinity",
}


def _load_derived(
    path: Path,
    identity_hash: str,
    source_sha256: str,
    name: str,
    blocks: int,
    models: int,
    components: int,
) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["identity_hash"].item()) != identity_hash:
                raise ValueError(f"Derived atlas identity mismatch for {name}")
            if str(data["source_sha256"].item()) != source_sha256:
                raise ValueError(f"Derived atlas source mismatch for {name}")
            if str(data["tensor_name"].item()) != name:
                raise ValueError(f"Derived atlas tensor mismatch for {name}")
            result = {key: np.array(data[key], copy=True) for key in data.files}
        expected = {
            "energy": (blocks, models),
            "norm": (blocks, models),
            "pair_cosine": (blocks, models, models),
            "pair_conflict": (blocks, models, models),
            "departure_cosine": (blocks, models),
            "departure_conflict": (blocks, models),
            "route_conflict": (blocks, models),
            "novelty_energy_departure": (blocks, models),
            "novelty_fraction_departure": (blocks, models),
            "svd_component_energy": (blocks, components),
            "svd_model_component_dot": (blocks, models, components),
            "svd_model_component_affinity": (blocks, models, components),
        }
        if any(np.asarray(result[key]).shape != shape for key, shape in expected.items()):
            return None
        if any(
            not np.all(np.isfinite(result[key]))
            for key in _DERIVED_ARRAYS
            if key not in {"pair_conflict", "departure_conflict", "route_conflict"}
        ):
            return None
        return result
    except ValueError as error:
        if "identity mismatch" in str(error) or "source mismatch" in str(error):
            raise
        return None
    except Exception:
        return None


def _write_progress(output: Path, identity_hash: str, **values: Any) -> None:
    payload = {
        "schema_version": ATLAS_ANALYSIS_SCHEMA_VERSION,
        "identity_hash": identity_hash,
        "pid": os.getpid(),
        "updated_at": _now(),
        **values,
    }
    _atomic_json(output / "analysis_progress.json", payload)


def _protection_threshold(energies: list[np.ndarray], fraction: float) -> float:
    if fraction <= 0 or not energies:
        return math.inf
    positive = np.concatenate(energies)
    positive = positive[positive > 0]
    if positive.size == 0:
        return math.inf
    descending = np.sort(positive)[::-1]
    position = int(
        np.searchsorted(
            np.cumsum(descending), fraction * float(np.sum(descending)), side="left"
        )
    )
    return float(descending[min(position, len(descending) - 1)])


def _component_coefficients(analysis: dict) -> np.ndarray:
    left = np.asarray(analysis["left_vectors"], dtype=np.float64)
    singular = np.asarray(analysis["singular_values"], dtype=np.float64)
    result = np.zeros_like(left)
    np.divide(left, singular[None, :], out=result, where=singular[None, :] > 0)
    return result


def _quadratic_energy(grams: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.einsum(
            "dm,gmn,dn->gd",
            coefficients,
            grams,
            coefficients,
            optimize=True,
        ),
        0.0,
    )


def _direct_residual_energy(
    grams: np.ndarray, departure_index: int
) -> np.ndarray:
    diagonal = np.maximum(np.diagonal(grams, axis1=-2, axis2=-1), 0.0)
    departure_energy = diagonal[..., departure_index]
    dots = grams[..., departure_index, :]
    projection = np.zeros_like(diagonal)
    np.divide(
        dots * dots,
        departure_energy[..., None],
        out=projection,
        where=departure_energy[..., None] > 0,
    )
    return np.maximum(diagonal - projection, 0.0)


def _group_summary(
    labels: list[str],
    global_gram: np.ndarray,
    total_novelty: np.ndarray,
    conflict_energy: np.ndarray,
    route_conflict_energy: np.ndarray,
    departure_index: int,
) -> list[dict]:
    energy = np.maximum(np.diag(global_gram), 0.0)
    dep_energy = float(energy[departure_index])
    summaries = []
    for index, label in enumerate(labels):
        denominator = math.sqrt(max(dep_energy * float(energy[index]), 0.0))
        cosine = (
            float(global_gram[departure_index, index]) / denominator
            if denominator
            else 0.0
        )
        novelty_fraction = (
            float(total_novelty[index] / energy[index]) if energy[index] > 0 else 0.0
        )
        conflict_fraction = (
            float(conflict_energy[index] / energy[index]) if energy[index] > 0 else 0.0
        )
        route_conflict_fraction = (
            float(route_conflict_energy[index] / energy[index])
            if energy[index] > 0
            else 0.0
        )
        norm_ratio = (
            math.sqrt(float(energy[index]) / dep_energy) if dep_energy > 0 else 0.0
        )
        evidence_score = (
            math.sqrt(max(novelty_fraction, 0.0))
            * max(0.0, 1.0 - route_conflict_fraction)
            / max(1.0, norm_ratio)
        )
        summaries.append(
            {
                "label": label,
                "energy": float(energy[index]),
                "norm": math.sqrt(float(energy[index])),
                "norm_ratio_to_departure": norm_ratio,
                "departure_cosine": cosine,
                "local_novelty_energy": float(total_novelty[index]),
                "local_novelty_fraction": novelty_fraction,
                "local_conflict_energy_fraction": conflict_fraction,
                "route_conflict_energy_fraction": route_conflict_fraction,
                "evidence_score": evidence_score,
                "is_departure": index == departure_index,
            }
        )
    return summaries


def _find_prior_donors(prior_recipe: str | None, labels: list[str]) -> list[str]:
    if prior_recipe:
        recipe = _read_json(Path(prior_recipe))
        return [
            str(item.get("donor_label", item.get("label", "")))
            for item in recipe.get("donors", [])
            if str(item.get("donor_label", item.get("label", ""))) in labels
        ]
    return []


def _ranked_groups(
    energies: np.ndarray, values: list[Any], allowed: np.ndarray | None = None
) -> list[Any]:
    positions = range(len(values)) if allowed is None else np.flatnonzero(allowed)
    return [
        values[index]
        for index in sorted(
            positions, key=lambda item: (-float(energies[item]), str(values[item]))
        )
        if float(energies[index]) > 0
    ]


def _candidate_recipe(
    candidate_id: str,
    rationale: str,
    donor_order: list[str],
    strength: float,
    architecture_count: int,
    labels: list[str],
    analysis: dict,
    analysis_dir: Path,
    output: Path,
    departure_label: str,
    departure_path: str,
    global_gram: np.ndarray,
    arch_gram: np.ndarray,
    architectures: list[str],
    layer_gram: np.ndarray,
    layers: list[int],
    summaries: dict[str, dict],
    winner_only: bool,
    max_alpha: float = 0.20,
    max_rms_ratio: float = 0.005,
) -> dict:
    coefficients = compute_residual_coefficients(
        analysis, departure_label, donor_order
    )
    component_coefficients = _component_coefficients(analysis)
    component_energy = _quadratic_energy(
        global_gram[None, :, :], component_coefficients.T
    )[0]
    dep_index = labels.index(departure_label)
    departure_norm = math.sqrt(max(float(global_gram[dep_index, dep_index]), 0.0))
    excluded_arch = {
        architecture
        for architecture in architectures
        if any(pattern in architecture.casefold() for pattern in HARD_EXCLUSIONS)
    }
    allowed_arch = np.asarray(
        [architecture not in excluded_arch for architecture in architectures],
        dtype=bool,
    )
    donors = []
    for donor_index, (label, coefficient) in enumerate(
        zip(donor_order, coefficients)
    ):
        residual_energy = max(float(coefficient @ global_gram @ coefficient), 0.0)
        residual_norm = math.sqrt(residual_energy)
        alpha = (
            min(max_alpha, max(0.01, strength * departure_norm / residual_norm))
            if residual_norm > 0
            else 0.01
        )
        arch_energy = _quadratic_energy(
            arch_gram, coefficient[None, :]
        )[:, 0]
        selected_arch = _ranked_groups(
            arch_energy, architectures, allowed=allowed_arch
        )[:architecture_count]
        layer_energy = _quadratic_energy(
            layer_gram, coefficient[None, :]
        )[:, 0]
        layer_allowed = np.asarray([layer >= 0 for layer in layers], dtype=bool)
        boost_layers = _ranked_groups(
            layer_energy, layers, allowed=layer_allowed
        )[:4]
        dots = component_coefficients.T @ global_gram @ coefficient
        denominator = np.sqrt(np.maximum(component_energy * residual_energy, 0.0))
        affinity = np.zeros_like(dots)
        np.divide(np.abs(dots), denominator, out=affinity, where=denominator > 0)
        component_order = sorted(
            range(len(affinity)), key=lambda item: (-float(affinity[item]), item)
        )
        components = [
            index + 1 for index in component_order if index + 1 != 4
        ][:2]
        donor_summary = summaries[label]
        donors.append(
            {
                "donor_label": label,
                "label": label,
                "alpha": round(alpha, 8),
                "components": components,
                "architectures": selected_arch,
                "boost_layers": [int(value) for value in boost_layers],
                "layer_boost": 1.35,
                "require_novel": False,
                "evidence": {
                    "order": donor_index + 1,
                    "sequential_residual_norm": residual_norm,
                    "target_effective_strength": strength,
                    "departure_cosine": donor_summary["departure_cosine"],
                    "local_novelty_fraction": donor_summary[
                        "local_novelty_fraction"
                    ],
                    "route_conflict_energy_fraction": donor_summary[
                        "route_conflict_energy_fraction"
                    ],
                    "component_affinities": {
                        f"C{component}": float(affinity[component - 1])
                        for component in components
                    },
                    "top_architecture_residual_energy": {
                        architecture: float(arch_energy[architectures.index(architecture)])
                        for architecture in selected_arch
                    },
                },
            }
        )
    return {
        "schema": "routed_recipe",
        "version": ROUTED_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "candidate_rationale": rationale,
        "analysis_dir": str(analysis_dir),
        "departure_label": departure_label,
        "departure_path": departure_path,
        "output_path": str(output / "candidate_outputs" / candidate_id),
        "routing_path": str(output / "runtime_routing" / candidate_id),
        "dtype": "source",
        "missing_policy": "error",
        "donors": donors,
        "routing": {
            "fine_block_size": int(analysis["_atlas_block_size"]),
            "coarse_block_size": int(analysis["_atlas_block_size"]) * 4,
            "protection_energy_fraction": DEFAULT_PROTECTION_FRACTION,
            "hard_exclusions": list(HARD_EXCLUSIONS),
            "conflict_threshold": DEFAULT_CONFLICT_THRESHOLD,
            "winner_only": winner_only,
            "blend_cosine_threshold": 0.25,
            "blend_score_ratio": 0.80,
            "require_coarse_stability": True,
            "max_rms_ratio": max_rms_ratio,
            "novelty_ratio": 0.10,
            "protected_component": min(4, int(analysis["n_components"])),
            "projection_iterations": 8,
        },
        "atlas_preplan": {
            "schema": "atlas_route_preplan",
            "path": str(output / "route_plans" / candidate_id),
            "requires_weight_rms_finalization": True,
        },
    }


def _design_candidates(
    output: Path,
    analysis: dict,
    analysis_dir: Path,
    labels: list[str],
    departure_label: str,
    summaries_list: list[dict],
    global_gram: np.ndarray,
    arch_gram: np.ndarray,
    architectures: list[str],
    layer_gram: np.ndarray,
    layers: list[int],
    prior_recipe: str | None,
) -> list[dict]:
    summaries = {item["label"]: item for item in summaries_list}
    ranked = [
        item["label"]
        for item in sorted(
            (item for item in summaries_list if not item["is_departure"]),
            key=lambda item: (-item["evidence_score"], item["label"]),
        )
    ]
    prior = _find_prior_donors(prior_recipe, labels)
    prior_supported = [
        label
        for label in prior
        if not summaries[label]["is_departure"]
        and summaries[label]["local_novelty_fraction"] > 0
        and summaries[label]["route_conflict_energy_fraction"] < 0.75
    ]
    legacy_order = prior_supported if len(prior_supported) >= 2 else ranked[:5]
    specs = [
        (
            "conservative_top3",
            "Three strongest measured donors, narrow architecture territories, low effective strength.",
            ranked[:3],
            0.045,
            2,
            True,
        ),
        (
            "balanced_top5",
            "Five strongest measured donors with broader measured territories.",
            ranked[:5],
            0.065,
            3,
            True,
            0.20,
            0.005,
        ),
        (
            "rich_top5_15pct",
            "Assertive five-donor model targeting 15% effective residual strength with a 0.75% local RMS cap.",
            ranked[:5],
            0.150,
            3,
            True,
            0.75,
            0.0075,
        ),
        (
            "simplified_top2",
            "Two-donor control that minimizes sequential residual interactions.",
            ranked[:2],
            0.080,
            3,
            False,
        ),
        (
            "reverse_order_ablation",
            "Balanced donor set in reverse order to measure order sensitivity.",
            list(reversed(ranked[:5])),
            0.055,
            3,
            True,
        ),
        (
            "prior_recipe_remeasured",
            "Donors from --prior-recipe (or the top five measured donors if none given), re-measured against the atlas.",
            legacy_order,
            0.055,
            3,
            True,
        ),
    ]
    target_paths = [str(value) for value in analysis.get("target_paths", [])]
    departure_index = labels.index(departure_label)
    departure_path = (
        target_paths[departure_index]
        if len(target_paths) == len(labels)
        else ""
    )
    if prior_recipe:
        prior_payload = _read_json(Path(prior_recipe))
        if prior_payload.get("departure_label") == departure_label:
            departure_path = str(prior_payload.get("departure_path", departure_path))
    recipes = []
    recipes_dir = output / "candidate_recipes"
    recipes_dir.mkdir(parents=True, exist_ok=True)
    analysis["_atlas_block_size"] = int(analysis["_atlas_block_size"])
    for spec in specs:
        candidate_id, rationale, donors, strength, arch_count, winner_only = spec[:6]
        max_alpha, max_rms_ratio = (
            spec[6:8] if len(spec) == 8 else (0.20, 0.005)
        )
        recipe = _candidate_recipe(
            candidate_id,
            rationale,
            donors,
            strength,
            arch_count,
            labels,
            analysis,
            analysis_dir,
            output,
            departure_label,
            departure_path,
            global_gram,
            arch_gram,
            architectures,
            layer_gram,
            layers,
            summaries,
            winner_only,
            max_alpha=max_alpha,
            max_rms_ratio=max_rms_ratio,
        )
        _atomic_json(recipes_dir / f"{candidate_id}.json", recipe)
        recipes.append(recipe)
    return recipes


def _route_metric_from_gram(
    gram: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    shape: np.ndarray,
    tensor_name: str,
    recipe: dict,
    analysis: dict,
    coefficients: np.ndarray,
) -> dict:
    donors = recipe["donors"]
    dep_index = list(analysis["labels"]).index(recipe["departure_label"])
    residual_energy = np.maximum(
        np.einsum(
            "dm,bmn,dn->bd", coefficients, gram, coefficients, optimize=True
        ),
        0.0,
    )
    residual_dot_departure = np.einsum(
        "dm,bm->bd", coefficients, gram[:, :, dep_index], optimize=True
    )
    pair_dots = np.einsum(
        "dm,bmn,en->bde", coefficients, gram, coefficients, optimize=True
    )
    selected = sorted(
        {
            int(component)
            for donor in donors
            for component in donor.get("components", [])
        }
        | {int(recipe["routing"]["protected_component"])}
    )
    component_coefficients = _component_coefficients(analysis)
    selected_coefficients = component_coefficients[:, [value - 1 for value in selected]]
    component_energy = np.maximum(
        np.einsum(
            "mk,bmn,nk->bk",
            selected_coefficients,
            gram,
            selected_coefficients,
            optimize=True,
        ),
        0.0,
    )
    component_dot = np.einsum(
        "dm,bmn,nk->bdk",
        coefficients,
        gram,
        selected_coefficients,
        optimize=True,
    )
    protected_position = selected.index(int(recipe["routing"]["protected_component"]))
    return {
        "shape": shape,
        "starts": starts,
        "ends": ends,
        "dep_energy": np.maximum(gram[:, dep_index, dep_index], 0.0),
        "residual_energy": residual_energy,
        "residual_dot_departure": residual_dot_departure,
        "pair_dots": pair_dots,
        "selected_components": np.asarray(selected, dtype=np.int32),
        "component_dot": component_dot,
        "component_energy": component_energy,
        "c4_dot": component_dot[:, :, protected_position],
        "c4_energy": component_energy[:, protected_position],
        "route_prior": np.asarray(
            [_route_prior(donor, tensor_name) for donor in donors],
            dtype=np.float64,
        ),
    }


def _load_preplan(
    path: Path,
    identity_hash: str,
    source_sha256: str,
    tensor_name: str,
    blocks: int,
) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if (
                str(data["identity_hash"].item()) != identity_hash
                or str(data["source_sha256"].item()) != source_sha256
                or str(data["tensor_name"].item()) != tensor_name
            ):
                raise ValueError(f"Preplan identity mismatch for {tensor_name}")
            result = {key: np.array(data[key], copy=True) for key in data.files}
        if any(
            np.asarray(result[key]).shape != (blocks,)
            for key in (
                "winner",
                "second",
                "mix",
                "scale",
                "protected",
                "winner_score",
                "second_score",
            )
        ):
            return None
        residual = np.asarray(result.get("sequential_residual_energy"))
        if residual.ndim != 2 or residual.shape[0] != blocks:
            return None
        if not np.all(np.isfinite(result["scale"])):
            return None
        return result
    except ValueError as error:
        if "identity mismatch" in str(error):
            raise
        return None
    except Exception:
        return None


def _build_preplan(
    output: Path,
    atlas_dir: Path,
    atlas_index: dict,
    analysis_identity_hash: str,
    recipe: dict,
    analysis: dict,
    protection_threshold: float,
    progress_callback,
) -> dict:
    candidate_id = str(recipe["candidate_id"])
    plan_root = output / "route_plans" / candidate_id
    plans_dir = plan_root / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    recipe_hash = _canonical_hash(recipe)
    identity = {
        "schema_version": ATLAS_PREPLAN_SCHEMA_VERSION,
        "analysis_identity_hash": analysis_identity_hash,
        "atlas_identity_hash": atlas_index["identity_hash"],
        "recipe_hash": recipe_hash,
        "candidate_id": candidate_id,
    }
    identity["identity_hash"] = _canonical_hash(identity)
    identity_hash = identity["identity_hash"]
    index_path = plan_root / "route_index.json"
    if index_path.exists():
        existing = _read_json(index_path)
        if existing.get("identity_hash") != identity_hash:
            raise ValueError(
                f"Preplan identity mismatch for {candidate_id}; use a new output directory"
            )

    donor_labels = [str(item["donor_label"]) for item in recipe["donors"]]
    coefficients = compute_residual_coefficients(
        analysis, recipe["departure_label"], donor_labels
    )
    settings = _settings(recipe)
    records = {}
    winner_counts = {label: 0 for label in donor_labels}
    totals = {
        "blocks": 0,
        "accepted": 0,
        "protected": 0,
        "excluded": 0,
        "conflicts": 0,
    }
    projection_numerator = 0.0
    projection_denominator = 0.0
    tensor_items = sorted(atlas_index["tensors"].items())
    for tensor_number, (name, record) in enumerate(tensor_items, start=1):
        source, source_sha = _load_atlas_tensor(
            atlas_dir,
            analysis_identity_hash,
            atlas_index["identity_hash"],
            name,
            record,
            len(analysis["labels"]),
        )
        gram = np.asarray(source["gram"], dtype=np.float64)
        metric = _route_metric_from_gram(
            gram,
            np.asarray(source["starts"], dtype=np.int64),
            np.asarray(source["ends"], dtype=np.int64),
            np.asarray(source["shape"], dtype=np.int64),
            name,
            recipe,
            analysis,
            coefficients,
        )
        plan_path = plans_dir / f"{_tensor_key(name)}.npz"
        plan = _load_preplan(
            plan_path, identity_hash, source_sha, name, int(record["blocks"])
        )
        if plan is None:
            plan, stats = _initial_plan(
                metric, recipe["donors"], settings, protection_threshold, name
            )
            scores, _ = _scores(metric, recipe["donors"], settings)
            winner_score = np.zeros(len(plan["winner"]), dtype=np.float64)
            second_score = np.zeros(len(plan["winner"]), dtype=np.float64)
            positions = np.arange(len(plan["winner"]))
            valid = plan["winner"] >= 0
            winner_score[valid] = scores[
                positions[valid], np.asarray(plan["winner"])[valid]
            ]
            second_valid = plan["second"] >= 0
            second_score[second_valid] = scores[
                positions[second_valid], np.asarray(plan["second"])[second_valid]
            ]
            protected = (
                np.asarray(metric["dep_energy"]) >= protection_threshold
                if math.isfinite(protection_threshold)
                else np.zeros(len(plan["winner"]), dtype=bool)
            )
            if float(settings["protection_energy_fraction"]) <= 0:
                protected.fill(False)
            _atomic_npz(
                plan_path,
                schema_version=np.array(ATLAS_PREPLAN_SCHEMA_VERSION, dtype="U16"),
                identity_hash=np.array(identity_hash, dtype="U64"),
                atlas_identity_hash=np.array(
                    atlas_index["identity_hash"], dtype="U64"
                ),
                source_sha256=np.array(source_sha, dtype="U64"),
                tensor_name=np.array(name, dtype="U2048"),
                shape=np.asarray(metric["shape"], dtype=np.int64),
                starts=np.asarray(metric["starts"], dtype=np.int64),
                ends=np.asarray(metric["ends"], dtype=np.int64),
                winner=np.asarray(plan["winner"], dtype=np.int16),
                second=np.asarray(plan["second"], dtype=np.int16),
                mix=np.asarray(plan["mix"], dtype=np.float32),
                scale=np.ones(len(plan["winner"]), dtype=np.float64),
                protected=protected.astype(np.uint8),
                winner_score=winner_score,
                second_score=second_score,
                sequential_residual_energy=np.asarray(
                    metric["residual_energy"], dtype=np.float64
                ),
                scale_status=np.array(
                    "nominal_unscaled_requires_departure_weight_rms", dtype="U64"
                ),
            )
            plan = _load_preplan(
                plan_path, identity_hash, source_sha, name, int(record["blocks"])
            )
            assert plan is not None
        else:
            protected = np.asarray(plan["protected"], dtype=bool)
            conflicts = _scores(metric, recipe["donors"], settings)[1]
            stats = {
                "blocks": len(plan["winner"]),
                "accepted": int(np.count_nonzero(plan["winner"] >= 0)),
                "protected": int(np.count_nonzero(protected)),
                "excluded": int(_is_hard_excluded(name, settings))
                * len(plan["winner"]),
                "conflicts": int(
                    np.count_nonzero(
                        conflicts < float(settings["conflict_threshold"])
                    )
                ),
            }
        raw_energy, raw_dot = _raw_terms(metric, plan, recipe["donors"])
        accepted = np.asarray(plan["winner"]) >= 0
        projection_numerator += float(np.sum(raw_dot[accepted]))
        projection_denominator += float(
            np.sum(np.asarray(metric["c4_energy"])[accepted])
        )
        for donor_index, label in enumerate(donor_labels):
            winner_counts[label] += int(
                np.count_nonzero(np.asarray(plan["winner"]) == donor_index)
            )
        for key in totals:
            totals[key] += int(stats[key])
        records[name] = {
            "plan": str(plan_path.relative_to(plan_root)),
            "plan_sha256": sha256_file(str(plan_path)),
            "source_atlas_file": record["file"],
            "source_atlas_sha256": source_sha,
            "blocks": int(record["blocks"]),
            "shape": [int(value) for value in record["shape"]],
        }
        progress_callback(candidate_id, tensor_number, len(tensor_items), name)

    provisional_beta = (
        projection_numerator / projection_denominator
        if projection_denominator > 0
        else 0.0
    )
    route_index = {
        "schema": "atlas_route_preplan",
        "schema_version": ATLAS_PREPLAN_SCHEMA_VERSION,
        "identity": identity,
        "identity_hash": identity_hash,
        "analysis_identity_hash": analysis_identity_hash,
        "atlas_identity_hash": atlas_index["identity_hash"],
        "atlas_path": str(atlas_dir.resolve()),
        "recipe_hash": recipe_hash,
        "candidate_id": candidate_id,
        "fine_block_size": int(atlas_index["block_size"]),
        "coarse_block_size": int(settings["coarse_block_size"]),
        "donor_labels": donor_labels,
        "residual_coefficients": coefficients.tolist(),
        "protected_component": int(settings["protected_component"]),
        "projection_beta_nominal_unscaled": provisional_beta,
        "scale_status": "nominal_one_not_full_weight_rms_capped",
        "requires_finalization": {
            "departure_weight_rms_scale": True,
            "projection_beta_after_scaling": True,
            "reason": (
                "The atlas contains departure-delta energy, not full departure "
                "weight energy. Exact max_rms_ratio scales and the scale-coupled "
                "projection beta must be finalized while reading departure weights."
            ),
        },
        "tensors": records,
    }
    _atomic_json(index_path, route_index)
    summary = {
        "schema": "atlas_route_preplan_summary",
        "schema_version": ATLAS_PREPLAN_SCHEMA_VERSION,
        "status": "complete",
        "identity_hash": identity_hash,
        "candidate_id": candidate_id,
        **totals,
        "winner_counts": winner_counts,
        "protection_threshold": (
            protection_threshold if math.isfinite(protection_threshold) else None
        ),
        "protection_energy_fraction": float(
            settings["protection_energy_fraction"]
        ),
        "scale_status": route_index["scale_status"],
        "projection_beta_nominal_unscaled": provisional_beta,
    }
    _atomic_json(plan_root / "route_summary.json", summary)
    return summary


def _plot_heatmap(
    path: Path,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    colorbar: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = max(10.0, min(24.0, 0.22 * len(column_labels) + 5.0))
    height = max(5.0, 0.42 * len(row_labels) + 2.5)
    fig, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(
        matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax
    )
    axis.set_title(title)
    axis.set_yticks(range(len(row_labels)))
    axis.set_yticklabels(row_labels, fontsize=7)
    step = max(1, int(math.ceil(len(column_labels) / 40)))
    ticks = list(range(0, len(column_labels), step))
    axis.set_xticks(ticks)
    axis.set_xticklabels(
        [column_labels[index] for index in ticks], rotation=90, fontsize=7
    )
    fig.colorbar(image, ax=axis, label=colorbar, fraction=0.025, pad=0.02)
    fig.tight_layout()
    temporary = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(temporary, dpi=160, format="png")
    plt.close(fig)
    os.replace(temporary, path)


def _make_plots(
    output: Path,
    labels: list[str],
    layers: list[int],
    architectures: list[str],
    layer_gram: np.ndarray,
    layer_conflict_energy: np.ndarray,
    layer_novelty_energy: np.ndarray,
    global_gram: np.ndarray,
    analysis: dict,
    territory_scores: np.ndarray,
    territory_counts: np.ndarray,
    departure_index: int,
) -> list[str]:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    layer_names = ["global" if value < 0 else str(value) for value in layers]
    layer_energy = np.maximum(
        np.diagonal(layer_gram, axis1=1, axis2=2), 0.0
    ).T
    global_energy = np.maximum(np.diag(global_gram), 0.0)
    normalized_energy = np.zeros_like(layer_energy)
    np.divide(
        layer_energy,
        global_energy[:, None],
        out=normalized_energy,
        where=global_energy[:, None] > 0,
    )
    energy_path = plots / "model_energy_landscape.png"
    _plot_heatmap(
        energy_path,
        normalized_energy,
        labels,
        layer_names,
        "Model energy landscape by layer",
        "fraction of each model's total energy",
    )

    dep_energy = np.maximum(layer_gram[:, departure_index, departure_index], 0.0)
    denominator = np.sqrt(dep_energy[:, None] * np.maximum(layer_energy.T, 0.0))
    layer_cosine = np.zeros_like(layer_energy.T)
    np.divide(
        layer_gram[:, departure_index, :],
        denominator,
        out=layer_cosine,
        where=denominator > 0,
    )
    conflict_fraction = np.zeros_like(layer_energy.T)
    np.divide(
        layer_conflict_energy,
        layer_energy.T,
        out=conflict_fraction,
        where=layer_energy.T > 0,
    )
    conflict_path = plots / "departure_model_cosine_conflict.png"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, 1, figsize=(max(12, 0.22 * len(layers) + 5), 9), constrained_layout=True
    )
    for axis, values, title, cmap, minimum, maximum in (
        (
            axes[0],
            layer_cosine.T,
            "departure vs model cosine by layer",
            "RdBu_r",
            -1,
            1,
        ),
        (
            axes[1],
            conflict_fraction.T,
            "Local negative-cosine energy fraction by layer",
            "magma",
            0,
            1,
        ),
    ):
        image = axis.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=minimum,
            vmax=maximum,
        )
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels, fontsize=7)
        step = max(1, int(math.ceil(len(layer_names) / 40)))
        ticks = list(range(0, len(layer_names), step))
        axis.set_xticks(ticks)
        axis.set_xticklabels(
            [layer_names[index] for index in ticks], rotation=90, fontsize=7
        )
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    temporary = conflict_path.with_suffix(".png.tmp")
    fig.savefig(temporary, dpi=160, format="png")
    plt.close(fig)
    os.replace(temporary, conflict_path)

    novelty_fraction = np.zeros_like(layer_energy.T)
    np.divide(
        layer_novelty_energy,
        layer_energy.T,
        out=novelty_fraction,
        where=layer_energy.T > 0,
    )
    novelty_path = plots / "novelty_landscape.png"
    _plot_heatmap(
        novelty_path,
        novelty_fraction.T,
        labels,
        layer_names,
        "Local novelty relative to departure by layer",
        "residual energy fraction",
        vmin=0,
        vmax=1,
    )

    components = _component_coefficients(analysis)
    component_energy = _quadratic_energy(
        global_gram[None, :, :], components.T
    )[0]
    model_energy = np.maximum(np.diag(global_gram), 0.0)
    dots = global_gram @ components
    affinity_denominator = np.sqrt(model_energy[:, None] * component_energy[None, :])
    affinity = np.zeros_like(dots)
    np.divide(dots, affinity_denominator, out=affinity, where=affinity_denominator > 0)
    component_path = plots / "svd_component_affinity.png"
    _plot_heatmap(
        component_path,
        np.clip(affinity, -1, 1),
        labels,
        [f"C{index + 1}" for index in range(components.shape[1])],
        "Global affinity of models to SVD components",
        "cosine affinity",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    territory = np.argmax(territory_scores, axis=2)
    no_winner = np.max(territory_scores, axis=2) <= 0
    territory = territory.astype(np.int16)
    territory[no_winner] = -1
    territory_path = plots / "donor_territory.png"
    territory_rows = [
        "global" if value < 0 else f"layer {value}" for value in layers
    ]
    _plot_heatmap(
        territory_path,
        territory,
        territory_rows,
        architectures,
        "Atlas donor territory winner by layer and architecture",
        "model index (-1 = protected/unassigned)",
        cmap="tab20",
        vmin=-1,
        vmax=max(len(labels) - 1, 1),
    )
    return [
        str(path.relative_to(output))
        for path in (
            energy_path,
            conflict_path,
            novelty_path,
            component_path,
            territory_path,
        )
    ]


def _write_candidate_report(
    output: Path,
    summaries: list[dict],
    recipes: list[dict],
    plan_summaries: list[dict],
    threshold: float,
) -> None:
    plan_by_id = {item["candidate_id"]: item for item in plan_summaries}
    lines = [
        "# Atlas-routed composition candidates",
        "",
        "All evidence below comes from the local Gram atlas; model/delta weights were not reread.",
        (
            f"departure protection uses the top 90% routable departure-delta energy "
            f"(threshold `{threshold:.10g}`), excluding embed/head/norm tensors."
        ),
        "",
        "## All-model evidence screen",
        "",
        "| Model | norm/departure | departure cosine | local novelty | route-conflict energy | score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {label} | {norm_ratio_to_departure:.4f} | {departure_cosine:.4f} | "
            "{local_novelty_fraction:.4f} | {route_conflict_energy_fraction:.4f} | "
            "{evidence_score:.4f} |".format(**item)
        )
    lines.extend(["", "## Candidates", ""])
    for recipe in recipes:
        summary = plan_by_id[recipe["candidate_id"]]
        lines.extend(
            [
                f"### {recipe['candidate_id']}",
                "",
                recipe["candidate_rationale"],
                "",
                f"- Donor order: {', '.join(item['donor_label'] for item in recipe['donors'])}",
                (
                    f"- Atlas preplan: {summary['accepted']}/{summary['blocks']} blocks accepted; "
                    f"{summary['protected']} protected; {summary['excluded']} hard-excluded."
                ),
                "- Strengths and scopes:",
            ]
        )
        for donor in recipe["donors"]:
            evidence = donor["evidence"]
            lines.append(
                f"  - **{donor['donor_label']}**: alpha {donor['alpha']:.5f}; "
                f"components {donor['components']}; architectures "
                f"{donor['architectures']}; boost layers {donor['boost_layers']}; "
                f"novelty {evidence['local_novelty_fraction']:.3f}; "
                f"route-conflict {evidence['route_conflict_energy_fraction']:.3f}; "
                f"component affinities {evidence['component_affinities']}."
            )
        lines.append("")
    lines.extend(
        [
            "## Cached-plan limitation",
            "",
            "Each cached NPZ contains deterministic winner, second, mix, and nominal "
            "scale=1 arrays derived only from local Gram matrices. It is an audit "
            "preplan, not an exact composition-ready RMS plan: the atlas has "
            "departure-delta RMS but not full departure-weight RMS. A later "
            "composition finalizer must read departure weights, apply `max_rms_ratio`, "
            "and recompute the scale-coupled protected-component projection beta.",
            "",
        ]
    )
    path = output / "candidate_report.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def analyze_local_atlas(
    atlas_dir: str,
    analysis_dir: str,
    output_dir: str | None = None,
    departure_label: str | None = DEFAULT_DEPARTURE_LABEL,
    prior_recipe: str | None = None,
) -> dict:
    """Build derived artifacts, plots, candidate recipes, and atlas-only preplans."""
    if not departure_label:
        raise ValueError("departure_label is required")

    atlas_root = Path(atlas_dir)
    analysis_root = relocated_path(analysis_dir)
    output = Path(output_dir) if output_dir else atlas_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("LLMTRAITS_ATLAS_ANALYSIS_JOB_TOKEN", "")
    job_path = output / "analysis_job.json"
    if job_path.exists():
        prior_job = _read_json(job_path)
        prior_pid = int(prior_job.get("pid", 0))
        if (
            prior_pid
            and prior_pid != os.getpid()
            and str(prior_job.get("job_token", "")) != token
            and _pid_alive(prior_pid)
        ):
            raise RuntimeError(f"Atlas analysis is already running with PID {prior_pid}")
    _atomic_json(
        job_path,
        {
            "pid": os.getpid(),
            "job_token": token,
            "mode": "analysis",
            "atlas_dir": str(atlas_root.resolve()),
            "analysis_dir": str(analysis_root.resolve()),
            "output_dir": str(output.resolve()),
            "started_at": _now(),
        },
    )

    identity_hash = ""
    try:
        analysis = load_analysis(str(analysis_root))
        atlas_index, atlas_identity, departure_index = _validate_preflight(
            atlas_root, analysis, departure_label
        )
        labels = [str(value) for value in atlas_index["labels"]]
        identity = _analysis_identity(
            atlas_root,
            analysis_root,
            atlas_index,
            atlas_identity,
            departure_label,
            prior_recipe,
        )
        identity_hash = identity["identity_hash"]
        identity_path = output / "analysis_identity.json"
        if identity_path.exists():
            existing = _read_json(identity_path)
            if existing.get("identity_hash") != identity_hash:
                raise ValueError(
                    "Derived atlas identity mismatch; use a new output directory"
                )
        _atomic_json(identity_path, identity)
        analysis["_atlas_block_size"] = int(atlas_index["block_size"])
        tensor_items = sorted(atlas_index["tensors"].items())
        model_count = len(labels)
        component_count = int(analysis["n_components"])
        layer_values = sorted({_layer(name) for name, _ in tensor_items})
        architectures = sorted({_arch(name) for name, _ in tensor_items})
        layer_positions = {value: index for index, value in enumerate(layer_values)}
        arch_positions = {value: index for index, value in enumerate(architectures)}
        layer_arch_gram = np.zeros(
            (len(layer_values), len(architectures), model_count, model_count),
            dtype=np.float64,
        )
        layer_arch_blocks = np.zeros(
            (len(layer_values), len(architectures)), dtype=np.int64
        )
        layer_arch_parameters = np.zeros_like(layer_arch_blocks)
        total_novelty = np.zeros(model_count, dtype=np.float64)
        conflict_energy = np.zeros(model_count, dtype=np.float64)
        route_conflict_energy = np.zeros(model_count, dtype=np.float64)
        layer_novelty_energy = np.zeros(
            (len(layer_values), model_count), dtype=np.float64
        )
        layer_conflict_energy = np.zeros_like(layer_novelty_energy)
        routable_departure_energies: list[np.ndarray] = []
        tensor_rows = []
        derived_records = {}
        derived_dir = output / "derived_tensors"
        derived_dir.mkdir(parents=True, exist_ok=True)

        for tensor_number, (name, record) in enumerate(tensor_items, start=1):
            _write_progress(
                output,
                identity_hash,
                status="running",
                phase="derived",
                completed_tensors=tensor_number - 1,
                tensor_total=len(tensor_items),
                current_tensor=name,
            )
            source, source_sha = _load_atlas_tensor(
                atlas_root,
                identity_hash,
                atlas_index["identity_hash"],
                name,
                record,
                model_count,
            )
            gram = np.asarray(source["gram"], dtype=np.float64)
            derived_path = derived_dir / f"{_tensor_key(name)}.npz"
            metrics = _load_derived(
                derived_path,
                identity_hash,
                source_sha,
                name,
                int(record["blocks"]),
                model_count,
                component_count,
            )
            if metrics is None:
                metrics = derive_block_metrics(
                    gram,
                    departure_index,
                    np.asarray(analysis["left_vectors"], dtype=np.float64),
                    np.asarray(analysis["singular_values"], dtype=np.float64),
                )
                _atomic_npz(
                    derived_path,
                    schema_version=np.array(
                        ATLAS_ANALYSIS_SCHEMA_VERSION, dtype="U16"
                    ),
                    identity_hash=np.array(identity_hash, dtype="U64"),
                    atlas_identity_hash=np.array(
                        atlas_index["identity_hash"], dtype="U64"
                    ),
                    source_sha256=np.array(source_sha, dtype="U64"),
                    tensor_name=np.array(name, dtype="U2048"),
                    shape=np.asarray(source["shape"], dtype=np.int64),
                    starts=np.asarray(source["starts"], dtype=np.int64),
                    ends=np.asarray(source["ends"], dtype=np.int64),
                    layer=np.array(_layer(name), dtype=np.int32),
                    architecture=np.array(_arch(name), dtype="U64"),
                    **metrics,
                )
                metrics = _load_derived(
                    derived_path,
                    identity_hash,
                    source_sha,
                    name,
                    int(record["blocks"]),
                    model_count,
                    component_count,
                )
                assert metrics is not None
            layer_position = layer_positions[_layer(name)]
            arch_position = arch_positions[_arch(name)]
            tensor_gram = np.sum(gram, axis=0)
            layer_arch_gram[layer_position, arch_position] += tensor_gram
            layer_arch_blocks[layer_position, arch_position] += int(record["blocks"])
            layer_arch_parameters[layer_position, arch_position] += int(
                record["parameters"]
            )
            total_novelty += np.sum(metrics["novelty_energy_departure"], axis=0)
            conflict_energy += np.sum(
                metrics["energy"] * metrics["departure_conflict"], axis=0
            )
            route_conflict_energy += np.sum(
                metrics["energy"] * metrics["route_conflict"], axis=0
            )
            layer_novelty_energy[layer_position] += np.sum(
                metrics["novelty_energy_departure"], axis=0
            )
            layer_conflict_energy[layer_position] += np.sum(
                metrics["energy"] * metrics["departure_conflict"], axis=0
            )
            if not any(
                pattern in name.casefold() for pattern in HARD_EXCLUSIONS
            ):
                routable_departure_energies.append(
                    np.asarray(metrics["energy"][:, departure_index], dtype=np.float64)
                )
            row = {
                "tensor_name": name,
                "layer": _layer(name),
                "architecture": _arch(name),
                "blocks": int(record["blocks"]),
                "parameters": int(record["parameters"]),
            }
            tensor_energy = np.maximum(np.diag(tensor_gram), 0.0)
            tensor_cosine = _safe_cosine(tensor_gram[None, :, :])[0]
            tensor_novelty = np.sum(metrics["novelty_energy_departure"], axis=0)
            for model_index, label in enumerate(labels):
                prefix = f"m{model_index + 1}"
                row[f"{prefix}_energy"] = float(tensor_energy[model_index])
                row[f"{prefix}_departure_cosine"] = float(
                    tensor_cosine[departure_index, model_index]
                )
                row[f"{prefix}_novelty_fraction"] = (
                    float(tensor_novelty[model_index] / tensor_energy[model_index])
                    if tensor_energy[model_index] > 0
                    else 0.0
                )
            tensor_rows.append(row)
            derived_records[name] = {
                "file": str(derived_path.relative_to(output)),
                "sha256": sha256_file(str(derived_path)),
                "source_atlas_sha256": source_sha,
                "blocks": int(record["blocks"]),
                "layer": _layer(name),
                "architecture": _arch(name),
            }

        global_gram = np.sum(layer_arch_gram, axis=(0, 1))
        layer_gram = np.sum(layer_arch_gram, axis=1)
        arch_gram = np.sum(layer_arch_gram, axis=0)
        threshold = _protection_threshold(
            routable_departure_energies, DEFAULT_PROTECTION_FRACTION
        )
        summaries = _group_summary(
            labels,
            global_gram,
            total_novelty,
            conflict_energy,
            route_conflict_energy,
            departure_index,
        )
        summary_payload = {
            "schema_version": ATLAS_ANALYSIS_SCHEMA_VERSION,
            "identity_hash": identity_hash,
            "atlas_identity_hash": atlas_index["identity_hash"],
            "departure_label": departure_label,
            "labels": labels,
            "block_size": int(atlas_index["block_size"]),
            "tensor_count": int(atlas_index["tensor_count"]),
            "block_count": int(atlas_index["block_count"]),
            "protected_departure": {
                "energy_fraction": DEFAULT_PROTECTION_FRACTION,
                "routable_energy_threshold": (
                    threshold if math.isfinite(threshold) else None
                ),
                "hard_exclusions": list(HARD_EXCLUSIONS),
            },
            "models": summaries,
            "metric_definitions": {
                "local_novelty_energy": (
                    "sum_b max(G_ii - G_iD^2/G_DD, 0), with full model energy "
                    "used when local departure energy is zero"
                ),
                "svd_local_component_energy": (
                    "c_k^T G_block c_k where c_k = left_vectors[:,k]/singular_value[k]"
                ),
                "route_conflict": (
                    f"local departure cosine < {DEFAULT_CONFLICT_THRESHOLD}"
                ),
            },
        }
        _atomic_json(output / "analysis_summary.json", summary_payload)
        aggregation_path = output / "aggregates.npz"
        _atomic_npz(
            aggregation_path,
            schema_version=np.array(ATLAS_ANALYSIS_SCHEMA_VERSION, dtype="U16"),
            identity_hash=np.array(identity_hash, dtype="U64"),
            labels=np.asarray(labels, dtype="U256"),
            layers=np.asarray(layer_values, dtype=np.int32),
            architectures=np.asarray(architectures, dtype="U64"),
            global_gram=global_gram,
            layer_gram=layer_gram,
            architecture_gram=arch_gram,
            layer_architecture_gram=layer_arch_gram,
            layer_architecture_blocks=layer_arch_blocks,
            layer_architecture_parameters=layer_arch_parameters,
            layer_novelty_energy=layer_novelty_energy,
            layer_conflict_energy=layer_conflict_energy,
            total_novelty_energy=total_novelty,
            conflict_energy=conflict_energy,
            route_conflict_energy=route_conflict_energy,
            protection_threshold=np.array(threshold, dtype=np.float64),
        )
        tensor_fields = list(tensor_rows[0]) if tensor_rows else []
        _atomic_csv(output / "tensor_summary.csv", tensor_fields, tensor_rows)
        _atomic_json(
            output / "derived_index.json",
            {
                "schema_version": ATLAS_ANALYSIS_SCHEMA_VERSION,
                "identity_hash": identity_hash,
                "atlas_identity_hash": atlas_index["identity_hash"],
                "tensors": derived_records,
                "aggregates": str(aggregation_path.relative_to(output)),
                "aggregates_sha256": sha256_file(str(aggregation_path)),
            },
        )

        _write_progress(
            output,
            identity_hash,
            status="running",
            phase="inspection",
            completed_tensors=len(tensor_items),
            tensor_total=len(tensor_items),
        )
        territory_scores = np.zeros(
            (len(layer_values), len(architectures), model_count), dtype=np.float64
        )
        territory_counts = np.zeros_like(territory_scores, dtype=np.int64)
        island_heaps: list[list[tuple]] = [[] for _ in labels]
        conflict_heap: list[tuple] = []
        protected_heap: list[tuple] = []
        for name, record in tensor_items:
            derived_path = output / derived_records[name]["file"]
            with np.load(derived_path, allow_pickle=False) as data:
                starts = np.asarray(data["starts"], dtype=np.int64)
                ends = np.asarray(data["ends"], dtype=np.int64)
                energy = np.asarray(data["energy"], dtype=np.float64)
                novelty = np.asarray(data["novelty_energy_departure"], dtype=np.float64)
                novelty_fraction = np.asarray(
                    data["novelty_fraction_departure"], dtype=np.float64
                )
                departure_cosine = np.asarray(data["departure_cosine"], dtype=np.float64)
            layer_position = layer_positions[_layer(name)]
            arch_position = arch_positions[_arch(name)]
            protected = energy[:, departure_index] >= threshold
            if not math.isfinite(threshold):
                protected.fill(False)
            hard_excluded = any(
                pattern in name.casefold() for pattern in HARD_EXCLUSIONS
            )
            scores = np.sqrt(np.maximum(novelty, 0.0))
            scores[departure_cosine < DEFAULT_CONFLICT_THRESHOLD] = 0.0
            scores[:, departure_index] = 0.0
            if hard_excluded:
                scores.fill(0.0)
            scores[protected] = 0.0
            winners = np.argmax(scores, axis=1)
            winner_values = scores[np.arange(len(scores)), winners]
            winners[winner_values <= 0] = -1
            for model_index in range(model_count):
                selected = winners == model_index
                territory_counts[layer_position, arch_position, model_index] += int(
                    np.count_nonzero(selected)
                )
                territory_scores[layer_position, arch_position, model_index] += float(
                    np.sum(winner_values[selected])
                )
                if model_index == departure_index or not len(scores):
                    continue
                local = scores[:, model_index]
                take = min(10, len(local))
                positions = (
                    np.argpartition(local, -take)[-take:] if take else np.array([], int)
                )
                for block in positions:
                    value = float(local[block])
                    if value <= 0:
                        continue
                    item = (
                        value,
                        name,
                        int(block),
                        int(starts[block]),
                        int(ends[block]),
                        float(novelty_fraction[block, model_index]),
                        float(departure_cosine[block, model_index]),
                    )
                    heapq.heappush(island_heaps[model_index], item)
                    if len(island_heaps[model_index]) > 100:
                        heapq.heappop(island_heaps[model_index])
            if len(scores):
                conflict_scores = np.maximum(-departure_cosine, 0.0) * energy
                conflict_scores[:, departure_index] = 0.0
                flat = conflict_scores.reshape(-1)
                take = min(10, len(flat))
                for flat_index in (
                    np.argpartition(flat, -take)[-take:] if take else []
                ):
                    value = float(flat[flat_index])
                    if value <= 0:
                        continue
                    block, model_index = np.unravel_index(
                        int(flat_index), conflict_scores.shape
                    )
                    item = (
                        value,
                        name,
                        int(block),
                        int(model_index),
                        int(starts[block]),
                        int(ends[block]),
                        float(departure_cosine[block, model_index]),
                    )
                    heapq.heappush(conflict_heap, item)
                    if len(conflict_heap) > 200:
                        heapq.heappop(conflict_heap)
                protected_values = np.where(
                    protected, energy[:, departure_index], 0.0
                )
                take = min(10, len(protected_values))
                for block in (
                    np.argpartition(protected_values, -take)[-take:] if take else []
                ):
                    value = float(protected_values[block])
                    if value <= 0:
                        continue
                    item = (
                        value,
                        name,
                        int(block),
                        int(starts[block]),
                        int(ends[block]),
                    )
                    heapq.heappush(protected_heap, item)
                    if len(protected_heap) > 200:
                        heapq.heappop(protected_heap)

        territory_rows = []
        for layer_position, layer in enumerate(layer_values):
            for arch_position, architecture in enumerate(architectures):
                scores = territory_scores[layer_position, arch_position]
                winner_index = int(np.argmax(scores)) if np.max(scores) > 0 else -1
                territory_rows.append(
                    {
                        "layer": layer,
                        "architecture": architecture,
                        "winner": labels[winner_index] if winner_index >= 0 else "",
                        "winner_score": (
                            float(scores[winner_index]) if winner_index >= 0 else 0.0
                        ),
                        "assigned_blocks": (
                            int(
                                territory_counts[
                                    layer_position, arch_position, winner_index
                                ]
                            )
                            if winner_index >= 0
                            else 0
                        ),
                        "total_blocks": int(
                            layer_arch_blocks[layer_position, arch_position]
                        ),
                    }
                )
        _atomic_csv(
            output / "donor_territories.csv",
            [
                "layer",
                "architecture",
                "winner",
                "winner_score",
                "assigned_blocks",
                "total_blocks",
            ],
            territory_rows,
        )
        island_rows = []
        for model_index, heap in enumerate(island_heaps):
            for item in sorted(heap, reverse=True):
                value, name, block, start, end, novelty_fraction, cosine = item
                island_rows.append(
                    {
                        "donor": labels[model_index],
                        "tensor_name": name,
                        "block": block,
                        "start": start,
                        "end": end,
                        "novelty_norm": value,
                        "novelty_fraction": novelty_fraction,
                        "departure_cosine": cosine,
                    }
                )
        _atomic_csv(
            output / "donor_islands.csv",
            [
                "donor",
                "tensor_name",
                "block",
                "start",
                "end",
                "novelty_norm",
                "novelty_fraction",
                "departure_cosine",
            ],
            island_rows,
        )
        conflict_rows = [
            {
                "model": labels[item[3]],
                "tensor_name": item[1],
                "block": item[2],
                "start": item[4],
                "end": item[5],
                "conflict_energy_score": item[0],
                "departure_cosine": item[6],
            }
            for item in sorted(conflict_heap, reverse=True)
        ]
        _atomic_csv(
            output / "conflicts.csv",
            [
                "model",
                "tensor_name",
                "block",
                "start",
                "end",
                "conflict_energy_score",
                "departure_cosine",
            ],
            conflict_rows,
        )
        protected_rows = [
            {
                "tensor_name": item[1],
                "block": item[2],
                "start": item[3],
                "end": item[4],
                "departure_energy": item[0],
            }
            for item in sorted(protected_heap, reverse=True)
        ]
        _atomic_csv(
            output / "protected_departure_regions.csv",
            ["tensor_name", "block", "start", "end", "departure_energy"],
            protected_rows,
        )

        _write_progress(
            output,
            identity_hash,
            status="running",
            phase="candidates",
            completed_tensors=len(tensor_items),
            tensor_total=len(tensor_items),
        )
        recipes = _design_candidates(
            output,
            analysis,
            analysis_root,
            labels,
            departure_label,
            summaries,
            global_gram,
            arch_gram,
            architectures,
            layer_gram,
            layer_values,
            prior_recipe,
        )
        plan_summaries = []

        def plan_progress(candidate_id, completed, total, current):
            _write_progress(
                output,
                identity_hash,
                status="running",
                phase="plans",
                candidate=candidate_id,
                completed_tensors=completed,
                tensor_total=total,
                current_tensor=current,
            )

        for recipe in recipes:
            plan_summaries.append(
                _build_preplan(
                    output,
                    atlas_root,
                    atlas_index,
                    identity_hash,
                    recipe,
                    analysis,
                    threshold,
                    plan_progress,
                )
            )
        _write_candidate_report(
            output, summaries, recipes, plan_summaries, threshold
        )
        _write_progress(
            output,
            identity_hash,
            status="running",
            phase="plots",
            completed_tensors=len(tensor_items),
            tensor_total=len(tensor_items),
        )
        plot_paths = _make_plots(
            output,
            labels,
            layer_values,
            architectures,
            layer_gram,
            layer_conflict_energy,
            layer_novelty_energy,
            global_gram,
            analysis,
            territory_scores,
            territory_counts,
            departure_index,
        )
        result = {
            "schema_version": ATLAS_ANALYSIS_SCHEMA_VERSION,
            "status": "complete",
            "identity_hash": identity_hash,
            "atlas_identity_hash": atlas_index["identity_hash"],
            "output_dir": str(output.resolve()),
            "tensor_count": len(tensor_items),
            "block_count": int(atlas_index["block_count"]),
            "candidate_count": len(recipes),
            "candidates": [recipe["candidate_id"] for recipe in recipes],
            "plots": plot_paths,
            "protection_threshold": (
                threshold if math.isfinite(threshold) else None
            ),
            "completed_at": _now(),
        }
        _atomic_json(output / "analysis_index.json", result)
        _write_progress(
            output,
            identity_hash,
            status="complete",
            phase="complete",
            completed_tensors=len(tensor_items),
            tensor_total=len(tensor_items),
        )
        return result
    except Exception as error:
        _write_progress(
            output,
            identity_hash,
            status="failed",
            phase="failed",
            error=str(error),
        )
        raise


def launch_atlas_analysis(
    atlas_dir: str,
    analysis_dir: str,
    output_dir: str | None = None,
    departure_label: str | None = DEFAULT_DEPARTURE_LABEL,
    prior_recipe: str | None = None,
) -> tuple[int, str]:
    """Launch ``atlas-analyze`` as a detached worker."""
    if not departure_label:
        raise ValueError("departure_label is required")

    output = Path(output_dir) if output_dir else Path(atlas_dir) / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    job_path = output / "analysis_job.json"
    if job_path.exists():
        job = _read_json(job_path)
        pid = int(job.get("pid", 0))
        if pid and _pid_alive(pid):
            raise RuntimeError(f"Atlas analysis is already running with PID {pid}")
    progress_path = output / "analysis_progress.json"
    if progress_path.exists() and _read_json(progress_path).get("status") == "complete":
        raise RuntimeError("Atlas analysis is already complete")
    token = secrets.token_hex(16)
    log_path = output / "analysis_worker.log"
    argv = [
        sys.executable,
        "-m",
        "llmtraits",
        "atlas-analyze",
        "--atlas-dir",
        os.path.abspath(atlas_dir),
        "--analysis",
        os.path.abspath(analysis_dir),
        "--output-dir",
        os.path.abspath(str(output)),
        "--departure-label",
        departure_label,
    ]
    if prior_recipe:
        argv.extend(["--prior-recipe", os.path.abspath(prior_recipe)])
    _atomic_json(
        job_path,
        {
            "pid": 0,
            "job_token": token,
            "mode": "launching",
            "atlas_dir": os.path.abspath(atlas_dir),
            "analysis_dir": os.path.abspath(analysis_dir),
            "output_dir": os.path.abspath(str(output)),
            "launched_at": _now(),
        },
    )
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n=== Atlas analysis launch {_now()} ===\nargv: {argv}\n")
        log.flush()
        kwargs = (
            {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
            }
            if sys.platform == "win32"
            else {"start_new_session": True}
        )
        process = subprocess.Popen(
            argv,
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "LLMTRAITS_ATLAS_ANALYSIS_JOB_TOKEN": token,
            },
            **kwargs,
        )
    worker_pid = process.pid
    for _ in range(100):
        time.sleep(0.1)
        try:
            active = _read_json(job_path)
            if (
                active.get("job_token") == token
                and int(active.get("pid", 0)) > 0
                and active.get("mode") == "analysis"
            ):
                worker_pid = int(active["pid"])
                break
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    _atomic_json(
        output / "analysis_launch.json",
        {
            "pid": worker_pid,
            "launcher_pid": process.pid,
            "job_token": token,
            "atlas_dir": os.path.abspath(atlas_dir),
            "analysis_dir": os.path.abspath(analysis_dir),
            "output_dir": os.path.abspath(str(output)),
            "log_path": str(log_path),
            "launched_at": _now(),
        },
    )
    return worker_pid, str(log_path)
