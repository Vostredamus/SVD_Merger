"""
Block-level diagnostics for SVD analysis.

Computes per-block statistics for each departure model × component:
  - novel support (departure changed little, component changed strongly)
  - shared support (both changed, aligned)
  - sign conflicts
  - layer/architecture localization of novel energy

Key constants:
  BLOCK_SIZE = 1_048_576   (flat parameter count per block, tails included)
  NOVELTY_RATIO = 0.10     (dep_rms <= ratio * comp_rms at physical scale)
  MATERIAL_BLOCK_FRACTION = 0.001  (min comp_block_energy fraction of unit-norm total)
  MAX_TOP_BLOCKS = 50      (max novel blocks stored per departure×component)
  DIAG_SCHEMA_VERSION = "1.0"

A 'novel' block satisfies BOTH:
  1. dep_block_energy <= NOVELTY_RATIO^2 * comp_block_energy * sigma_k^2
     (departure RMS is <= NOVELTY_RATIO times component RMS at physical scale)
  2. comp_block_energy > MATERIAL_BLOCK_FRACTION
     (block is not numerical dust)

Note: SVD components are derived including all departure models. A high novel
support fraction means the component has amplitude where the departure delta is
flat, but the component itself still depends on all deltas including departure.
"""

import json
import os
import re
from pathlib import Path
from .paths import relocated_path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .tensor_stream import iter_flat_ranges, open_model_tensor_slices, read_flat_slice
from .utils import ensure_dir

BLOCK_SIZE = 1_048_576
NOVELTY_RATIO = 0.10
MATERIAL_BLOCK_FRACTION = 0.001
MAX_TOP_BLOCKS = 50
DIAG_SCHEMA_VERSION = "1.0"

_ARCH_COMPONENTS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "input_layernorm",
    "post_attention_layernorm",
    "embed_tokens",
    "norm",
    "lm_head",
]
_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
_EPS = 1e-12


def _arch_component(tensor_name: str) -> str:
    for component in _ARCH_COMPONENTS:
        if component in tensor_name:
            return component
    return "other"


def _layer_index(tensor_name: str) -> int | None:
    match = _LAYER_RE.search(tensor_name)
    if not match:
        return None
    return int(match.group(1))


def compute_block_diagnostics(
    shared_names: list,
    models: list,
    left_vectors: np.ndarray,
    singular_values: np.ndarray,
    layer_labels: list,
    arch_labels: list,
    layer_lookup: dict,
    arch_lookup: dict,
    block_size: int = BLOCK_SIZE,
    novelty_ratio: float = NOVELTY_RATIO,
    material_block_fraction: float = MATERIAL_BLOCK_FRACTION,
    max_top_blocks: int = MAX_TOP_BLOCKS,
    progress_callback=None,
) -> dict:
    """
    Compute block-level diagnostics by streaming through all shared tensors.

    Returns a dict with aggregated arrays and top-block records.
    """
    n_models = len(models)
    n_components = int(left_vectors.shape[1])
    n_layers = len(layer_labels)
    n_archs = len(arch_labels)
    global_layer_pos = n_layers - 1

    labels = [model["label"] for model in models]
    U = np.asarray(left_vectors, dtype=np.float64)  # (n_models, n_components)
    sigma = np.asarray(singular_values, dtype=np.float64)  # (n_components,)
    sigma_safe = np.where(sigma > 0, sigma, 1.0)

    novel_support = np.zeros((n_models, n_components), dtype=np.float64)
    shared_support = np.zeros((n_models, n_components), dtype=np.float64)
    sign_conflict_num = np.zeros((n_models, n_components), dtype=np.float64)
    material_total = np.zeros((n_models, n_components), dtype=np.float64)
    novel_layer_energy = np.zeros((n_models, n_components, n_layers), dtype=np.float64)
    novel_arch_energy = np.zeros((n_models, n_components, n_archs), dtype=np.float64)

    top_blocks = [[[] for _ in range(n_components)] for _ in range(n_models)]

    total_blocks = 0
    ratio_sq = float(novelty_ratio) ** 2

    for tensor_index, name in enumerate(shared_names, start=1):
        if progress_callback:
            progress_callback(tensor_index, len(shared_names), name)
        layer_idx = _layer_index(name)
        layer_pos = layer_lookup.get(layer_idx, global_layer_pos) if layer_idx is not None else global_layer_pos
        arch_pos = arch_lookup.get(_arch_component(name), n_archs - 1)

        with open_model_tensor_slices(models, name) as (handles, shape):
          for offset, end in iter_flat_ranges(shape, block_size):
            D_block32 = np.empty((n_models, end - offset), dtype=np.float32)
            for model_index, handle in enumerate(handles):
                D_block32[model_index] = read_flat_slice(
                    handle, name, shape, offset, end
                )
            D_block = D_block32.astype(np.float64)
            proj = U.T @ D_block  # (n_components, L)
            v_blocks = proj / sigma_safe[:, np.newaxis]  # (n_components, L)
            comp_e = np.einsum("kl,kl->k", v_blocks, v_blocks)  # (n_components,)
            comp_e[sigma <= 0] = 0.0
            dep_e = np.einsum("pl,pl->p", D_block, D_block)  # (n_models,)
            local_dot = v_blocks @ D_block.T  # (n_components, n_models)

            is_material = comp_e > material_block_fraction  # (n_components,)
            # novelty per (model, component): dep_e[p] <= ratio^2 * comp_e[k] * sigma_k^2
            thresh = ratio_sq * comp_e * (sigma ** 2)  # (n_components,)
            novel_mat = (dep_e[:, np.newaxis] <= thresh[np.newaxis, :]) & is_material[np.newaxis, :]

            comp_e_row = comp_e[np.newaxis, :]  # (1, n_components)
            material_row = is_material[np.newaxis, :]
            material_total += np.where(material_row, comp_e_row, 0.0)
            novel_support += np.where(novel_mat, comp_e_row, 0.0)
            shared_support += np.where(material_row & ~novel_mat, comp_e_row, 0.0)
            conflict_mat = material_row & (local_dot.T < 0.0)
            sign_conflict_num += np.where(conflict_mat, comp_e_row, 0.0)

            novel_layer_energy[:, :, layer_pos] += np.where(novel_mat, comp_e_row, 0.0)
            novel_arch_energy[:, :, arch_pos] += np.where(novel_mat, comp_e_row, 0.0)

            novel_pk = np.argwhere(novel_mat)
            for p, k in novel_pk:
                top_blocks[p][k].append(
                    {
                        "tensor_name": name,
                        "block_offset": int(offset),
                        "comp_block_energy": float(comp_e[k]),
                        "dep_block_energy": float(dep_e[p]),
                        "local_dot": float(local_dot[k, p]),
                    }
                )

            total_blocks += 1

        # Periodic truncation to keep memory bounded on large models.
        for p in range(n_models):
            for k in range(n_components):
                bucket = top_blocks[p][k]
                if len(bucket) > 4 * max_top_blocks:
                    bucket.sort(key=lambda item: item["comp_block_energy"], reverse=True)
                    del bucket[max_top_blocks:]

    novel_support_frac = novel_support.astype(np.float32)
    shared_support_frac = shared_support.astype(np.float32)
    sign_conflict_frac = np.zeros((n_models, n_components), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(material_total > 0, sign_conflict_num / material_total, 0.0)
    sign_conflict_frac = sc.astype(np.float32)

    top_records = []
    for p in range(n_models):
        for k in range(n_components):
            bucket = top_blocks[p][k]
            bucket.sort(key=lambda item: item["comp_block_energy"], reverse=True)
            for item in bucket[:max_top_blocks]:
                top_records.append(
                    {
                        "departure_label": labels[p],
                        "model_index": int(p),
                        "component_k": int(k + 1),
                        **item,
                    }
                )

    return {
        "schema_version": DIAG_SCHEMA_VERSION,
        "labels": labels,
        "n_models": n_models,
        "n_components": n_components,
        "layer_labels": list(layer_labels),
        "arch_labels": list(arch_labels),
        "block_size": int(block_size),
        "novelty_ratio": float(novelty_ratio),
        "material_block_fraction": float(material_block_fraction),
        "max_top_blocks": int(max_top_blocks),
        "total_blocks": int(total_blocks),
        "novel_support_frac": novel_support_frac,
        "shared_support_frac": shared_support_frac,
        "sign_conflict_num": sign_conflict_num.astype(np.float64),
        "sign_conflict_frac": sign_conflict_frac,
        "material_total": material_total.astype(np.float32),
        "novel_layer_energy": novel_layer_energy.astype(np.float32),
        "novel_arch_energy": novel_arch_energy.astype(np.float32),
        "top_blocks": top_records,
    }


def merge_block_diagnostics(diagnostics: list[dict], template: dict) -> dict:
    """Merge independently checkpointed per-tensor diagnostic accumulators."""
    if not diagnostics:
        raise ValueError("No block diagnostics to merge")
    result = dict(template)
    array_keys = [
        "novel_support_frac",
        "shared_support_frac",
        "sign_conflict_num",
        "material_total",
        "novel_layer_energy",
        "novel_arch_energy",
    ]
    for key in array_keys:
        result[key] = np.sum(
            [np.asarray(item[key], dtype=np.float64) for item in diagnostics], axis=0
        )
    result["total_blocks"] = sum(int(item["total_blocks"]) for item in diagnostics)
    material = result["material_total"]
    conflict_fraction = np.zeros_like(material, dtype=np.float64)
    np.divide(
        result["sign_conflict_num"],
        material,
        out=conflict_fraction,
        where=material > 0,
    )
    result["sign_conflict_frac"] = conflict_fraction.astype(np.float32)
    result["novel_support_frac"] = result["novel_support_frac"].astype(np.float32)
    result["shared_support_frac"] = result["shared_support_frac"].astype(np.float32)
    result["material_total"] = material.astype(np.float32)
    result["novel_layer_energy"] = result["novel_layer_energy"].astype(np.float32)
    result["novel_arch_energy"] = result["novel_arch_energy"].astype(np.float32)

    records = [record for item in diagnostics for record in item.get("top_blocks", [])]
    bounded = []
    for model_index in range(result["n_models"]):
        for component in range(1, result["n_components"] + 1):
            bucket = [
                record
                for record in records
                if record["model_index"] == model_index
                and record["component_k"] == component
            ]
            bucket.sort(
                key=lambda record: (
                    -record["comp_block_energy"],
                    record["tensor_name"],
                    record["block_offset"],
                )
            )
            bounded.extend(bucket[: result["max_top_blocks"]])
    result["top_blocks"] = bounded
    return result


def _diag_npz_path(analysis_dir: str) -> str:
    return os.path.join(analysis_dir, "svd_block_diagnostics.npz")


def _diag_json_path(analysis_dir: str) -> str:
    return os.path.join(analysis_dir, "svd_block_diagnostics.json")


def save_block_diagnostics(output_dir: str, diag: dict) -> None:
    """Save diagnostics to svd_block_diagnostics.npz and svd_block_diagnostics.json."""
    ensure_dir(output_dir)
    npz_path = Path(_diag_npz_path(output_dir))
    npz_tmp = npz_path.with_suffix(npz_path.suffix + ".tmp")
    with open(npz_tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.array(diag["schema_version"], dtype="U32"),
            labels=np.array(diag["labels"], dtype="U512"),
            n_models=np.array(diag["n_models"], dtype=np.int64),
            n_components=np.array(diag["n_components"], dtype=np.int64),
            layer_labels=np.array(diag["layer_labels"], dtype="U64"),
            arch_labels=np.array(diag["arch_labels"], dtype="U64"),
            block_size=np.array(diag["block_size"], dtype=np.int64),
            novelty_ratio=np.array(diag["novelty_ratio"], dtype=np.float64),
            material_block_fraction=np.array(diag["material_block_fraction"], dtype=np.float64),
            max_top_blocks=np.array(diag["max_top_blocks"], dtype=np.int64),
            total_blocks=np.array(diag["total_blocks"], dtype=np.int64),
            novel_support_frac=diag["novel_support_frac"],
            shared_support_frac=diag["shared_support_frac"],
            sign_conflict_frac=diag["sign_conflict_frac"],
            material_total=diag["material_total"],
            novel_layer_energy=diag["novel_layer_energy"],
            novel_arch_energy=diag["novel_arch_energy"],
        )
    os.replace(npz_tmp, npz_path)
    payload = {
        "schema_version": diag["schema_version"],
        "labels": diag["labels"],
        "n_models": diag["n_models"],
        "n_components": diag["n_components"],
        "layer_labels": diag["layer_labels"],
        "arch_labels": diag["arch_labels"],
        "block_size": diag["block_size"],
        "novelty_ratio": diag["novelty_ratio"],
        "material_block_fraction": diag["material_block_fraction"],
        "max_top_blocks": diag["max_top_blocks"],
        "total_blocks": diag["total_blocks"],
        "top_blocks": diag["top_blocks"],
    }
    path = Path(_diag_json_path(output_dir))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(str(tmp), str(path))


def load_block_diagnostics(analysis_dir: str) -> dict | None:
    """Load diagnostics, returns None if not found (older artifact)."""
    npz_path = relocated_path(_diag_npz_path(analysis_dir))
    if not npz_path.exists():
        return None
    json_payload = {}
    json_path = relocated_path(_diag_json_path(analysis_dir))
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as handle:
            json_payload = json.load(handle)
    with np.load(npz_path, allow_pickle=False) as data:
        diag = {
            "schema_version": str(data["schema_version"].item()),
            "labels": [str(item) for item in data["labels"].tolist()],
            "n_models": int(data["n_models"].item()),
            "n_components": int(data["n_components"].item()),
            "layer_labels": [str(item) for item in data["layer_labels"].tolist()],
            "arch_labels": [str(item) for item in data["arch_labels"].tolist()],
            "block_size": int(data["block_size"].item()),
            "novelty_ratio": float(data["novelty_ratio"].item()),
            "material_block_fraction": float(data["material_block_fraction"].item()),
            "max_top_blocks": int(data["max_top_blocks"].item()),
            "total_blocks": int(data["total_blocks"].item()),
            "novel_support_frac": data["novel_support_frac"],
            "shared_support_frac": data["shared_support_frac"],
            "sign_conflict_frac": data["sign_conflict_frac"],
            "material_total": data["material_total"],
            "novel_layer_energy": data["novel_layer_energy"],
            "novel_arch_energy": data["novel_arch_energy"],
        }
    diag["top_blocks"] = json_payload.get("top_blocks", [])
    return diag


def departure_overlap_table(analysis: dict, diag: dict, departure_label: str) -> list:
    """Build the per-component departure-overlap table rows."""
    labels = analysis["labels"]
    p = labels.index(departure_label)
    coeffs = np.asarray(analysis["model_coefficients"], dtype=np.float64)[p]
    norms_sq = np.asarray(analysis["norms_sq"], dtype=np.float64)
    norm_sq_p = float(norms_sq[p])
    norm_p = float(np.sqrt(max(norm_sq_p, 0.0)))
    explained = np.asarray(analysis["explained_energy_ratio"], dtype=np.float64)
    n_components = int(analysis["n_components"])

    layer_labels = diag["layer_labels"] if diag else []
    arch_labels = diag["arch_labels"] if diag else []

    rows = []
    for k in range(n_components):
        a_pk = float(coeffs[k])
        signed_alignment = a_pk / max(norm_p, _EPS)
        departure_energy_share = (a_pk * a_pk) / max(norm_sq_p, _EPS)
        row = {
            "component": k + 1,
            "coefficient": a_pk,
            "signed_alignment": signed_alignment,
            "departure_energy_share": departure_energy_share,
            "explained_ensemble_energy": float(explained[k]) if k < len(explained) else 0.0,
        }
        if diag is not None:
            row["shared_support_fraction"] = float(diag["shared_support_frac"][p, k])
            row["novel_support_fraction"] = float(diag["novel_support_frac"][p, k])
            row["sign_conflict_fraction"] = float(diag["sign_conflict_frac"][p, k])
            layer_energy = np.asarray(diag["novel_layer_energy"][p, k], dtype=np.float64)
            arch_energy = np.asarray(diag["novel_arch_energy"][p, k], dtype=np.float64)
            top_layers = np.argsort(layer_energy)[::-1][:3]
            top_archs = np.argsort(arch_energy)[::-1][:2]
            row["dominant_layers"] = ", ".join(
                layer_labels[i] for i in top_layers if layer_energy[i] > 0
            )
            row["dominant_arch"] = ", ".join(
                arch_labels[i] for i in top_archs if arch_energy[i] > 0
            )
        else:
            row["shared_support_fraction"] = None
            row["novel_support_fraction"] = None
            row["sign_conflict_fraction"] = None
            row["dominant_layers"] = ""
            row["dominant_arch"] = ""
        rows.append(row)
    return rows


def _save_fig(fig, output_path: str) -> None:
    dpi = 150
    max_inches = 3000.0 / dpi
    w, h = fig.get_size_inches()
    fig.set_size_inches(min(w, max_inches), min(h, max_inches))
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_departure_energy(analysis: dict, diag: dict, departure_label: str, output_path: str) -> None:
    """Bar chart: departure_energy_share per component."""
    rows = departure_overlap_table(analysis, diag, departure_label)
    xs = np.arange(1, len(rows) + 1)
    values = [row["departure_energy_share"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(xs, values, color="#4472c4")
    ax.set_xlabel("Component")
    ax.set_ylabel("Departure energy share")
    ax.set_xticks(xs)
    ax.set_title(f"Departure energy share: {departure_label}")
    _save_fig(fig, output_path)


def plot_novel_support(analysis: dict, diag: dict, output_path: str) -> None:
    """Heatmap: models × components, color = novel_support_frac."""
    matrix = np.asarray(diag["novel_support_frac"], dtype=np.float64)
    labels = diag["labels"]
    n_components = matrix.shape[1]
    width = max(5.5, min(16.0, 1.0 + 0.55 * n_components))
    height = max(4.0, min(16.0, 1.2 + 0.45 * len(labels)))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n_components))
    ax.set_xticklabels([f"svd_{i + 1}" for i in range(n_components)], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Novel support fraction (models × components)")
    fig.colorbar(image, ax=ax)
    _save_fig(fig, output_path)


def overlap_csv_path(output_dir: str) -> str:
    return os.path.join(output_dir, "component_departure_overlap.csv")


def save_overlap_csv(analysis: dict, diag: dict, output_dir: str) -> None:
    """Save long-form CSV: model, component, all overlap metrics."""
    import csv

    ensure_dir(output_dir)
    fieldnames = [
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
    ]
    path = overlap_csv_path(output_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label in analysis["labels"]:
            for row in departure_overlap_table(analysis, diag, label):
                writer.writerow(
                    {
                        "model": label,
                        "component": row["component"],
                        "coefficient": row["coefficient"],
                        "signed_alignment": row["signed_alignment"],
                        "departure_energy_share": row["departure_energy_share"],
                        "explained_ensemble_energy": row["explained_ensemble_energy"],
                        "shared_support_fraction": row.get("shared_support_fraction"),
                        "novel_support_fraction": row.get("novel_support_fraction"),
                        "sign_conflict_fraction": row.get("sign_conflict_fraction"),
                        "dominant_layers": row.get("dominant_layers", ""),
                        "dominant_arch": row.get("dominant_arch", ""),
                    }
                )
    os.replace(tmp, path)
