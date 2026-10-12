import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

from .manifest import Manifest, load_manifest, save_manifest
from .paths import relocated_path
from .utils import ensure_dir


def compute_block_rms(tensor_2d: np.ndarray, block_size: int = 128) -> np.ndarray:
    h, w = tensor_2d.shape
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    padded = np.pad(tensor_2d, ((0, pad_h), (0, pad_w)), mode="constant")
    out_h = padded.shape[0] // block_size
    out_w = padded.shape[1] // block_size
    blocks = padded.reshape(out_h, block_size, out_w, block_size).transpose(0, 2, 1, 3)
    return np.sqrt(np.mean(np.square(blocks), axis=(2, 3)))


def _load_stats_lookup(delta_dir: str, manifest: Manifest) -> dict[str, dict]:
    if not manifest.stats_json:
        return {}
    path = relocated_path(delta_dir) / manifest.stats_json
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {item["name"]: item for item in payload}


def compute_landscapes(delta_dir: str, manifest: Manifest) -> str:
    output_path = relocated_path(delta_dir) / "landscapes.npz"
    stats_lookup = _load_stats_lookup(delta_dir, manifest)
    arrays = {}
    grouped: dict[str, list[str]] = {}
    for name, record in manifest.tensors.items():
        grouped.setdefault(record.shard_file, []).append(name)
    for shard_file, names in grouped.items():
        shard_path = relocated_path(delta_dir) / shard_file
        if not shard_path.exists():
            continue
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_tensor(name).numpy()
                if tensor.ndim != 2:
                    continue
                grid_h = math.ceil(tensor.shape[0] / 128)
                grid_w = math.ceil(tensor.shape[1] / 128)
                if grid_h > 128 or grid_w > 128:
                    continue
                base_rms = max(float(stats_lookup.get(name, {}).get("base_rms", 0.0)), 1e-12)
                arrays[name + "_landscape"] = compute_block_rms(tensor, block_size=128) / base_rms
    np.savez_compressed(output_path, **arrays)
    return str(output_path)


def render_overview(delta_dir: str, manifest: Manifest, output_path: str | None = None) -> str:
    output_path = output_path or os.path.join(delta_dir, "overview.png")
    stats_lookup = _load_stats_lookup(delta_dir, manifest)
    components = [
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
    layer_ids = sorted(
        {
            int(name.split(".")[2])
            for name in stats_lookup
            if name.startswith("model.layers.")
        }
    )
    row_labels = ["globals"] + [f"layer_{idx}" for idx in layer_ids]
    data = np.full((len(row_labels), len(components)), np.nan, dtype=np.float32)
    comp_index = {name: idx for idx, name in enumerate(components)}

    for name, stat in stats_lookup.items():
        if name == "model.embed_tokens.weight":
            data[0, comp_index["embed_tokens"]] = stat["rel_l2"]
        elif name == "model.norm.weight":
            data[0, comp_index["norm"]] = stat["rel_l2"]
        elif name == "lm_head.weight":
            data[0, comp_index["lm_head"]] = stat["rel_l2"]
        elif name.startswith("model.layers."):
            parts = name.split(".")
            layer = int(parts[2])
            row = 1 + layer_ids.index(layer)
            for component in components[:9]:
                if component in name:
                    data[row, comp_index[component]] = stat["rel_l2"]
                    break

    cmap = plt.get_cmap("inferno").with_extremes(bad="#888888")
    valid = data[np.isfinite(data)]
    vmin = float(np.percentile(valid, 2)) if valid.size else 0.0
    vmax = float(np.percentile(valid, 98)) if valid.size else 1.0
    fig, ax = plt.subplots(figsize=(12, max(4, len(row_labels) * 0.35)))
    image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=max(vmax, vmin + 1e-6))
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Relative delta magnitude overview")
    fig.colorbar(image, ax=ax, shrink=0.8, label="rel_l2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_layer_mosaics(delta_dir: str, manifest: Manifest, output_dir: str | None = None) -> list[str]:
    output_dir = output_dir or os.path.join(delta_dir, "mosaics")
    ensure_dir(output_dir)
    landscapes_path = relocated_path(delta_dir) / (
        manifest.landscapes_npz or "landscapes.npz"
    )
    if not landscapes_path.exists():
        return []
    npz = np.load(landscapes_path)
    component_order = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    layer_to_components: dict[int, dict[str, np.ndarray]] = {}
    for key in npz.files:
        if not key.startswith("model.layers."):
            continue
        parts = key.split(".")
        layer = int(parts[2])
        for component in component_order:
            if component in key:
                layer_to_components.setdefault(layer, {})[component] = npz[key]
                break
    written = []
    for layer, comp_map in sorted(layer_to_components.items()):
        if not comp_map:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(12, 8))
        axes = axes.flatten()
        for ax in axes:
            ax.axis("off")
        for idx, component in enumerate(component_order):
            if component not in comp_map:
                continue
            ax = axes[idx]
            ax.axis("on")
            ax.imshow(comp_map[component], aspect="auto", cmap="inferno")
            ax.set_title(component)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Layer {layer:03d} delta mosaics")
        fig.tight_layout()
        path = os.path.join(output_dir, f"layer_{layer:03d}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def visualize_command(args) -> None:
    manifest = load_manifest(args.delta_dir)
    if manifest is None:
        raise FileNotFoundError(f"No manifest found in {args.delta_dir}")
    if not args.skip_landscapes and (
        args.force
        or not manifest.landscapes_npz
        or not (relocated_path(args.delta_dir) / manifest.landscapes_npz).exists()
    ):
        manifest.landscapes_npz = os.path.basename(compute_landscapes(args.delta_dir, manifest))
        save_manifest(args.delta_dir, manifest)
    overview = render_overview(args.delta_dir, manifest)
    mosaics = render_layer_mosaics(args.delta_dir, manifest)
    print(f"Overview: {overview}")
    print(f"Mosaics: {len(mosaics)}")
