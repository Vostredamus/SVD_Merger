import csv
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .manifest import load_manifest
from .svd_checkpoint import SVDCheckpointStore
from .tensor_stream import DEFAULT_CHUNK_SIZE, streamed_tensor_gram
from .utils import ensure_dir

SVD_SCHEMA_VERSION = "1.0"
TOP_K_MAX = 10
SUPPORT_THRESHOLD = 0.001
LOGGER = logging.getLogger(__name__)

ARCH_COMPONENTS = [
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

_ARCH_LABELS = ARCH_COMPONENTS + ["other"]
_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def _arch_component(tensor_name: str) -> str:
    for component in ARCH_COMPONENTS:
        if component in tensor_name:
            return component
    return "other"


def _layer_index(tensor_name: str) -> int | None:
    match = _LAYER_RE.search(tensor_name)
    if not match:
        return None
    return int(match.group(1))


class SVDAnalyzer:
    def __init__(self, top_k=None):
        self.top_k = top_k
        self.models = []
        self.results: dict | None = None
        self.metadata: dict = {}
        self.block_diag: dict | None = None

    def add_model(self, delta_dir, label=None):
        manifest = load_manifest(delta_dir)
        if manifest is None:
            raise FileNotFoundError(f"No manifest found in {delta_dir}")
        self.models.append(
            {
                "delta_dir": os.path.abspath(delta_dir),
                "label": label or Path(delta_dir).name,
                "manifest": manifest,
            }
        )

    def _find_shared_tensors(self) -> tuple[set, set]:
        if not self.models:
            raise ValueError("No models added")
        name_union = set()
        shared = None
        shape_map = {}
        for model in self.models:
            model_shapes = {name: tuple(record.shape) for name, record in model["manifest"].tensors.items()}
            name_union.update(model_shapes)
            shape_map[model["delta_dir"]] = model_shapes
            names = set(model_shapes)
            shared = names if shared is None else shared & names
        shared = set(shared or set())
        for name in list(shared):
            shapes = {shape_map[model["delta_dir"]][name] for model in self.models}
            if len(shapes) != 1:
                shared.remove(name)
        excluded = name_union - shared
        return shared, excluded

    def _write_progress(self, progress_dir, **payload):
        if not progress_dir:
            return
        ensure_dir(str(progress_dir))
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        path = Path(progress_dir) / "svd_progress.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)

    def compute(self, progress_dir=None) -> dict:
        if not self.models:
            raise ValueError("No models added")
        from .support import (
            BLOCK_SIZE,
            MATERIAL_BLOCK_FRACTION,
            MAX_TOP_BLOCKS,
            NOVELTY_RATIO,
        )

        n_models = len(self.models)
        top_k = min(TOP_K_MAX, n_models) if self.top_k is None else max(1, min(int(self.top_k), n_models))
        shared_tensors, excluded_tensors = self._find_shared_tensors()
        if not shared_tensors:
            raise ValueError("No shared tensors across all models")
        shared_names = sorted(shared_tensors)
        excluded_names = sorted(excluded_tensors)
        checkpoint_store = SVDCheckpointStore(
            progress_dir,
            {
                "labels": [model["label"] for model in self.models],
                "base_fingerprints": [
                    model["manifest"].base_fingerprint for model in self.models
                ],
                "target_fingerprints": [
                    model["manifest"].target_fingerprint for model in self.models
                ],
                "shared_names": shared_names,
                "top_k": top_k,
                "chunk_size": DEFAULT_CHUNK_SIZE * 8,
                "block_size": BLOCK_SIZE,
                "novelty_ratio": NOVELTY_RATIO,
                "material_block_fraction": MATERIAL_BLOCK_FRACTION,
                "max_top_blocks": MAX_TOP_BLOCKS,
            },
        )
        G = np.zeros((n_models, n_models), dtype=np.float64)
        norms_sq = np.zeros(n_models, dtype=np.float64)
        tensor_energy = [dict() for _ in range(n_models)]
        sign_conflicts = []
        total_shared_params = 0
        gram_contributions = []
        for tensor_index, name in enumerate(shared_names, start=1):
            checkpoint = checkpoint_store.load_gram(tensor_index, name)
            resumed = checkpoint is not None
            self._write_progress(
                progress_dir,
                phase="gram",
                status="loading_checkpoint" if resumed else "processing",
                tensor_index=tensor_index,
                completed_tensors=tensor_index - 1,
                tensor_total=len(shared_names),
                tensor_name=name,
                resumed=resumed,
            )
            if resumed:
                G_t, params = checkpoint
                LOGGER.info(
                    "SVD Gram pass: resumed tensor %d/%d %s",
                    tensor_index,
                    len(shared_names),
                    name,
                )
            else:
                LOGGER.info(
                    "SVD Gram pass: tensor %d/%d %s",
                    tensor_index,
                    len(shared_names),
                    name,
                )
                G_t = streamed_tensor_gram(
                    self.models, name, DEFAULT_CHUNK_SIZE * 8
                )
                params = int(
                    math.prod(self.models[0]["manifest"].tensors[name].shape)
                )
                checkpoint_store.save_gram(tensor_index, name, G_t, params)
            gram_contributions.append(G_t)
            total_shared_params += params
            for i in range(n_models):
                energy = float(G_t[i, i])
                norms_sq[i] += energy
                tensor_energy[i][name] = energy
            G += G_t
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    dot = float(G_t[i, j])
                    if dot < 0:
                        sign_conflicts.append(
                            {
                                "pair": [self.models[i]["label"], self.models[j]["label"]],
                                "tensor": name,
                                "dot": dot,
                            }
                        )
            self._write_progress(
                progress_dir,
                phase="gram",
                tensor_index=tensor_index,
                completed_tensors=tensor_index,
                tensor_total=len(shared_names),
                tensor_name=name,
                resumed=resumed,
                status="checkpointed",
            )

        cosine = np.zeros_like(G)
        for i in range(n_models):
            for j in range(n_models):
                denom = math.sqrt(max(norms_sq[i], 1e-12) * max(norms_sq[j], 1e-12))
                cosine[i, j] = G[i, j] / denom if denom else 0.0

        active_sets = []
        for i in range(n_models):
            if norms_sq[i] <= 0:
                active_sets.append(set())
                continue
            active_sets.append(
                {
                    name
                    for name, energy in tensor_energy[i].items()
                    if energy / norms_sq[i] >= SUPPORT_THRESHOLD
                }
            )

        support_jaccard = np.zeros_like(G)
        shared_energy_frac = np.zeros_like(G)
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    support_jaccard[i, j] = 1.0
                    shared_energy_frac[i, j] = 1.0 if norms_sq[i] > 0 else 0.0
                    continue
                union = active_sets[i] | active_sets[j]
                inter = active_sets[i] & active_sets[j]
                support_jaccard[i, j] = len(inter) / len(union) if union else 1.0
                shared_energy = sum(min(tensor_energy[i][name], tensor_energy[j][name]) for name in inter)
                shared_energy_frac[i, j] = shared_energy / norms_sq[i] if norms_sq[i] > 0 else 0.0

        unique_residual = np.zeros(n_models, dtype=np.float64)
        for i in range(n_models):
            if norms_sq[i] <= 0:
                unique_residual[i] = 1.0
                continue
            others = [j for j in range(n_models) if j != i]
            if not others:
                unique_residual[i] = 1.0
                continue
            G_others = G[np.ix_(others, others)]
            g_i = G[np.ix_(others, [i])].reshape(-1)
            proj_sq = float(g_i @ np.linalg.pinv(G_others, rcond=1e-8) @ g_i)
            proj_sq = float(np.clip(proj_sq, 0.0, norms_sq[i]))
            unique_residual[i] = 1.0 - proj_sq / norms_sq[i]

        eigvals, eigvecs = np.linalg.eigh(G)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        for k in range(eigvecs.shape[1]):
            pivot = int(np.argmax(np.abs(eigvecs[:, k])))
            if eigvecs[pivot, k] < 0:
                eigvecs[:, k] *= -1.0
        clipped_eigvals = np.clip(eigvals, 0.0, None)
        singular_all = np.sqrt(clipped_eigvals)
        total_energy = float(np.sum(clipped_eigvals))
        explained_all = clipped_eigvals / total_energy if total_energy > 0 else np.zeros_like(clipped_eigvals)

        left_vectors = eigvecs[:, :top_k]
        singular_values = singular_all[:top_k]
        model_coefficients = left_vectors * singular_values[np.newaxis, :]
        explained_top = explained_all[:top_k]

        layer_indices = sorted({idx for name in shared_names for idx in [_layer_index(name)] if idx is not None})
        layer_labels = [f"layer_{idx}" for idx in layer_indices] + ["global"]
        layer_lookup = {idx: pos for pos, idx in enumerate(layer_indices)}
        global_index = len(layer_labels) - 1
        arch_labels = list(_ARCH_LABELS)
        arch_lookup = {name: pos for pos, name in enumerate(arch_labels)}
        component_layer = np.zeros((top_k, len(layer_labels)), dtype=np.float64)
        component_arch = np.zeros((top_k, len(arch_labels)), dtype=np.float64)

        self._write_progress(
            progress_dir,
            phase="localization",
            tensor_index=0,
            completed_tensors=0,
            tensor_total=len(shared_names),
            tensor_name="",
        )
        for tensor_index, (name, G_t) in enumerate(
            zip(shared_names, gram_contributions), start=1
        ):
            layer_idx = _layer_index(name)
            layer_pos = layer_lookup[layer_idx] if layer_idx is not None else global_index
            arch_pos = arch_lookup[_arch_component(name)]
            for k in range(top_k):
                sigma = singular_values[k]
                if sigma <= 0:
                    continue
                u_k = left_vectors[:, k]
                energy = float((u_k @ G_t @ u_k) / (sigma * sigma))
                if energy < 0 and abs(energy) < 1e-10:
                    energy = 0.0
                component_layer[k, layer_pos] += max(energy, 0.0)
                component_arch[k, arch_pos] += max(energy, 0.0)
        self._write_progress(
            progress_dir,
            phase="localization",
            tensor_index=len(shared_names),
            completed_tensors=len(shared_names),
            tensor_total=len(shared_names),
            tensor_name=shared_names[-1],
        )

        component_layer_fracs = np.zeros_like(component_layer)
        component_arch_fracs = np.zeros_like(component_arch)
        for k in range(top_k):
            layer_sum = float(np.sum(component_layer[k]))
            arch_sum = float(np.sum(component_arch[k]))
            if layer_sum > 0:
                component_layer_fracs[k] = component_layer[k] / layer_sum
            if arch_sum > 0:
                component_arch_fracs[k] = component_arch[k] / arch_sum

        from .support import (
            DIAG_SCHEMA_VERSION,
            compute_block_diagnostics,
            merge_block_diagnostics,
        )

        diagnostic_parts = []
        for tensor_index, name in enumerate(shared_names, start=1):
            diagnostic = checkpoint_store.load_diagnostic(tensor_index, name)
            resumed = diagnostic is not None
            self._write_progress(
                progress_dir,
                phase="block_diagnostics",
                status="loading_checkpoint" if resumed else "processing",
                tensor_index=tensor_index,
                completed_tensors=tensor_index - 1,
                tensor_total=len(shared_names),
                tensor_name=name,
                resumed=resumed,
            )
            if diagnostic is None:
                LOGGER.info(
                    "SVD block diagnostics: tensor %d/%d %s",
                    tensor_index,
                    len(shared_names),
                    name,
                )
                diagnostic = compute_block_diagnostics(
                    shared_names=[name],
                    models=self.models,
                    left_vectors=left_vectors,
                    singular_values=singular_values,
                    layer_labels=layer_labels,
                    arch_labels=arch_labels,
                    layer_lookup=layer_lookup,
                    arch_lookup=arch_lookup,
                )
                checkpoint_store.save_diagnostic(
                    tensor_index, name, diagnostic
                )
            else:
                LOGGER.info(
                    "SVD block diagnostics: resumed tensor %d/%d %s",
                    tensor_index,
                    len(shared_names),
                    name,
                )
            diagnostic_parts.append(diagnostic)
            self._write_progress(
                progress_dir,
                phase="block_diagnostics",
                tensor_index=tensor_index,
                completed_tensors=tensor_index,
                tensor_total=len(shared_names),
                tensor_name=name,
                resumed=resumed,
                status="checkpointed",
            )

        diagnostic_template = {
            "schema_version": DIAG_SCHEMA_VERSION,
            "labels": [model["label"] for model in self.models],
            "n_models": n_models,
            "n_components": top_k,
            "layer_labels": list(layer_labels),
            "arch_labels": list(arch_labels),
            "block_size": BLOCK_SIZE,
            "novelty_ratio": NOVELTY_RATIO,
            "material_block_fraction": MATERIAL_BLOCK_FRACTION,
            "max_top_blocks": MAX_TOP_BLOCKS,
        }
        self.block_diag = merge_block_diagnostics(
            diagnostic_parts, diagnostic_template
        )

        self.results = {
            "schema_version": SVD_SCHEMA_VERSION,
            "labels": [model["label"] for model in self.models],
            "delta_dirs": [model["delta_dir"] for model in self.models],
            "n_models": n_models,
            "n_components": top_k,
            "shared_tensor_count": len(shared_names),
            "excluded_tensor_count": len(excluded_names),
            "excluded_tensors": excluded_names,
            "total_shared_params": total_shared_params,
            "support_threshold": SUPPORT_THRESHOLD,
            "singular_values": singular_values,
            "left_vectors": left_vectors,
            "model_coefficients": model_coefficients,
            "all_singular_values": singular_all,
            "all_eigenvalues": clipped_eigvals,
            "explained_energy_ratio": explained_top,
            "all_explained_energy_ratio": explained_all,
            "gram": G,
            "norms_sq": norms_sq,
            "cosine_similarity": cosine,
            "support_jaccard": support_jaccard,
            "shared_energy_frac": shared_energy_frac,
            "unique_residual": unique_residual,
            "sign_conflict_count": len(sign_conflicts),
            "sign_conflict_energy": float(sum(abs(item["dot"]) for item in sign_conflicts)),
            "sign_conflicts_top": sorted(sign_conflicts, key=lambda item: abs(item["dot"]), reverse=True)[:20],
            "layer_labels": layer_labels,
            "arch_labels": arch_labels,
            "component_layer_fracs": component_layer_fracs,
            "component_arch_fracs": component_arch_fracs,
        }
        self.metadata = {
            "base_paths": [model["manifest"].base_path for model in self.models],
            "base_fingerprints": [model["manifest"].base_fingerprint for model in self.models],
            "target_paths": [model["manifest"].target_path for model in self.models],
            "target_fingerprints": [model["manifest"].target_fingerprint for model in self.models],
        }
        return self.results

    def save(self, output_dir):
        if self.results is None:
            self.compute()
        ensure_dir(output_dir)
        npz_path = Path(output_dir) / "svd_analysis.npz"
        npz_tmp = npz_path.with_suffix(npz_path.suffix + ".tmp")
        with open(npz_tmp, "wb") as handle:
            np.savez_compressed(
                handle,
                schema_version=np.array(self.results["schema_version"], dtype="U32"),
                labels=np.array(self.results["labels"], dtype="U512"),
                delta_dirs=np.array(self.results["delta_dirs"], dtype="U1024"),
                n_models=np.array(self.results["n_models"], dtype=np.int64),
                n_components=np.array(self.results["n_components"], dtype=np.int64),
                shared_tensor_count=np.array(self.results["shared_tensor_count"], dtype=np.int64),
                excluded_tensor_count=np.array(self.results["excluded_tensor_count"], dtype=np.int64),
                excluded_tensors=np.array(self.results["excluded_tensors"], dtype="U512"),
                total_shared_params=np.array(self.results["total_shared_params"], dtype=np.int64),
                support_threshold=np.array(self.results["support_threshold"], dtype=np.float64),
                singular_values=self.results["singular_values"],
                left_vectors=self.results["left_vectors"],
                model_coefficients=self.results["model_coefficients"],
                all_singular_values=self.results["all_singular_values"],
                all_eigenvalues=self.results["all_eigenvalues"],
                explained_energy_ratio=self.results["explained_energy_ratio"],
                all_explained_energy_ratio=self.results["all_explained_energy_ratio"],
                gram=self.results["gram"],
                norms_sq=self.results["norms_sq"],
                cosine_similarity=self.results["cosine_similarity"],
                support_jaccard=self.results["support_jaccard"],
                shared_energy_frac=self.results["shared_energy_frac"],
                unique_residual=self.results["unique_residual"],
                sign_conflict_count=np.array(self.results["sign_conflict_count"], dtype=np.int64),
                sign_conflict_energy=np.array(self.results["sign_conflict_energy"], dtype=np.float64),
                layer_labels=np.array(self.results["layer_labels"], dtype="U64"),
                arch_labels=np.array(self.results["arch_labels"], dtype="U64"),
                component_layer_fracs=self.results["component_layer_fracs"],
                component_arch_fracs=self.results["component_arch_fracs"],
            )
        os.replace(npz_tmp, npz_path)
        payload = {key: self._jsonify(value) for key, value in self.results.items()}
        payload.update(self.metadata)
        json_path = Path(output_dir) / "svd_analysis.json"
        json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
        with open(json_tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(json_tmp, json_path)
        csv_path = Path(output_dir) / "svd_coefficients.csv"
        csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with open(csv_tmp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model"] + [f"svd_{idx + 1}" for idx in range(self.results["n_components"])])
            for label, row in zip(self.results["labels"], self.results["model_coefficients"]):
                writer.writerow([label] + [float(value) for value in row])
        os.replace(csv_tmp, csv_path)
        if self.block_diag is not None:
            from .support import save_block_diagnostics, save_overlap_csv

            save_block_diagnostics(output_dir, self.block_diag)
            save_overlap_csv(self.results, self.block_diag, output_dir)

    def plot(self, output_dir):
        if self.results is None:
            self.compute()
        ensure_dir(output_dir)
        labels = self.results["labels"]
        k = self.results["n_components"]
        self._plot_heatmap(
            self.results["model_coefficients"],
            [f"svd_{idx + 1}" for idx in range(k)],
            labels,
            os.path.join(output_dir, "svd_model_coefficients.png"),
            "SVD model coefficients",
            cmap="RdBu_r",
            annotate=len(labels) <= 15 and k <= 10,
            center_zero=True,
        )
        self._plot_explained_energy(os.path.join(output_dir, "svd_explained_energy.png"))
        self._plot_heatmap(
            self.results["component_layer_fracs"],
            self.results["layer_labels"],
            [f"svd_{idx + 1}" for idx in range(k)],
            os.path.join(output_dir, "svd_component_layers.png"),
            "Component layer localization",
            cmap="hot",
        )
        self._plot_heatmap(
            self.results["component_arch_fracs"],
            self.results["arch_labels"],
            [f"svd_{idx + 1}" for idx in range(k)],
            os.path.join(output_dir, "svd_component_architecture.png"),
            "Component architecture localization",
            cmap="hot",
        )
        self._plot_heatmap(
            self.results["cosine_similarity"],
            labels,
            labels,
            os.path.join(output_dir, "model_cosine_similarity.png"),
            "Model cosine similarity",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            center_zero=True,
        )
        self._plot_heatmap(
            self.results["support_jaccard"],
            labels,
            labels,
            os.path.join(output_dir, "model_support_jaccard.png"),
            "Support Jaccard",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        self._plot_unique_residual(os.path.join(output_dir, "model_unique_residual.png"))
        self._plot_heatmap(
            self.results["shared_energy_frac"],
            labels,
            labels,
            os.path.join(output_dir, "model_shared_energy.png"),
            "Directional shared energy fraction",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
        )
        if self.block_diag is not None:
            from .support import plot_departure_energy, plot_novel_support

            plot_novel_support(
                self.results, self.block_diag, os.path.join(output_dir, "svd_novel_support.png")
            )
            plot_departure_energy(
                self.results,
                self.block_diag,
                labels[0],
                os.path.join(output_dir, "svd_departure_energy.png"),
            )

    def _plot_explained_energy(self, path):
        explained = self.results["explained_energy_ratio"]
        cumulative = np.cumsum(explained)
        fig, ax1 = self._new_figure(8, 4.5)
        xs = np.arange(1, len(explained) + 1)
        ax1.bar(xs, explained, color="#4472c4")
        ax1.set_xlabel("Component")
        ax1.set_ylabel("Explained energy")
        ax1.set_xticks(xs)
        ax2 = ax1.twinx()
        ax2.plot(xs, cumulative, color="#d62728", marker="o")
        ax2.set_ylabel("Cumulative")
        ax1.set_title("Explained energy by component")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _plot_unique_residual(self, path):
        values = self.results["unique_residual"]
        colors = ["#1f77b4" if value >= 0.5 else "#ff7f0e" if value >= 0.2 else "#d62728" for value in values]
        fig, ax = self._new_figure(8, 4.5)
        ax.bar(np.arange(len(values)), values, color=colors)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(self.results["labels"], rotation=45, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Fraction")
        ax.set_title("Unique residual")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _plot_heatmap(
        self,
        matrix,
        x_labels,
        y_labels,
        path,
        title,
        cmap,
        vmin=None,
        vmax=None,
        annotate=False,
        center_zero=False,
    ):
        width = max(5.5, min(16.0, 1.0 + 0.55 * len(x_labels)))
        height = max(4.0, min(16.0, 1.2 + 0.45 * len(y_labels)))
        fig, ax = self._new_figure(width, height)
        if center_zero and vmin is None and vmax is None:
            bound = float(np.max(np.abs(matrix))) if np.size(matrix) else 1.0
            vmin, vmax = -bound, bound
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_title(title)
        if annotate:
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _new_figure(self, width, height):
        dpi = 150
        max_inches = 3000.0 / dpi
        fig, ax = plt.subplots(figsize=(min(width, max_inches), min(height, max_inches)))
        return fig, ax

    def _jsonify(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: self._jsonify(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonify(item) for item in value]
        return value


def svd_command(args) -> None:
    registry_path = Path(args.registry)
    with open(registry_path, "r", encoding="utf-8") as handle:
        registry = json.load(handle)
    output_dir = Path(args.output_dir)
    delta_root = registry_path.parent
    models = registry.get("models", [])
    if len(models) < 2:
        raise ValueError("SVD requires at least two models in the registry")

    cross_json = output_dir / "cross_analysis.json"
    if cross_json.exists():
        with open(cross_json, "r", encoding="utf-8") as handle:
            cross_labels = json.load(handle).get("labels", [])
        registry_labels = [model["name"] for model in models]
        if cross_labels != registry_labels:
            raise ValueError("Existing cross-analysis model order does not match registry.json")

    analyzer = SVDAnalyzer()
    for model in models:
        delta_dir = Path(model.get("delta_path", delta_root / model["slug"]))
        manifest = load_manifest(str(delta_dir))
        if manifest is None:
            raise FileNotFoundError(f"Missing delta manifest: {delta_dir}")
        if manifest.target_fingerprint != model["fingerprint"]:
            raise ValueError(f"Target fingerprint mismatch for {model['name']}")
        if not manifest.shards or any(
            shard.status != "complete" for shard in manifest.shards.values()
        ):
            raise ValueError(f"Incomplete delta shards for {model['name']}")
        analyzer.add_model(str(delta_dir), model["name"])

    try:
        analyzer.compute(progress_dir=str(output_dir))
        analyzer._write_progress(str(output_dir), phase="saving")
        analyzer.save(str(output_dir))
        analyzer.plot(str(output_dir))
        analyzer._write_progress(str(output_dir), phase="complete")
    except BaseException as exc:
        analyzer._write_progress(
            str(output_dir),
            phase="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    print(output_dir)
