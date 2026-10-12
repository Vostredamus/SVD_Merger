import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .manifest import load_manifest
from .paths import relocated_path
from .tensor_stream import streamed_tensor_gram
from .utils import ensure_dir


def _component_name(tensor_name: str) -> str:
    for component in [
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
    ]:
        if component in tensor_name:
            return component
    return "other"


@dataclass
class _CrossModel:
    delta_dir: str
    label: str
    manifest: object
    stats: dict


class CrossAnalyzer:
    def __init__(self):
        self.models: list[_CrossModel] = []
        self.results: dict | None = None

    def add_model(self, delta_dir: str, label: str | None = None):
        manifest = load_manifest(delta_dir)
        if manifest is None:
            raise FileNotFoundError(f"No manifest found in {delta_dir}")
        stats = {}
        if manifest.stats_json:
            with open(
                relocated_path(delta_dir) / manifest.stats_json,
                "r",
                encoding="utf-8",
            ) as handle:
                stats = {item["name"]: item for item in json.load(handle)}
        self.models.append(_CrossModel(delta_dir=delta_dir, label=label or Path(delta_dir).name, manifest=manifest, stats=stats))

    def compute(self) -> dict:
        if not self.models:
            raise ValueError("No models added")
        n = len(self.models)
        G = np.zeros((n, n), dtype=np.float64)
        per_group: dict[str, np.ndarray] = {}
        norms_sq = np.zeros(n, dtype=np.float64)
        tensor_energy = [dict() for _ in range(n)]
        sign_conflicts = []
        all_names = sorted({name for model in self.models for name in model.manifest.tensors})
        for name in all_names:
            records = [model.manifest.tensors.get(name) for model in self.models]
            if any(record is None for record in records):
                continue
            shapes = {tuple(record.shape) for record in records if record is not None}
            if len(shapes) != 1:
                continue
            tensor_gram = streamed_tensor_gram(
                [
                    {"delta_dir": model.delta_dir, "manifest": model.manifest}
                    for model in self.models
                ],
                name,
            )
            group = _component_name(name)
            group_G = per_group.setdefault(group, np.zeros((n, n), dtype=np.float64))
            for i in range(n):
                energy = float(tensor_gram[i, i])
                norms_sq[i] += energy
                tensor_energy[i][name] = energy
                for j in range(i, n):
                    dot = float(tensor_gram[i, j])
                    G[i, j] += dot
                    if i != j:
                        G[j, i] += dot
                    group_G[i, j] += dot
                    if i != j:
                        group_G[j, i] += dot
                    if i < j and dot < 0:
                        sign_conflicts.append({"pair": [self.models[i].label, self.models[j].label], "tensor": name, "dot": dot})
        cosine = np.zeros_like(G)
        for i in range(n):
            for j in range(n):
                denom = np.sqrt(max(norms_sq[i], 1e-12) * max(norms_sq[j], 1e-12))
                cosine[i, j] = G[i, j] / denom if denom else 0.0
        eigvals, eigvecs = np.linalg.eigh(G)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = eigvecs[:, order]
        total_var = float(np.sum(eigvals))
        explained = eigvals / total_var if total_var > 0 else np.zeros_like(eigvals)
        coords = eigvecs * np.sqrt(np.maximum(eigvals, 0.0))
        overlap = np.zeros_like(G)
        jaccard = np.zeros_like(G)
        exclusivity = np.zeros(n, dtype=np.float64)
        for i in range(n):
            shared_with_any = 0.0
            for j in range(n):
                if i == j:
                    overlap[i, j] = 1.0
                    jaccard[i, j] = 1.0
                    continue
                shared = max(G[i, j], 0.0)
                overlap[i, j] = shared / max(min(norms_sq[i], norms_sq[j]), 1e-12)
                union = norms_sq[i] + norms_sq[j] - shared
                jaccard[i, j] = shared / max(union, 1e-12)
                shared_with_any = max(shared_with_any, shared)
            exclusivity[i] = max(norms_sq[i] - shared_with_any, 0.0)

        top_tensors = {}
        for idx, model in enumerate(self.models):
            items = sorted(tensor_energy[idx].items(), key=lambda item: item[1], reverse=True)[:10]
            top_tensors[model.label] = [{"name": name, "energy": value} for name, value in items]

        layer_profiles = {}
        component_profiles = {}
        for model in self.models:
            by_layer = {}
            by_component = {}
            for name, stat in model.stats.items():
                if name.startswith("model.layers."):
                    layer = int(name.split(".")[2])
                    by_layer.setdefault(layer, []).append(float(stat["rel_l2"]))
                component = _component_name(name)
                by_component.setdefault(component, []).append(float(stat["rel_l2"]))
            layer_profiles[model.label] = {str(layer): float(np.mean(values)) for layer, values in sorted(by_layer.items())}
            component_profiles[model.label] = {component: float(np.mean(values)) for component, values in sorted(by_component.items())}

        self.results = {
            "labels": [model.label for model in self.models],
            "gram": G,
            "norms_sq": norms_sq,
            "cosine_similarity": cosine,
            "per_group_gram": per_group,
            "eigenvalues": eigvals,
            "explained_variance_ratio": explained,
            "pca_coords": coords,
            "overlap": overlap,
            "jaccard": jaccard,
            "exclusive_energy": exclusivity,
            "top_tensors": top_tensors,
            "sign_conflicts": sign_conflicts,
            "layer_profiles": layer_profiles,
            "component_profiles": component_profiles,
        }
        return self.results

    def save(self, output_dir: str):
        if self.results is None:
            self.compute()
        ensure_dir(output_dir)
        labels = self.results["labels"]
        np.savez_compressed(
            os.path.join(output_dir, "cross_analysis.npz"),
            gram=self.results["gram"],
            cosine_similarity=self.results["cosine_similarity"],
            eigenvalues=self.results["eigenvalues"],
            explained_variance_ratio=self.results["explained_variance_ratio"],
            pca_coords=self.results["pca_coords"],
            overlap=self.results["overlap"],
            jaccard=self.results["jaccard"],
            exclusive_energy=self.results["exclusive_energy"],
            labels=np.array(labels, dtype=object),
        )
        with open(os.path.join(output_dir, "cross_analysis.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    key: self._jsonify(value)
                    for key, value in self.results.items()
                    if key not in {"gram", "cosine_similarity", "eigenvalues", "explained_variance_ratio", "pca_coords", "overlap", "jaccard", "exclusive_energy", "per_group_gram", "norms_sq"}
                },
                handle,
                indent=2,
            )
        with open(os.path.join(output_dir, "cosine_similarity.csv"), "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["label"] + labels)
            for label, row in zip(labels, self.results["cosine_similarity"]):
                writer.writerow([label] + [float(value) for value in row])

    def plot(self, output_dir: str):
        if self.results is None:
            self.compute()
        ensure_dir(output_dir)
        labels = self.results["labels"]
        self._plot_heatmap(self.results["cosine_similarity"], labels, os.path.join(output_dir, "cosine_similarity.png"), "Cosine similarity")
        self._plot_heatmap(self.results["overlap"], labels, os.path.join(output_dir, "overlap.png"), "Positive overlap")
        self._plot_pca(os.path.join(output_dir, "pca.png"))
        self._plot_profiles(os.path.join(output_dir, "component_profiles.png"), self.results["component_profiles"], "Component profiles")
        self._plot_profiles(os.path.join(output_dir, "layer_profiles.png"), self.results["layer_profiles"], "Layer profiles")

    def _plot_heatmap(self, matrix, labels, path, title):
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _plot_pca(self, path):
        coords = self.results["pca_coords"]
        labels = self.results["labels"]
        fig, ax = plt.subplots(figsize=(6, 5))
        xs = coords[:, 0] if coords.shape[1] > 0 else np.zeros(len(labels))
        ys = coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(labels))
        ax.scatter(xs, ys)
        for x, y, label in zip(xs, ys, labels):
            ax.text(x, y, label)
        ax.set_title("PCA coordinates")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _plot_profiles(self, path, profiles: dict, title: str):
        fig, ax = plt.subplots(figsize=(8, 4))
        for label, mapping in profiles.items():
            keys = list(mapping.keys())
            ax.plot(range(len(keys)), list(mapping.values()), marker="o", label=label)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _jsonify(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: self._jsonify(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonify(item) for item in value]
        return value


def cross_command(args) -> None:
    analyzer = CrossAnalyzer()
    labels = args.label or []
    for idx, delta_dir in enumerate(args.delta_dirs):
        analyzer.add_model(delta_dir, label=labels[idx] if idx < len(labels) else None)
    analyzer.compute()
    output_dir = args.output_dir or os.path.join(os.getcwd(), "cross_analysis")
    analyzer.save(output_dir)
    analyzer.plot(output_dir)
    from .svd import SVDAnalyzer as _SVDAnalyzer

    svd = _SVDAnalyzer()
    for model in analyzer.models:
        svd.add_model(model.delta_dir, model.label)
    try:
        svd.compute(progress_dir=output_dir)
        svd._write_progress(output_dir, phase="saving")
        svd.save(output_dir)
        svd.plot(output_dir)
        svd._write_progress(output_dir, phase="complete")
    except BaseException as exc:
        svd._write_progress(
            output_dir,
            phase="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    print(output_dir)
