import csv
import json
import math
import os
from dataclasses import asdict, dataclass

import torch

from .utils import ensure_dir


@dataclass
class TensorStats:
    name: str
    shape: list[int]
    n_params: int
    base_l2: float
    target_l2: float
    delta_l2: float
    base_rms: float
    target_rms: float
    delta_rms: float
    rel_l2: float
    mean_abs: float
    max_abs: float
    cosine_sim: float
    sparsity_exact: float
    sparsity_1e3: float


def compute_stats(name, base_t, target_t, delta_t) -> TensorStats:
    base_f = base_t.to(torch.float32)
    target_f = target_t.to(torch.float32)
    delta_f = delta_t.to(torch.float32)
    n_params = int(delta_f.numel())
    base_l2 = float(torch.linalg.vector_norm(base_f).item())
    target_l2 = float(torch.linalg.vector_norm(target_f).item())
    delta_l2 = float(torch.linalg.vector_norm(delta_f).item())
    base_rms = math.sqrt(float(torch.mean(base_f.square()).item()))
    target_rms = math.sqrt(float(torch.mean(target_f.square()).item()))
    delta_rms = math.sqrt(float(torch.mean(delta_f.square()).item()))
    denom = max(base_l2 * target_l2, 1e-12)
    cosine_sim = float(torch.dot(base_f.reshape(-1), target_f.reshape(-1)).item() / denom)
    sparsity_exact = float((delta_f == 0).to(torch.float32).mean().item())
    sparsity_1e3 = float((delta_f.abs() < 1e-3).to(torch.float32).mean().item())
    return TensorStats(
        name=name,
        shape=list(delta_f.shape),
        n_params=n_params,
        base_l2=base_l2,
        target_l2=target_l2,
        delta_l2=delta_l2,
        base_rms=base_rms,
        target_rms=target_rms,
        delta_rms=delta_rms,
        rel_l2=delta_l2 / (base_l2 + 1e-12),
        mean_abs=float(delta_f.abs().mean().item()),
        max_abs=float(delta_f.abs().max().item()),
        cosine_sim=cosine_sim,
        sparsity_exact=sparsity_exact,
        sparsity_1e3=sparsity_1e3,
    )


def save_stats(stats_list: list[TensorStats], output_dir: str) -> tuple[str, str]:
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "stats.csv")
    json_path = os.path.join(output_dir, "stats.json")
    field_names = list(asdict(stats_list[0]).keys()) if stats_list else list(TensorStats.__annotations__.keys())
    csv_tmp = csv_path + ".tmp"
    json_tmp = json_path + ".tmp"
    with open(csv_tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for stat in stats_list:
            writer.writerow(asdict(stat))
    with open(json_tmp, "w", encoding="utf-8") as handle:
        json.dump([asdict(stat) for stat in stats_list], handle, indent=2)
    os.replace(csv_tmp, csv_path)
    os.replace(json_tmp, json_path)
    return csv_path, json_path
