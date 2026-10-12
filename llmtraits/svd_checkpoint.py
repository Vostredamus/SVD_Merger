import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CHECKPOINT_SCHEMA_VERSION = "1.0"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


class SVDCheckpointStore:
    def __init__(self, progress_dir: str | None, identity: dict):
        self.enabled = bool(progress_dir)
        self.root = Path(progress_dir) / "svd_checkpoints" if progress_dir else None
        self.identity = {"schema_version": CHECKPOINT_SCHEMA_VERSION, **identity}
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        identity_path = self.root / "identity.json"
        if identity_path.exists():
            with open(identity_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing != self.identity:
                raise ValueError(
                    "SVD checkpoint identity does not match the current models or settings. "
                    f"Move or delete {self.root} before starting a different analysis."
                )
        else:
            _atomic_json(identity_path, self.identity)

    def _paths(self, phase: str, tensor_index: int):
        stem = f"{tensor_index:04d}"
        directory = self.root / phase
        return directory / f"{stem}.npz", directory / f"{stem}.json"

    def load_gram(self, tensor_index: int, name: str):
        if not self.enabled:
            return None
        npz_path, json_path = self._paths("gram", tensor_index)
        if not npz_path.exists() or not json_path.exists():
            return None
        with open(json_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("tensor_name") != name:
            raise ValueError(f"Gram checkpoint tensor mismatch at index {tensor_index}")
        with np.load(npz_path, allow_pickle=False) as data:
            gram = data["gram"].astype(np.float64, copy=False)
            params = int(data["params"].item())
        return gram, params

    def save_gram(self, tensor_index: int, name: str, gram: np.ndarray, params: int):
        if not self.enabled:
            return
        npz_path, json_path = self._paths("gram", tensor_index)
        _atomic_npz(
            npz_path,
            gram=np.asarray(gram, dtype=np.float64),
            params=np.array(params, dtype=np.int64),
        )
        _atomic_json(
            json_path,
            {
                "tensor_index": tensor_index,
                "tensor_name": name,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def load_diagnostic(self, tensor_index: int, name: str):
        if not self.enabled:
            return None
        npz_path, json_path = self._paths("block_diagnostics", tensor_index)
        if not npz_path.exists() or not json_path.exists():
            return None
        with open(json_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("tensor_name") != name:
            raise ValueError(f"Diagnostic checkpoint tensor mismatch at index {tensor_index}")
        with np.load(npz_path, allow_pickle=False) as data:
            result = {
                "novel_support_frac": data["novel_support_frac"],
                "shared_support_frac": data["shared_support_frac"],
                "sign_conflict_num": data["sign_conflict_num"],
                "material_total": data["material_total"],
                "novel_layer_energy": data["novel_layer_energy"],
                "novel_arch_energy": data["novel_arch_energy"],
                "total_blocks": int(data["total_blocks"].item()),
            }
        result["top_blocks"] = metadata.get("top_blocks", [])
        return result

    def save_diagnostic(self, tensor_index: int, name: str, diagnostic: dict):
        if not self.enabled:
            return
        npz_path, json_path = self._paths("block_diagnostics", tensor_index)
        _atomic_npz(
            npz_path,
            novel_support_frac=diagnostic["novel_support_frac"],
            shared_support_frac=diagnostic["shared_support_frac"],
            sign_conflict_num=diagnostic["sign_conflict_num"],
            material_total=diagnostic["material_total"],
            novel_layer_energy=diagnostic["novel_layer_energy"],
            novel_arch_energy=diagnostic["novel_arch_energy"],
            total_blocks=np.array(diagnostic["total_blocks"], dtype=np.int64),
        )
        _atomic_json(
            json_path,
            {
                "tensor_index": tensor_index,
                "tensor_name": name,
                "top_blocks": diagnostic.get("top_blocks", []),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
