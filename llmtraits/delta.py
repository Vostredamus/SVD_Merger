import json
import logging
import math
import os
import signal
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm import tqdm

from .manifest import Manifest, ShardStatus, TensorRecord, load_manifest, plan_shards, save_manifest, utc_now
from .paths import relocated_path
from .scan import cached_model_fingerprint
from .stats import TensorStats, compute_stats, save_stats
from .utils import (
    atomic_save_safetensors,
    ensure_dir,
    free_space_gb,
    load_tensor_index,
    model_size_bytes,
    sha256_file,
    tensor_bytes_sha256,
)
from .visualize import compute_landscapes
from .compose import archive_model_metadata

LOGGER = logging.getLogger(__name__)


class DeltaComputer:
    def __init__(self, base_dir, target_dir, output_dir, dtype="float16", force=False, skip_landscapes=False):
        self.base_dir = os.path.abspath(relocated_path(base_dir))
        self.target_dir = os.path.abspath(relocated_path(target_dir))
        self.output_dir = os.path.abspath(relocated_path(output_dir))
        self.dtype = dtype
        self.force = force
        self.skip_landscapes = skip_landscapes
        self._current_tmp_path: Path | None = None

    def run(self) -> Manifest:
        ensure_dir(self.output_dir)
        self._install_signal_handlers()
        base_index = load_tensor_index(self.base_dir)
        target_index = load_tensor_index(self.target_dir)
        base_meta = self._tensor_metadata(self.base_dir, base_index)
        target_meta = self._tensor_metadata(self.target_dir, target_index)
        base_fp = cached_model_fingerprint(self.base_dir)
        target_fp = cached_model_fingerprint(self.target_dir)

        manifest = None if self.force else load_manifest(self.output_dir)
        if manifest is None or manifest.base_fingerprint != base_fp or manifest.target_fingerprint != target_fp or manifest.delta_dtype != self.dtype:
            manifest = Manifest(
                base_path=self.base_dir,
                base_fingerprint=base_fp,
                target_path=self.target_dir,
                target_fingerprint=target_fp,
                delta_dtype=self.dtype,
                created_at=utc_now(),
                updated_at=utc_now(),
            )

        shard_plan = plan_shards(base_index, target_index)
        self._reconcile_manifest(manifest, shard_plan)
        manifest.missing_tensors = sorted(set(base_index) - set(target_index))
        manifest.extra_tensors = sorted(set(target_index) - set(base_index))
        incompatible = []
        for name in sorted(set(base_index) & set(target_index)):
            base_shape = base_meta[name]["shape"]
            target_shape = target_meta[name]["shape"]
            if self._mark_incompatible(base_shape, target_shape, name):
                incompatible.append(name)
        manifest.incompatible_tensors = sorted(incompatible)
        self._verify_existing_shards(manifest)
        save_manifest(self.output_dir, manifest)

        stats_by_name: dict[str, TensorStats] = self._load_existing_stats(manifest)
        compatible = {
            name
            for name in set(base_index) & set(target_index)
            if name not in manifest.incompatible_tensors
        }
        stats_by_name.update(self._load_shard_stats(manifest))
        self._invalidate_shards_without_stats(manifest, shard_plan, compatible, stats_by_name)
        save_manifest(self.output_dir, manifest)

        for shard_name, names in tqdm(list(shard_plan.items()), desc="delta shards"):
            shard_status = manifest.shards[shard_name]
            shard_path = Path(self.output_dir) / shard_name
            names = [name for name in names if name in compatible]
            if not names:
                shard_status.status = "complete"
                shard_status.file_sha256 = None
                save_manifest(self.output_dir, manifest)
                continue
            if shard_status.status == "complete" and shard_status.file_sha256 and shard_path.exists():
                continue

            self._check_disk_space(names, base_meta)
            delta_tensors = {}
            shard_stats: dict[str, TensorStats] = {}
            for name in names:
                base_tensor, base_dtype = self._load_tensor_with_dtype(
                    base_index[name], name
                )
                target_tensor, target_dtype = self._load_tensor_with_dtype(
                    target_index[name], name
                )
                delta_f32 = target_tensor - base_tensor
                stored = delta_f32.to(torch.float16 if self.dtype == "float16" else torch.float32)
                delta_tensors[name] = stored.contiguous()
                shard_stats[name] = compute_stats(
                    name, base_tensor, target_tensor, delta_f32
                )
                manifest.tensors[name] = TensorRecord(
                    name=name,
                    shard_file=shard_name,
                    shape=list(delta_f32.shape),
                    orig_dtype=str(target_dtype or base_dtype).replace("torch.", ""),
                    stored_dtype=self.dtype,
                    base_sha256=tensor_bytes_sha256(base_tensor),
                    delta_sha256=tensor_bytes_sha256(delta_tensors[name]),
                )
                del base_tensor, target_tensor, delta_f32, stored

            shard_status.status = "pending"
            shard_status.file_sha256 = None
            save_manifest(self.output_dir, manifest)
            self._current_tmp_path = Path(str(shard_path) + ".tmp")
            try:
                file_sha256 = atomic_save_safetensors(delta_tensors, str(shard_path))
                self._save_shard_stats(shard_name, shard_stats)
                stats_by_name.update(shard_stats)
            finally:
                self._cleanup_tmp()
                self._current_tmp_path = None
            shard_status.status = "complete"
            shard_status.file_sha256 = file_sha256
            save_manifest(self.output_dir, manifest)

        stats_list = [stats_by_name[name] for name in sorted(stats_by_name)]
        csv_path, json_path = save_stats(stats_list, self.output_dir)
        manifest.stats_csv = os.path.basename(csv_path)
        manifest.stats_json = os.path.basename(json_path)
        if not self.skip_landscapes:
            manifest.landscapes_npz = os.path.basename(compute_landscapes(self.output_dir, manifest))
        archive_model_metadata(self.target_dir, self.output_dir)
        save_manifest(self.output_dir, manifest)
        return manifest

    def _install_signal_handlers(self):
        def _handler(signum, _frame):
            self._cleanup_tmp()
            raise KeyboardInterrupt(f"Interrupted by signal {signum}")

        signal.signal(signal.SIGINT, _handler)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _handler)

    def _cleanup_tmp(self):
        if self._current_tmp_path and self._current_tmp_path.exists():
            self._current_tmp_path.unlink()

    def _check_disk_space(self, names, base_meta):
        bytes_per_value = 2 if self.dtype == "float16" else 4
        estimated = sum(base_meta[name]["numel"] * bytes_per_value for name in names)
        required_gb = estimated / float(1024 ** 3) + 10.0
        free_gb = free_space_gb(self.output_dir)
        if free_gb < required_gb:
            raise RuntimeError(
                f"Insufficient free space for shard: need {required_gb:.2f} GB, have {free_gb:.2f} GB"
            )

    def _verify_existing_shards(self, manifest: Manifest):
        for shard_name, status in manifest.shards.items():
            if status.status != "complete" or not status.file_sha256:
                continue
            shard_path = Path(self.output_dir) / shard_name
            if not shard_path.exists() or sha256_file(str(shard_path)) != status.file_sha256:
                status.status = "corrupt"
                status.file_sha256 = None

    def _reconcile_manifest(self, manifest: Manifest, shard_plan: dict[str, list[str]]):
        current = set(manifest.shards)
        planned = set(shard_plan)
        for shard_name in planned - current:
            manifest.shards[shard_name] = ShardStatus(name=shard_name, status="pending", file_sha256=None)
        for shard_name in current - planned:
            del manifest.shards[shard_name]

    def _load_existing_stats(self, manifest: Manifest) -> dict[str, TensorStats]:
        if not manifest.stats_json:
            return {}
        path = Path(self.output_dir) / manifest.stats_json
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {item["name"]: TensorStats(**item) for item in payload}

    def _stats_fragment_path(self, shard_name: str) -> Path:
        return Path(self.output_dir) / ".state" / "stats" / f"{Path(shard_name).stem}.json"

    def _save_shard_stats(self, shard_name: str, stats: dict[str, TensorStats]):
        path = self._stats_fragment_path(shard_name)
        ensure_dir(str(path.parent))
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump([asdict(stats[name]) for name in sorted(stats)], handle, indent=2)
        os.replace(str(tmp_path), str(path))

    def _load_shard_stats(self, manifest: Manifest) -> dict[str, TensorStats]:
        result: dict[str, TensorStats] = {}
        for shard_name, status in manifest.shards.items():
            if status.status != "complete":
                continue
            path = self._stats_fragment_path(shard_name)
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as handle:
                for item in json.load(handle):
                    result[item["name"]] = TensorStats(**item)
        return result

    def _invalidate_shards_without_stats(self, manifest, shard_plan, compatible, stats_by_name):
        for shard_name, status in manifest.shards.items():
            if status.status != "complete" or not status.file_sha256:
                continue
            expected = {name for name in shard_plan[shard_name] if name in compatible}
            if not expected.issubset(stats_by_name):
                LOGGER.warning("Recomputing %s because resumable statistics are incomplete", shard_name)
                status.status = "pending"
                status.file_sha256 = None

    def _tensor_metadata(self, model_dir: str, tensor_index: dict[str, str]) -> dict[str, dict]:
        metadata: dict[str, dict] = {}
        grouped: dict[str, list[str]] = defaultdict(list)
        for name, path in tensor_index.items():
            grouped[path].append(name)
        for path, names in grouped.items():
            if path.lower().endswith(".safetensors"):
                with safe_open(path, framework="pt", device="cpu") as handle:
                    for name in names:
                        tensor = handle.get_tensor(name)
                        metadata[name] = {
                            "shape": list(tensor.shape),
                            "dtype": tensor.dtype,
                            "numel": int(tensor.numel()),
                        }
            else:
                state = torch.load(path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                    state = state["state_dict"]
                for name in names:
                    tensor = state[name]
                    metadata[name] = {
                        "shape": list(tensor.shape),
                        "dtype": tensor.dtype,
                        "numel": int(tensor.numel()),
                    }
        return metadata

    def _load_shard_tensors(self, model_dir, tensor_index, names) -> dict[str, torch.Tensor]:
        tensors, _ = self._load_shard_tensors_with_dtypes(model_dir, tensor_index, names)
        return tensors

    def _load_tensor_with_dtype(
        self, path: str, name: str
    ) -> tuple[torch.Tensor, torch.dtype]:
        if path.lower().endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(name)
                return tensor.to(torch.float32), tensor.dtype
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(
            state["state_dict"], dict
        ):
            state = state["state_dict"]
        tensor = state[name]
        return tensor.to(torch.float32), tensor.dtype

    def _load_shard_tensors_with_dtypes(self, model_dir, tensor_index, names):
        grouped: dict[str, list[str]] = defaultdict(list)
        for name in names:
            grouped[tensor_index[name]].append(name)
        tensors: dict[str, torch.Tensor] = {}
        dtypes: dict[str, torch.dtype] = {}
        for path, file_names in grouped.items():
            if path.lower().endswith(".safetensors"):
                with safe_open(path, framework="pt", device="cpu") as handle:
                    for name in file_names:
                        tensor = handle.get_tensor(name)
                        dtypes[name] = tensor.dtype
                        tensors[name] = tensor.to(torch.float32)
            else:
                state = torch.load(path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                    state = state["state_dict"]
                for name in file_names:
                    tensor = state[name]
                    dtypes[name] = tensor.dtype
                    tensors[name] = tensor.to(torch.float32)
        return tensors, dtypes

    def _mark_incompatible(self, base_shape, target_shape, name) -> bool:
        return list(base_shape) != list(target_shape)
