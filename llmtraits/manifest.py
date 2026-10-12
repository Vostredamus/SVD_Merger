import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import GLOBAL_SHARD_NAME, LAYER_SHARD_PREFIX, MANIFEST_NAME
from .paths import relocated_path
from .utils import ensure_dir


@dataclass
class TensorRecord:
    name: str
    shard_file: str
    shape: list[int]
    orig_dtype: str
    stored_dtype: str
    base_sha256: str
    delta_sha256: str


@dataclass
class ShardStatus:
    name: str
    status: str
    file_sha256: str | None


@dataclass
class Manifest:
    base_path: str
    base_fingerprint: str
    target_path: str
    target_fingerprint: str
    delta_dtype: str
    tensors: dict[str, TensorRecord] = field(default_factory=dict)
    shards: dict[str, ShardStatus] = field(default_factory=dict)
    missing_tensors: list[str] = field(default_factory=list)
    extra_tensors: list[str] = field(default_factory=list)
    incompatible_tensors: list[str] = field(default_factory=list)
    stats_csv: str | None = None
    stats_json: str | None = None
    landscapes_npz: str | None = None
    created_at: str = ""
    updated_at: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest_path(delta_dir: str) -> Path:
    return relocated_path(delta_dir) / MANIFEST_NAME


def _record_from_dict(cls, payload: dict):
    return cls(**payload)


def load_manifest(delta_dir: str) -> Manifest | None:
    path = _manifest_path(delta_dir)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["tensors"] = {name: _record_from_dict(TensorRecord, record) for name, record in data.get("tensors", {}).items()}
    data["shards"] = {name: _record_from_dict(ShardStatus, record) for name, record in data.get("shards", {}).items()}
    return Manifest(**data)


def save_manifest(delta_dir: str, manifest: Manifest):
    ensure_dir(delta_dir)
    manifest.updated_at = utc_now()
    if not manifest.created_at:
        manifest.created_at = manifest.updated_at
    payload = asdict(manifest)
    path = _manifest_path(delta_dir)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(str(tmp_path), str(path))


def plan_shards(base_tensors: dict[str, str], target_tensors: dict[str, str]) -> dict[str, list[str]]:
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.")
    groups: dict[str, list[str]] = {GLOBAL_SHARD_NAME: []}
    all_names = sorted(set(base_tensors) | set(target_tensors))
    for name in all_names:
        match = layer_pattern.match(name)
        if match:
            layer_idx = int(match.group(1))
            shard_name = f"{LAYER_SHARD_PREFIX}_{layer_idx:03d}.safetensors"
            groups.setdefault(shard_name, []).append(name)
        else:
            groups.setdefault(GLOBAL_SHARD_NAME, []).append(name)
    return {shard: sorted(names) for shard, names in sorted(groups.items()) if names}
