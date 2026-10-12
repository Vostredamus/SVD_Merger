import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .manifest import load_manifest
from .paths import relocated_path
from .utils import ensure_dir, load_tensor_index, sha256_file


@dataclass
class DeltaSpec:
    delta_dir: str
    weight: float
    label: str | None = None


def _group_by_basename(tensor_index: dict[str, str]) -> dict[str, list[str]]:
    grouped = {}
    for name, path in tensor_index.items():
        grouped.setdefault(os.path.basename(path), []).append(name)
    return {key: sorted(value) for key, value in sorted(grouped.items())}


def _load_named_tensors(file_path: str, names: list[str]) -> dict[str, torch.Tensor]:
    if file_path.lower().endswith(".safetensors"):
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            return {name: handle.get_tensor(name).to(torch.float32) for name in names}
    state = torch.load(file_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return {name: state[name].to(torch.float32) for name in names}


def _copy_metadata_files(base_dir: str, output_dir: str):
    allowed = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "chat_template.jinja",
    }
    for name in allowed:
        src = Path(base_dir) / name
        if src.exists():
            shutil.copy2(src, Path(output_dir) / name)


def archive_model_metadata(model_dir: str, delta_dir: str) -> Path:
    """Preserve non-weight checkpoint files needed after deleting a full model."""

    source = relocated_path(model_dir)
    destination = relocated_path(delta_dir) / "model_metadata"
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if (
            name.endswith((".safetensors", ".bin"))
            or name == "model.safetensors.index.json"
            or name == "pytorch_model.bin.index.json"
        ):
            continue
        shutil.copy2(path, destination / name)
    return destination


def _load_compose_state(output_dir: str) -> dict:
    path = Path(output_dir) / "compose_state.json"
    if not path.exists():
        return {"shards": {}}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.setdefault("shards", {})
    return payload


def _save_compose_state(output_dir: str, state: dict) -> None:
    path = Path(output_dir) / "compose_state.json"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    os.replace(str(tmp_path), str(path))


def _filtered_specs(delta_specs: list["DeltaSpec"]) -> list["DeltaSpec"]:
    return [spec for spec in delta_specs if abs(float(spec.weight)) >= 1e-12]


def _compose_impl(
    base_dir: str,
    delta_specs: list[DeltaSpec],
    output_dir: str,
    dtype: str,
    missing_policy: str,
    resumable: bool,
) -> None:
    ensure_dir(output_dir)
    base_index = load_tensor_index(base_dir)
    shard_groups = _group_by_basename(base_index)
    active_specs = _filtered_specs(delta_specs)
    manifests = []
    for spec in active_specs:
        manifest = load_manifest(spec.delta_dir)
        if manifest is None:
            raise FileNotFoundError(f"No manifest found in {spec.delta_dir}")
        manifests.append((spec, manifest))

    state = _load_compose_state(output_dir) if resumable else {"shards": {}}
    weight_map = {}
    for shard_name, tensor_names in shard_groups.items():
        shard_path = Path(output_dir) / shard_name
        if resumable and shard_path.exists():
            expected_sha = state["shards"].get(shard_name)
            if expected_sha and sha256_file(str(shard_path)) == expected_sha:
                print(f"Skipping complete shard: {shard_name}")
                for name in tensor_names:
                    weight_map[name] = shard_name
                continue
        source_path = next(base_index[name] for name in tensor_names)
        base_tensors = _load_named_tensors(source_path, tensor_names)
        out_tensors = {}
        for name in tensor_names:
            accum = base_tensors[name].clone()
            for spec, manifest in manifests:
                record = manifest.tensors.get(name)
                if record is None:
                    if missing_policy == "error":
                        raise ValueError(f"Missing delta tensor: {name} in {spec.delta_dir}")
                    if missing_policy in {"skip", "zero"}:
                        continue
                shard_delta_path = relocated_path(spec.delta_dir) / record.shard_file
                with safe_open(str(shard_delta_path), framework="pt", device="cpu") as handle:
                    delta_tensor = handle.get_tensor(name).to(torch.float32)
                if list(delta_tensor.shape) != list(accum.shape):
                    if missing_policy == "error":
                        raise ValueError(f"Shape mismatch for {name}: {tuple(delta_tensor.shape)} vs {tuple(accum.shape)}")
                    if missing_policy in {"skip", "zero"}:
                        continue
                accum = accum + float(spec.weight) * delta_tensor
            if dtype == "source":
                source_dtypes = {
                    getattr(torch, manifest.tensors[name].orig_dtype)
                    for _, manifest in manifests
                    if name in manifest.tensors
                }
                if len(source_dtypes) != 1:
                    raise ValueError(
                        f"Cannot infer one source dtype for {name}: {source_dtypes}"
                    )
                output_dtype = source_dtypes.pop()
            else:
                output_dtype = torch.float16 if dtype == "float16" else torch.float32
            out_tensors[name] = accum.to(output_dtype).contiguous()
            weight_map[name] = shard_name
        tmp_path = Path(str(shard_path) + ".tmp")
        save_file(out_tensors, str(tmp_path))
        os.replace(str(tmp_path), str(shard_path))
        if resumable:
            state["shards"][shard_name] = sha256_file(str(shard_path))
            _save_compose_state(output_dir, state)

    if len(shard_groups) > 1:
        index_payload = {
            "metadata": {"total_size": sum((Path(output_dir) / shard).stat().st_size for shard in shard_groups)},
            "weight_map": weight_map,
        }
        with open(Path(output_dir) / "model.safetensors.index.json", "w", encoding="utf-8") as handle:
            json.dump(index_payload, handle, indent=2, sort_keys=True)
    _copy_metadata_files(base_dir, output_dir)
    if (
        len(active_specs) == 1
        and float(active_specs[0].weight) == 1.0
        and (relocated_path(active_specs[0].delta_dir) / "model_metadata").is_dir()
    ):
        _copy_metadata_files(
            str(relocated_path(active_specs[0].delta_dir) / "model_metadata"),
            output_dir,
        )
    provenance = {
        "base_dir": os.path.abspath(base_dir),
        "deltas": [
            {"delta_dir": os.path.abspath(spec.delta_dir), "weight": spec.weight, "label": spec.label}
            for spec in delta_specs
        ],
        "dtype": dtype,
        "missing_policy": missing_policy,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(Path(output_dir) / "provenance.json", "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)


def compose(
    base_dir: str,
    delta_specs: list[DeltaSpec],
    output_dir: str,
    dtype: str = "float16",
    missing_policy: str = "error",
) -> None:
    _compose_impl(base_dir, delta_specs, output_dir, dtype, missing_policy, resumable=False)


def compose_resumable(
    base_dir: str,
    delta_specs: list[DeltaSpec],
    output_dir: str,
    dtype: str = "float16",
    missing_policy: str = "error",
) -> None:
    _compose_impl(base_dir, delta_specs, output_dir, dtype, missing_policy, resumable=True)


def compose_command(args) -> None:
    specs = []
    for item in args.delta:
        match = re.fullmatch(
            r"(.*):([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?::([^:]*))?",
            item,
        )
        if not match or not match.group(1):
            raise ValueError(f"Invalid --delta spec: {item}")
        delta_dir, weight, label = match.group(1), float(match.group(2)), match.group(3) or None
        specs.append(DeltaSpec(delta_dir=delta_dir, weight=weight, label=label))
    compose(
        base_dir=args.base,
        delta_specs=specs,
        output_dir=args.output,
        dtype=args.delta_dtype,
        missing_policy=args.missing_policy,
    )
