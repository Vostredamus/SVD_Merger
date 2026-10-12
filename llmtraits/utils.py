import hashlib
import json
import logging
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .paths import relocated_path

LOGGER = logging.getLogger(__name__)


def sha256_file(path: str, progress=None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            if progress:
                progress(len(chunk))
    return digest.hexdigest()


def sha256_model_dir(model_dir: str) -> str:
    root = Path(model_dir)
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".safetensors", ".bin"}
    )
    total_bytes = sum(path.stat().st_size for path in files)
    completed_bytes = 0
    next_report = 0.0
    LOGGER.info("Fingerprinting %s (%d files, %s)", root, len(files), fmt_bytes(total_bytes))
    for index, path in enumerate(files, start=1):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)

        def report(chunk_size):
            nonlocal completed_bytes, next_report
            completed_bytes += chunk_size
            percent = 100.0 * completed_bytes / max(total_bytes, 1)
            if percent >= next_report:
                LOGGER.info(
                    "Fingerprint %s: %5.1f%% (%s/%s), file %d/%d",
                    root.name,
                    percent,
                    fmt_bytes(completed_bytes),
                    fmt_bytes(total_bytes),
                    index,
                    len(files),
                )
                next_report = math.floor(percent) + 1.0

        digest.update(sha256_file(str(path), progress=report).encode("ascii"))
    LOGGER.info("Fingerprint complete: %s", root)
    return digest.hexdigest()


def atomic_save_safetensors(tensors: dict, path: str) -> str:
    final_path = Path(path)
    ensure_dir(str(final_path.parent))
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    cpu_tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
    save_file(cpu_tensors, str(tmp_path))
    os.replace(str(tmp_path), str(final_path))
    return sha256_file(str(final_path))


def free_space_gb(path: str) -> float:
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(str(probe))
    return usage.free / float(1024 ** 3)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def _load_index_json(index_path: Path, model_dir: Path) -> Dict[str, str]:
    with open(index_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {name: str(model_dir / rel_path) for name, rel_path in data["weight_map"].items()}


def load_tensor_index(model_dir: str) -> dict[str, str]:
    model_path = relocated_path(model_dir)
    safetensors_index = model_path / "model.safetensors.index.json"
    if safetensors_index.exists():
        return _load_index_json(safetensors_index, model_path)

    bin_index = model_path / "pytorch_model.bin.index.json"
    if bin_index.exists():
        return _load_index_json(bin_index, model_path)

    safetensors_files = sorted(model_path.glob("*.safetensors"))
    if len(safetensors_files) == 1:
        with safe_open(str(safetensors_files[0]), framework="pt", device="cpu") as handle:
            return {name: str(safetensors_files[0]) for name in handle.keys()}

    bin_files = sorted(model_path.glob("pytorch_model*.bin"))
    if len(bin_files) == 1:
        state = torch.load(str(bin_files[0]), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint format: {bin_files[0]}")
        return {name: str(bin_files[0]) for name in state.keys()}

    raise FileNotFoundError(f"No tensor index found in {model_dir}")


def _iter_file_groups(tensor_index: Dict[str, str], tensor_names: List[str]) -> Iterator[Tuple[str, List[str]]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for name in tensor_names:
        grouped[tensor_index[name]].append(name)
    for path in sorted(grouped):
        yield path, sorted(grouped[path])


def iter_tensors(model_dir: str, tensor_names: list[str]):
    tensor_index = load_tensor_index(model_dir)
    for file_path, names in _iter_file_groups(tensor_index, tensor_names):
        path = Path(file_path)
        if path.suffix.lower() == ".safetensors":
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for name in names:
                    yield name, handle.get_tensor(name).to(torch.float32)
        else:
            state = torch.load(str(path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            for name in names:
                yield name, state[name].to(torch.float32)


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    units = ["KB", "MB", "GB", "TB", "PB"]
    value = float(n)
    for unit in units:
        value /= 1024.0
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} PB"


def model_tensor_files(model_dir: str) -> List[Path]:
    root = Path(model_dir)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".safetensors", ".bin"}
    )


def model_size_bytes(model_dir: str) -> int:
    return sum(path.stat().st_size for path in model_tensor_files(model_dir))


def tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()
