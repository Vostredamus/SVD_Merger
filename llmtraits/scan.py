import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_OUTPUT, KNOWN_DUPLICATES, SCAN_DIRS
from .utils import ensure_dir, model_tensor_files, sha256_model_dir

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    path: str
    name: str
    is_valid: bool
    n_shards: int
    vocab_size: int | None
    fingerprint: str | None = None


def _read_vocab_size(model_dir: Path) -> int | None:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("vocab_size")
    except Exception:
        return None


def _is_valid_model(model_dir: Path) -> tuple[bool, int]:
    has_config = (model_dir / "config.json").exists()
    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    bin_files = sorted(model_dir.glob("pytorch_model*.bin"))
    n_shards = len(safetensors_files) or len(bin_files)
    return has_config and n_shards > 0, n_shards


def find_models(scan_dirs: list[str], skip_paths: set[str] = None) -> list[ModelInfo]:
    skip_paths = {os.path.abspath(path) for path in (skip_paths or set())}
    models: list[ModelInfo] = []
    seen_duplicate_names: set[str] = set()
    for scan_dir in scan_dirs:
        root = Path(scan_dir)
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if not any(child.iterdir()):
                continue
            child_path = os.path.abspath(str(child))
            if child_path in skip_paths:
                continue
            if child.name in KNOWN_DUPLICATES:
                if child.name in seen_duplicate_names:
                    LOGGER.info("Skipping known duplicate %s", child_path)
                    continue
                seen_duplicate_names.add(child.name)
            is_valid, n_shards = _is_valid_model(child)
            models.append(
                ModelInfo(
                    path=child_path,
                    name=child.name,
                    is_valid=is_valid,
                    n_shards=n_shards,
                    vocab_size=_read_vocab_size(child),
                )
            )
    return models


def _fingerprint_cache_path() -> Path:
    return Path(DEFAULT_OUTPUT) / ".fingerprint_cache.json"


def _load_cache() -> dict:
    path = _fingerprint_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _save_cache(cache: dict):
    path = _fingerprint_cache_path()
    ensure_dir(str(path.parent))
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
    os.replace(str(tmp_path), str(path))


def _dir_signature(path: str) -> tuple[list[dict], int]:
    root = Path(path)
    files = model_tensor_files(path)
    entries = []
    total_size = 0
    for file_path in files:
        stat = file_path.stat()
        total_size += stat.st_size
        entries.append(
            {
                "path": file_path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        )
    return entries, total_size


def _fingerprint_for(model: ModelInfo, cache: dict) -> tuple[str, int]:
    files, total_size = _dir_signature(model.path)
    cached = cache.get(model.path)
    if cached and cached.get("files") == files and cached.get("size") == total_size:
        model.fingerprint = cached["fingerprint"]
        return model.fingerprint, total_size
    fingerprint = sha256_model_dir(model.path)
    cache[model.path] = {"fingerprint": fingerprint, "files": files, "size": total_size}
    model.fingerprint = fingerprint
    return fingerprint, total_size


def cached_model_fingerprint(model_path: str) -> str:
    model_path = os.path.abspath(model_path)
    is_valid, n_shards = _is_valid_model(Path(model_path))
    if not is_valid:
        raise ValueError(f"Incomplete model checkpoint: {model_path}")
    model = ModelInfo(
        path=model_path,
        name=Path(model_path).name,
        is_valid=True,
        n_shards=n_shards,
        vocab_size=_read_vocab_size(Path(model_path)),
    )
    cache = _load_cache()
    fingerprint, _ = _fingerprint_for(model, cache)
    _save_cache(cache)
    return fingerprint


def dedup_models(models: list[ModelInfo]) -> tuple[list[ModelInfo], list[tuple[ModelInfo, ModelInfo]]]:
    cache = _load_cache()
    fingerprint_map: dict[tuple[int, int, str], ModelInfo] = {}
    uniques: list[ModelInfo] = []
    duplicates: list[tuple[ModelInfo, ModelInfo]] = []
    for model in models:
        if not model.is_valid:
            uniques.append(model)
            continue
        fingerprint, total_size = _fingerprint_for(model, cache)
        _save_cache(cache)
        key = (model.n_shards, total_size, fingerprint)
        if key in fingerprint_map:
            duplicates.append((fingerprint_map[key], model))
        else:
            fingerprint_map[key] = model
            uniques.append(model)
    return uniques, duplicates


def scan_command(args) -> None:
    scan_dirs = args.scan_dirs or SCAN_DIRS
    base = os.path.abspath(args.base) if getattr(args, "base", None) else None
    models = find_models(scan_dirs, skip_paths={base} if base else None)
    uniques, duplicates = dedup_models(models)
    for model in uniques:
        status = "valid" if model.is_valid else "incomplete"
        print(f"{status:10} {model.n_shards:3d} {model.name}  {model.path}")
    if duplicates:
        print("\nDuplicates:")
        for kept, duplicate in duplicates:
            print(f"{duplicate.path} == {kept.path}")
