"""Recipe-independent, resumable block-local Gram atlas."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .manifest import load_manifest
from .paths import relocated_path
from .recipes import load_analysis
from .tensor_stream import iter_flat_ranges, read_flat_slice
from .utils import sha256_file


ATLAS_SCHEMA_VERSION = "1.0"
DEFAULT_ATLAS_BLOCK_SIZE = 262_144
DEFAULT_CHECKPOINT_BLOCKS = 32
DEFAULT_GRAM_CHUNK_SIZE = 65_536


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def _model_key(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._") or "model"
    return f"{safe[:80]}-{hashlib.sha256(label.encode('utf-8')).hexdigest()[:8]}"


def _models_and_shared(analysis: dict) -> tuple[list[dict], list[str]]:
    labels = [str(value) for value in analysis["labels"]]
    delta_dirs = [os.path.abspath(str(value)) for value in analysis["delta_dirs"]]
    if len(labels) != len(delta_dirs) or len(set(labels)) != len(labels):
        raise ValueError("Analysis labels and delta directories are inconsistent")
    models = []
    shared = None
    shapes: dict[str, tuple[int, ...]] = {}
    for label, delta_dir in zip(labels, delta_dirs):
        manifest = load_manifest(delta_dir)
        if manifest is None:
            raise ValueError(f"Missing delta manifest: {delta_dir}")
        names = set(manifest.tensors)
        shared = names if shared is None else shared & names
        models.append({"label": label, "delta_dir": delta_dir, "manifest": manifest})
    shared_names = sorted(shared or set())
    for name in list(shared_names):
        current = {
            tuple(model["manifest"].tensors[name].shape) for model in models
        }
        if len(current) != 1:
            shared_names.remove(name)
        else:
            shapes[name] = next(iter(current))
    if not shared_names:
        raise ValueError("No shape-compatible tensors are shared by all models")
    return models, shared_names


def atlas_identity(analysis: dict, block_size: int) -> dict:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    models, shared_names = _models_and_shared(analysis)
    model_records = []
    for model in models:
        manifest = model["manifest"]
        shards = []
        for filename, record in sorted(manifest.shards.items()):
            path = relocated_path(model["delta_dir"]) / filename
            if not path.exists():
                raise ValueError(f"Missing delta shard: {path}")
            stat = path.stat()
            shards.append(
                {
                    "name": filename,
                    "sha256": record.file_sha256 or sha256_file(str(path)),
                    "size": stat.st_size,
                }
            )
        tensor_digest = _canonical_hash(
            {
                name: {
                    "shape": record.shape,
                    "shard": record.shard_file,
                    "delta_sha256": record.delta_sha256,
                }
                for name, record in sorted(manifest.tensors.items())
            }
        )
        model_records.append(
            {
                "label": model["label"],
                "delta_dir": model["delta_dir"],
                "base_fingerprint": manifest.base_fingerprint,
                "target_fingerprint": manifest.target_fingerprint,
                "tensor_manifest_sha256": tensor_digest,
                "shards": shards,
            }
        )
    identity = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "block_size": int(block_size),
        "labels": [model["label"] for model in models],
        "shared_tensors": shared_names,
        "models": model_records,
    }
    identity["identity_hash"] = _canonical_hash(identity)
    return identity


def _load_checkpoint(
    path: Path,
    identity_hash: str,
    tensor_name: str,
    shape: tuple[int, ...],
    block_count: int,
    model_count: int,
) -> tuple[np.ndarray, int] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if (
                str(data["identity_hash"].item()) != identity_hash
                or str(data["tensor_name"].item()) != tensor_name
                or tuple(int(value) for value in data["shape"]) != shape
            ):
                raise ValueError(f"Atlas checkpoint identity mismatch for {tensor_name}")
            gram = np.array(data["gram"], dtype=np.float64, copy=True)
            completed = int(data["completed_blocks"].item())
            if gram.shape != (block_count, model_count, model_count):
                return None
            if not 0 <= completed <= block_count:
                return None
            if not np.all(np.isfinite(gram[:completed])):
                return None
            return gram, completed
    except ValueError as error:
        if "identity mismatch" in str(error):
            raise
        return None
    except Exception:
        return None


def _save_checkpoint(
    path: Path,
    identity_hash: str,
    tensor_name: str,
    shape: tuple[int, ...],
    starts: np.ndarray,
    ends: np.ndarray,
    gram: np.ndarray,
    completed_blocks: int,
) -> None:
    _atomic_npz(
        path,
        schema_version=np.array(ATLAS_SCHEMA_VERSION, dtype="U16"),
        identity_hash=np.array(identity_hash, dtype="U64"),
        tensor_name=np.array(tensor_name, dtype="U2048"),
        shape=np.asarray(shape, dtype=np.int64),
        starts=starts,
        ends=ends,
        gram=gram,
        completed_blocks=np.array(completed_blocks, dtype=np.int64),
    )


def _write_statuses(
    output: Path,
    identity_hash: str,
    models: list[dict],
    status: str,
    completed_tensors: int,
    tensor_total: int,
    completed_blocks: int,
    block_total: int,
    completed_params: int,
    total_params: int,
    energy: np.ndarray,
    tensor_name: str | None = None,
    error: str | None = None,
) -> None:
    for index, model in enumerate(models):
        payload = {
            "schema_version": ATLAS_SCHEMA_VERSION,
            "identity_hash": identity_hash,
            "status": status,
            "label": model["label"],
            "delta_dir": model["delta_dir"],
            "completed_tensors": completed_tensors,
            "tensor_total": tensor_total,
            "completed_blocks": completed_blocks,
            "block_total": block_total,
            "completed_params": completed_params,
            "total_params": total_params,
            "delta_energy_scanned": float(energy[index]),
            "current_tensor": tensor_name,
            "pid": os.getpid(),
            "updated_at": _now(),
            "error": error,
        }
        _atomic_json(
            output / "models" / _model_key(model["label"]) / "analysis_status.json",
            payload,
        )


def scan_local_atlas(
    analysis_dir: str,
    output_dir: str,
    block_size: int = DEFAULT_ATLAS_BLOCK_SIZE,
    checkpoint_blocks: int = DEFAULT_CHECKPOINT_BLOCKS,
) -> dict:
    if checkpoint_blocks <= 0:
        raise ValueError("checkpoint_blocks must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    job_path = output / "atlas_job.json"
    job_token = os.environ.get("LLMTRAITS_ATLAS_JOB_TOKEN", "")
    if job_path.exists():
        with open(job_path, "r", encoding="utf-8") as handle:
            active_job = json.load(handle)
        active_pid = int(active_job.get("pid", 0))
        active_token = str(active_job.get("job_token", ""))
        if (
            active_pid
            and active_pid != os.getpid()
            and active_token != job_token
            and _pid_alive(active_pid)
        ):
            raise RuntimeError(f"Atlas scan is already running with PID {active_pid}")
    _atomic_json(
        job_path,
        {
            "pid": os.getpid(),
            "analysis_dir": os.path.abspath(analysis_dir),
            "output_dir": os.path.abspath(output_dir),
            "block_size": int(block_size),
            "mode": "scan",
            "job_token": job_token,
            "started_at": _now(),
        },
    )
    analysis = load_analysis(analysis_dir)
    models, shared_names = _models_and_shared(analysis)
    tensors_dir = output / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    tensor_shapes = {
        name: tuple(models[0]["manifest"].tensors[name].shape)
        for name in shared_names
    }
    tensor_blocks = {
        name: len(list(iter_flat_ranges(tensor_shapes[name], block_size)))
        for name in shared_names
    }
    tensor_params = {
        name: int(np.prod(tensor_shapes[name], dtype=np.int64))
        for name in shared_names
    }
    total_blocks = sum(tensor_blocks.values())
    total_params = sum(tensor_params.values())
    model_count = len(models)
    try:
        identity = atlas_identity(analysis, block_size)
        identity_hash = identity["identity_hash"]
        identity_path = output / "atlas_identity.json"
        if identity_path.exists():
            with open(identity_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("identity_hash") != identity_hash:
                raise ValueError("Atlas identity mismatch; use a new output directory")
        _atomic_json(identity_path, identity)
    except Exception as error:
        failed_hash = locals().get("identity_hash", "")
        _atomic_json(
            output / "atlas_progress.json",
            {
                "schema_version": ATLAS_SCHEMA_VERSION,
                "identity_hash": failed_hash,
                "status": "failed",
                "phase": "preflight",
                "completed_tensors": 0,
                "tensor_total": len(shared_names),
                "completed_blocks": 0,
                "block_total": total_blocks,
                "completed_params": 0,
                "total_params": total_params,
                "pid": os.getpid(),
                "updated_at": _now(),
                "error": str(error),
            },
        )
        _write_statuses(
            output,
            failed_hash,
            models,
            "failed",
            0,
            len(shared_names),
            0,
            total_blocks,
            0,
            total_params,
            np.zeros(model_count, dtype=np.float64),
            error=str(error),
        )
        raise

    energy = np.zeros(model_count, dtype=np.float64)
    completed_tensors = 0
    completed_blocks = 0
    completed_params = 0
    records = {}

    def write_progress(
        status: str,
        tensor_name=None,
        error=None,
        block_increment=0,
        param_increment=0,
        energy_increment=None,
    ):
        reported_energy = energy
        if energy_increment is not None:
            reported_energy = energy + energy_increment
        payload = {
            "schema_version": ATLAS_SCHEMA_VERSION,
            "identity_hash": identity_hash,
            "status": status,
            "phase": "scan",
            "completed_tensors": completed_tensors,
            "tensor_total": len(shared_names),
            "completed_blocks": completed_blocks + block_increment,
            "block_total": total_blocks,
            "completed_params": completed_params + param_increment,
            "total_params": total_params,
            "current_tensor": tensor_name,
            "pid": os.getpid(),
            "updated_at": _now(),
            "error": error,
        }
        _atomic_json(output / "atlas_progress.json", payload)
        _write_statuses(
            output,
            identity_hash,
            models,
            status,
            completed_tensors,
            len(shared_names),
            completed_blocks + block_increment,
            total_blocks,
            completed_params + param_increment,
            total_params,
            reported_energy,
            tensor_name,
            error,
        )

    try:
        write_progress("running")
        for name in shared_names:
            shape = tensor_shapes[name]
            ranges = list(iter_flat_ranges(shape, block_size))
            starts = np.asarray([start for start, _ in ranges], dtype=np.int64)
            ends = np.asarray([end for _, end in ranges], dtype=np.int64)
            path = tensors_dir / f"{_tensor_key(name)}.npz"
            loaded = _load_checkpoint(
                path, identity_hash, name, shape, len(ranges), model_count
            )
            if loaded is None:
                gram = np.zeros(
                    (len(ranges), model_count, model_count), dtype=np.float64
                )
                local_completed = 0
            else:
                gram, local_completed = loaded

            if local_completed < len(ranges):
                while local_completed < len(ranges):
                    batch_end = min(
                        local_completed + checkpoint_blocks, len(ranges)
                    )
                    batch_start_param = ranges[local_completed][0]
                    batch_end_param = ranges[batch_end - 1][1]
                    values = np.empty(
                        (model_count, batch_end_param - batch_start_param),
                        dtype=np.float32,
                    )

                    def read_model_range(index_and_model):
                        model_index, model = index_and_model
                        record = model["manifest"].tensors[name]
                        with safe_open(
                            str(
                                relocated_path(model["delta_dir"])
                                / record.shard_file
                            ),
                            framework="pt",
                            device="cpu",
                        ) as handle:
                            data = read_flat_slice(
                                handle,
                                name,
                                shape,
                                batch_start_param,
                                batch_end_param,
                            )
                        return model_index, data

                    with ThreadPoolExecutor(
                        max_workers=min(3, model_count)
                    ) as io_pool:
                        for model_index, data in io_pool.map(
                            read_model_range, enumerate(models)
                        ):
                            values[model_index] = data
                    for block_index in range(local_completed, batch_end):
                        start, end = ranges[block_index]
                        local_start = start - batch_start_param
                        local_end = end - batch_start_param
                        block64 = values[:, local_start:local_end].astype(
                            np.float64
                        )
                        local_gram = block64 @ block64.T
                        gram[block_index] = local_gram
                        local_completed = block_index + 1
                        if (
                            local_completed % checkpoint_blocks == 0
                            or local_completed == len(ranges)
                        ):
                            _save_checkpoint(
                                path,
                                identity_hash,
                                name,
                                shape,
                                starts,
                                ends,
                                gram,
                                local_completed,
                            )
                            prefix_energy = np.diagonal(
                                gram[:local_completed], axis1=1, axis2=2
                            ).sum(axis=0)
                            write_progress(
                                "running",
                                name,
                                block_increment=local_completed,
                                param_increment=int(ends[local_completed - 1]),
                                energy_increment=prefix_energy,
                            )
                    del values

            local_energy = np.diagonal(gram, axis1=1, axis2=2).sum(axis=0)
            energy += local_energy
            completed_tensors += 1
            completed_blocks += len(ranges)
            completed_params += tensor_params[name]
            records[name] = {
                "file": str(path.relative_to(output)),
                "sha256": sha256_file(str(path)),
                "shape": list(shape),
                "blocks": len(ranges),
                "parameters": tensor_params[name],
            }
            write_progress("running", name)

        atlas_index = {
            "schema_version": ATLAS_SCHEMA_VERSION,
            "identity_hash": identity_hash,
            "analysis_dir": os.path.abspath(analysis_dir),
            "block_size": int(block_size),
            "labels": [model["label"] for model in models],
            "tensor_count": len(shared_names),
            "block_count": total_blocks,
            "parameter_count": total_params,
            "model_energy": {
                model["label"]: float(energy[index])
                for index, model in enumerate(models)
            },
            "tensors": records,
            "completed_at": _now(),
        }
        _atomic_json(output / "atlas_index.json", atlas_index)
        write_progress("complete")
        return atlas_index
    except KeyboardInterrupt:
        write_progress("stopped", error="Interrupted; rerun the same command to resume")
        raise
    except Exception as error:
        write_progress("failed", error=str(error))
        raise


def _pid_alive(pid: int) -> bool:
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not process:
        return False
    exit_code = ctypes.c_ulong()
    try:
        if not ctypes.windll.kernel32.GetExitCodeProcess(
            process, ctypes.byref(exit_code)
        ):
            return False
        return exit_code.value == 259
    finally:
        ctypes.windll.kernel32.CloseHandle(process)


def launch_local_atlas(
    analysis_dir: str,
    output_dir: str,
    block_size: int = DEFAULT_ATLAS_BLOCK_SIZE,
    checkpoint_blocks: int = DEFAULT_CHECKPOINT_BLOCKS,
) -> tuple[int, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    job_path = output / "atlas_job.json"
    if job_path.exists():
        with open(job_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        pid = int(previous.get("pid", 0))
        if pid and _pid_alive(pid):
            raise RuntimeError(f"Atlas scan is already running with PID {pid}")
    progress_path = output / "atlas_progress.json"
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as handle:
            if json.load(handle).get("status") == "complete":
                raise RuntimeError("Atlas scan is already complete")

    log_path = output / "atlas_worker.log"
    job_token = secrets.token_hex(16)
    argv = [
        sys.executable,
        "-m",
        "llmtraits",
        "atlas-scan",
        "--analysis",
        os.path.abspath(analysis_dir),
        "--output-dir",
        os.path.abspath(output_dir),
        "--block-size",
        str(block_size),
        "--checkpoint-blocks",
        str(checkpoint_blocks),
    ]
    _atomic_json(
        job_path,
        {
            "pid": 0,
            "job_token": job_token,
            "analysis_dir": os.path.abspath(analysis_dir),
            "output_dir": os.path.abspath(output_dir),
            "block_size": int(block_size),
            "checkpoint_blocks": int(checkpoint_blocks),
            "mode": "launching",
            "launched_at": _now(),
        },
    )
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n=== Atlas launch {_now()} ===\nargv: {argv}\n")
        log.flush()
        kwargs = (
            {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
            }
            if sys.platform == "win32"
            else {"start_new_session": True}
        )
        worker_env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "LLMTRAITS_ATLAS_JOB_TOKEN": job_token,
        }
        process = subprocess.Popen(
            argv,
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=worker_env,
            **kwargs,
        )
    worker_pid = process.pid
    for _ in range(100):
        time.sleep(0.1)
        try:
            with open(job_path, "r", encoding="utf-8") as handle:
                active = json.load(handle)
            if (
                active.get("job_token") == job_token
                and int(active.get("pid", 0)) > 0
                and active.get("mode") == "scan"
            ):
                worker_pid = int(active["pid"])
                break
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    _atomic_json(
        output / "atlas_launch.json",
        {
            "pid": worker_pid,
            "launcher_pid": process.pid,
            "job_token": job_token,
            "analysis_dir": os.path.abspath(analysis_dir),
            "output_dir": os.path.abspath(output_dir),
            "block_size": int(block_size),
            "checkpoint_blocks": int(checkpoint_blocks),
            "log_path": str(log_path),
            "launched_at": _now(),
        },
    )
    return worker_pid, str(log_path)
