"""
Compose + ExLlamaV3 compression workflow.

Job state lifecycle:
  queued → preflight → composing → compressing → completed
                                               ↘ failed
                                               ↘ stopped

Each job writes its state to:
  <work_path>/job_state.json          (atomic writes, per-job)
  <jobs_root>/<output_name>/          (recipe, config, worker.log)

Never delete the full composed model automatically. Never start conversion
unless preflight succeeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ─── Constants ────────────────────────────────────────────────────────────────

JOB_SCHEMA_VERSION = "1.0"

# All of these point at your own ExLlamaV3 checkout and scratch directories.
# Set the environment variables below, or fill in the matching fields in the
# GUI's "Compression job" panel / a CompressionConfig instance directly --
# nothing here needs to be correct until you actually run a compression job.
DEFAULT_CONVERTER_PYTHON = os.environ.get("LLMTRAITS_CONVERTER_PYTHON", "")
DEFAULT_CONVERT_SCRIPT = os.environ.get("LLMTRAITS_CONVERT_SCRIPT", "")
DEFAULT_COMPOSED_ROOT = os.environ.get("LLMTRAITS_COMPOSED_ROOT", "./composed")
DEFAULT_COMPRESSED_ROOT = os.environ.get("LLMTRAITS_COMPRESSED_ROOT", "./compressed_models")
DEFAULT_WORK_ROOT = os.environ.get("LLMTRAITS_WORK_ROOT", "./compression_work")
DEFAULT_JOBS_ROOT = os.environ.get("LLMTRAITS_JOBS_ROOT", "./composition_jobs")
DEFAULT_BPW = 5.5
DEFAULT_CUDA_DEVICES = "0"
_PREFLIGHT_MARKER = "LLMTRAITS_PREFLIGHT_JSON="

BPW_MIN = 1.0
BPW_MAX = 8.0

# Minimum size of a real model tensor file (not a placeholder)
_MODEL_FILE_MIN_BYTES = 1024


def _converter_env(config: "CompressionConfig") -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = config.cuda_devices
    scripts_dir = str(Path(config.converter_python).parent)
    env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")
    return env

# Reserved Windows device names (forbidden as folder names)
_WINDOWS_RESERVED = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})

# ExLlamaV3 args.json identity-field aliases (canonical key first in each tuple)
_BITS_ALIASES = ("bits", "bpw")       # ExLlamaV3 writes "bits"; "bpw" is an alias
_INPUT_ALIASES = ("in_dir", "input_dir")
_OUTPUT_ALIASES = ("out_dir", "output_dir")


# ─── Naming helpers ───────────────────────────────────────────────────────────

def sanitize_folder_name(name: str) -> str:
    """
    Return a safe Windows single-segment folder name.

    Replaces illegal characters, dot-only names (. and ..), reserved Windows
    device names, strips trailing dots and spaces, and handles empty inputs.
    """
    if not name:
        return "_"
    # Replace illegal Windows filename characters
    result = re.sub(r'[\\/:*?"<>|]', "_", name)
    # Strip trailing dots and spaces (Windows forbids them)
    result = result.rstrip(". ")
    # Strip leading spaces
    result = result.lstrip(" ")
    if not result:
        return "_"
    # Replace dot-only names (. and .. are special directory entries)
    if re.fullmatch(r"\.+", result):
        return "_"
    # Prefix reserved Windows device names (e.g. CON, NUL, COM1, LPT1)
    base_part = result.split(".")[0].upper()
    if base_part in _WINDOWS_RESERVED:
        result = f"_{result}"
    return result


def format_bpw(bpw: float) -> str:
    """Format BPW for display and suffix use: 5.5→'5.5', 4.0→'4'."""
    s = f"{bpw:.2f}".rstrip("0").rstrip(".")
    return s


def default_output_name(model_name: str, bpw: float) -> str:
    """Build the default compressed-folder name: <model>_<bpw>bw."""
    return f"{sanitize_folder_name(model_name)}_{format_bpw(bpw)}bw"


# ─── Config ───────────────────────────────────────────────────────────────────

class CompressionConfig:
    """All parameters for a single compose+compress job. Serialisable to JSON."""

    SCHEMA = "compression_job"
    VERSION = "1.0"

    def __init__(
        self,
        *,
        recipe_path: str,
        model_name: str,
        output_name: str,
        composed_path: str,
        compressed_path: str,
        work_path: str,
        jobs_root: str = DEFAULT_JOBS_ROOT,
        converter_python: str = DEFAULT_CONVERTER_PYTHON,
        convert_script: str = DEFAULT_CONVERT_SCRIPT,
        bpw: float = DEFAULT_BPW,
        cuda_devices: str = DEFAULT_CUDA_DEVICES,
    ) -> None:
        self.recipe_path = recipe_path
        self.model_name = model_name
        self.output_name = output_name
        self.composed_path = composed_path
        self.compressed_path = compressed_path
        self.work_path = work_path
        self.jobs_root = jobs_root
        self.converter_python = converter_python
        self.convert_script = convert_script
        self.bpw = float(bpw)
        self.cuda_devices = cuda_devices

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema": self.SCHEMA,
            "version": self.VERSION,
            "recipe_path": self.recipe_path,
            "model_name": self.model_name,
            "output_name": self.output_name,
            "composed_path": self.composed_path,
            "compressed_path": self.compressed_path,
            "work_path": self.work_path,
            "jobs_root": self.jobs_root,
            "converter_python": self.converter_python,
            "convert_script": self.convert_script,
            "bpw": self.bpw,
            "cuda_devices": self.cuda_devices,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompressionConfig":
        if data.get("schema") != cls.SCHEMA:
            raise ValueError(
                f"Expected schema {cls.SCHEMA!r}, got {data.get('schema')!r}"
            )
        if data.get("version") != cls.VERSION:
            raise ValueError(
                f"Expected version {cls.VERSION!r}, got {data.get('version')!r}"
            )
        return cls(
            recipe_path=data["recipe_path"],
            model_name=data["model_name"],
            output_name=data["output_name"],
            composed_path=data["composed_path"],
            compressed_path=data["compressed_path"],
            work_path=data["work_path"],
            jobs_root=data.get("jobs_root", DEFAULT_JOBS_ROOT),
            converter_python=data.get("converter_python", DEFAULT_CONVERTER_PYTHON),
            convert_script=data.get("convert_script", DEFAULT_CONVERT_SCRIPT),
            bpw=float(data.get("bpw", DEFAULT_BPW)),
            cuda_devices=data.get("cuda_devices", DEFAULT_CUDA_DEVICES),
        )

    @classmethod
    def from_file(cls, path: str) -> "CompressionConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        _atomic_json_write(path, self.to_dict())

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty list = valid)."""
        errors: list[str] = []
        if not self.recipe_path:
            errors.append("recipe_path is required")
        elif not Path(self.recipe_path).exists():
            errors.append(f"Recipe file not found: {self.recipe_path}")

        if not self.output_name or not self.output_name.strip():
            errors.append("output_name must not be empty")
        elif sanitize_folder_name(self.output_name) != self.output_name:
            errors.append(
                f"output_name {self.output_name!r} is not a safe single-segment folder name; "
                f"use: {sanitize_folder_name(self.output_name)!r}"
            )

        if not (BPW_MIN <= self.bpw <= BPW_MAX):
            errors.append(
                f"bpw must be between {BPW_MIN} and {BPW_MAX}, got {self.bpw}"
            )

        if not self.converter_python:
            errors.append("converter_python is required")
        elif not Path(self.converter_python).is_file():
            errors.append(
                f"Converter Python executable not found: {self.converter_python}"
            )

        if not self.convert_script:
            errors.append("convert_script is required")
        elif not Path(self.convert_script).is_file():
            errors.append(f"Converter script not found: {self.convert_script}")

        if not self.composed_path:
            errors.append("composed_path is required")
        if not self.compressed_path:
            errors.append("compressed_path is required")
        if not self.work_path:
            errors.append("work_path is required")

        return errors


# ─── Atomic helpers ───────────────────────────────────────────────────────────

def _atomic_json_write(path: str, data: dict) -> None:
    """Write *data* as JSON to *path* atomically via a .tmp sibling."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    os.replace(str(tmp), str(p))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Job state ────────────────────────────────────────────────────────────────

def _state_path(work_path: str) -> Path:
    return Path(work_path) / "job_state.json"


def _load_job_state(work_path: str) -> dict:
    p = _state_path(work_path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_job_state(work_path: str, state: dict) -> None:
    Path(work_path).mkdir(parents=True, exist_ok=True)
    _atomic_json_write(str(_state_path(work_path)), state)


def _make_initial_state(config: CompressionConfig) -> dict:
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "status": "queued",
        "output_name": config.output_name,
        "composed_path": config.composed_path,
        "compressed_path": config.compressed_path,
        "work_path": config.work_path,
        "bpw": config.bpw,
        "pid": None,
        "compose_argv": None,
        "compress_argv": None,
        "queued_at": _now_iso(),
        "compose_started_at": None,
        "compose_completed_at": None,
        "compress_started_at": None,
        "compress_completed_at": None,
        "completed_at": None,
        "failed_at": None,
        "error": None,
        "output_size_bytes": None,
        "output_files": None,
    }


def _update_job_state(work_path: str, **fields) -> None:
    """Load, patch, and atomically save job state."""
    state = _load_job_state(work_path)
    state.update(fields)
    _save_job_state(work_path, state)


# ─── PID liveness ─────────────────────────────────────────────────────────────

def _check_pid_alive(pid: Optional[int]) -> bool:
    """Return True if *pid* is a live process on Windows."""
    if pid is None:
        return False
    try:
        import ctypes
        PROCESS_QUERY_INFORMATION = 0x0400
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong(0)
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return exit_code.value == STILL_ACTIVE
    except Exception:
        # Fallback for non-Windows or missing ctypes
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


# ─── Preflight ────────────────────────────────────────────────────────────────

def _infer_source_size(recipe_path: str) -> int:
    """
    Return the total safetensors bytes under the recipe's departure_path.
    Returns 0 if the recipe cannot be read or departure_path does not exist.
    """
    try:
        with open(recipe_path, "r", encoding="utf-8") as fh:
            recipe = json.load(fh)
        dep = recipe.get("departure_path", "")
        if not dep:
            return 0
        dp = Path(dep)
        if not dp.is_dir():
            return 0
        return sum(
            f.stat().st_size for f in dp.rglob("*.safetensors") if f.is_file()
        )
    except Exception:
        return 0


def run_preflight(
    config: CompressionConfig,
    timeout: int = 30,
    environment_only: bool = False,
) -> tuple[bool, str]:
    """
    Validate paths, BPW range, free space, and the converter environment.

    The converter itself is deliberately not imported or executed here.
    ExLlamaV3 imports native extensions before parsing ``--help`` and can block
    indefinitely even in a known-good environment.

    This call is intentionally safe to invoke multiple times and will never
    launch composition or conversion.
    """
    # 1. Field-level validation. The standalone environment command has no
    #    recipe or output paths, so validate only converter-related fields.
    if environment_only:
        errors = []
        if not (BPW_MIN <= config.bpw <= BPW_MAX):
            errors.append(
                f"bpw must be between {BPW_MIN} and {BPW_MAX}, got {config.bpw}"
            )
        if not config.converter_python or not Path(config.converter_python).is_file():
            errors.append(
                f"Converter Python executable not found: {config.converter_python}"
            )
        if not config.convert_script or not Path(config.convert_script).is_file():
            errors.append(f"Converter script not found: {config.convert_script}")
    else:
        errors = config.validate()
    if errors:
        return False, "Configuration errors:\n" + "\n".join(f"  • {e}" for e in errors)

    # 2. Per-drive disk space check
    #    Infer source checkpoint size from recipe departure_path (0 if unavailable).
    #    Required: composed ≥ source+10GiB, compressed ≥ src*bpw/16+10GiB, work ≥ src*bpw/16+20GiB
    _10G = 10 * 1024 ** 3
    _20G = 20 * 1024 ** 3
    source_size = 0 if environment_only else _infer_source_size(config.recipe_path)
    compressed_est = int(source_size * config.bpw / 16)

    _roles = [
        ("composed",    config.composed_path,    source_size + _10G),
        ("compressed",  config.compressed_path,  compressed_est + _10G),
        ("work",        config.work_path,         compressed_est + _20G),
    ]
    _drive_required: dict[str, int] = {}
    _drive_probe: dict[str, str] = {}
    for _label, _path, _req in _roles:
        _probe = Path(_path)
        while not _probe.exists() and _probe.parent != _probe:
            _probe = _probe.parent
        if not _probe.exists():
            continue
        _anchor = _probe.anchor.lower()
        _drive_probe.setdefault(_anchor, str(_probe))
        _drive_required[_anchor] = _drive_required.get(_anchor, 0) + _req

    for _anchor, _req in (() if environment_only else _drive_required.items()):
        try:
            _, _, _free = shutil.disk_usage(_drive_probe[_anchor])
            if _free < _req:
                return (
                    False,
                    f"Insufficient disk space on {_anchor}: "
                    f"need {_req / 1024**3:.1f} GB, "
                    f"have {_free / 1024**3:.1f} GB free",
                )
        except Exception:
            pass  # Drive not mounted yet; skip

    # 3. Run a non-invasive probe in the converter Python. Do not import
    #    ExLlamaV3 or execute convert.py: either action may initialize/compile
    #    native extensions before argument parsing.
    script_dir = str(Path(config.convert_script).parent)
    env = _converter_env(config)
    probe_code = f"""
import importlib.metadata
import importlib.util
import json
import pathlib
import shutil
import sys

result = {{}}
result["python_version"] = sys.version.split()[0]
for module_name in ("torch", "exllamav3", "ninja"):
    result[module_name + "_found"] = importlib.util.find_spec(module_name) is not None
    try:
        result[module_name + "_version"] = importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        result[module_name + "_version"] = None

converter_path = pathlib.Path(sys.argv[1])
source = converter_path.read_text(encoding="utf-8")
compile(source, str(converter_path), "exec")
result["converter_syntax"] = "ok"
result["ninja_executable"] = shutil.which("ninja")

if result["torch_found"]:
    import torch
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["gpu_count"] = int(torch.cuda.device_count())
else:
    result["cuda_available"] = False
    result["gpu_count"] = 0

print({_PREFLIGHT_MARKER!r} + json.dumps(result, sort_keys=True))
"""
    cmd = [config.converter_python, "-c", probe_code, config.convert_script]
    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (
            False,
            f"Static environment probe timed out after {timeout}s.\n"
            f"The converter itself was not imported or executed.",
        )
    except FileNotFoundError as exc:
        return False, f"Converter Python not found: {config.converter_python}\n{exc}"
    except Exception as exc:
        return False, f"Preflight subprocess error: {exc}"

    if result.returncode != 0:
        return (
            False,
            f"Static environment probe exited with code {result.returncode}.\n\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )

    marker_line = next(
        (
            line[len(_PREFLIGHT_MARKER):]
            for line in result.stdout.splitlines()
            if line.startswith(_PREFLIGHT_MARKER)
        ),
        None,
    )
    if marker_line is None:
        return (
            False,
            "Static environment probe returned no structured result.\n\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )
    try:
        details = json.loads(marker_line)
    except json.JSONDecodeError as exc:
        return False, f"Invalid static environment probe result: {exc}"

    missing = [
        name
        for name in ("torch", "exllamav3", "ninja")
        if not details.get(f"{name}_found")
    ]
    if missing:
        return False, "Missing converter modules: " + ", ".join(missing)
    if not details.get("ninja_executable"):
        return False, "Ninja module is installed, but ninja executable is not on PATH."
    if not details.get("cuda_available") or int(details.get("gpu_count", 0)) < 1:
        return (
            False,
            "CUDA is unavailable in the converter environment for "
            f"CUDA_VISIBLE_DEVICES={config.cuda_devices!r}.",
        )

    summary = (
        f"Python {details['python_version']}; "
        f"Torch {details.get('torch_version')}; "
        f"ExLlamaV3 {details.get('exllamav3_version')}; "
        f"Ninja {details.get('ninja_version')}; "
        f"{details['gpu_count']} visible GPU(s); converter syntax OK."
    )
    return (
        True,
        "Preflight OK (static checks only; converter was not imported or executed).\n"
        + summary,
    )


# ─── Converter argv builders ──────────────────────────────────────────────────

def _build_initial_convert_argv(config: CompressionConfig) -> list[str]:
    """Argv for a fresh conversion run."""
    return [
        config.converter_python,
        config.convert_script,
        "-i", config.composed_path,
        "-o", config.compressed_path,
        "-w", config.work_path,
        "-b", str(config.bpw),
        "-d", config.cuda_devices,
        "-v",
    ]


def _build_resume_convert_argv(config: CompressionConfig) -> list[str]:
    """Argv for a resumed conversion run."""
    return [
        config.converter_python,
        config.convert_script,
        "-resume",
        "-w", config.work_path,
        "-v",
    ]


# ─── Resume decision ──────────────────────────────────────────────────────────

def _should_resume_conversion(config: CompressionConfig) -> tuple[bool, str]:
    """
    Decide whether to use ``-resume`` or initial args.

    Returns ``(resume, reason)``.

    Raises ``ValueError`` if:
    - args.json is corrupt (fail closed — don't overwrite a checkpoint)
    - args.json has no recognized identity fields (unknown checkpoint)
    - args.json has partial identity (some but not all of bits/input/output)
    - args.json values conflict with the current config
    """
    args_json = Path(config.work_path) / "args.json"
    if not args_json.exists():
        return False, "No args.json in work dir; using initial args"

    # Corrupt JSON → fail closed (don't silently overwrite a checkpoint)
    try:
        with open(args_json, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
    except Exception as exc:
        raise ValueError(
            f"args.json in work dir is corrupt and cannot be parsed: {exc}\n"
            "Delete or rename the work directory to start fresh."
        ) from exc

    # Find which identity fields are present
    bits_key = next((k for k in _BITS_ALIASES if k in saved), None)
    input_key = next((k for k in _INPUT_ALIASES if k in saved), None)
    output_key = next((k for k in _OUTPUT_ALIASES if k in saved), None)

    recognized = [k for k in (bits_key, input_key, output_key) if k is not None]

    # No recognized identity fields → fail closed
    if not recognized:
        raise ValueError(
            "args.json contains no recognized identity fields "
            f"(expected any of: {_BITS_ALIASES + _INPUT_ALIASES + _OUTPUT_ALIASES}).\n"
            "Cannot determine whether this checkpoint belongs to the current job.\n"
            "Delete or rename the work directory to start fresh."
        )

    # Partial identity → fail closed
    missing_roles: list[str] = []
    if bits_key is None:
        missing_roles.append("/".join(_BITS_ALIASES))
    if input_key is None:
        missing_roles.append("/".join(_INPUT_ALIASES))
    if output_key is None:
        missing_roles.append("/".join(_OUTPUT_ALIASES))

    if missing_roles:
        raise ValueError(
            f"args.json has partial identity: found {recognized!r}, "
            f"missing {missing_roles!r}.\n"
            "Cannot safely verify this checkpoint belongs to the current job.\n"
            "Delete or rename the work directory to start fresh."
        )

    # Validate each field and check for mismatches
    mismatches: list[str] = []

    # bits / bpw
    try:
        saved_bits = float(saved[bits_key])
        if abs(saved_bits - config.bpw) > 1e-6:
            mismatches.append(f"bpw: saved={saved[bits_key]!r}, requested={config.bpw!r}")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"args.json has invalid {bits_key!r} value {saved[bits_key]!r}: {exc}"
        ) from exc

    # input dir
    saved_in_norm = os.path.normcase(os.path.normpath(str(saved[input_key])))
    cfg_in_norm = os.path.normcase(os.path.normpath(config.composed_path))
    if saved_in_norm != cfg_in_norm:
        mismatches.append(
            f"{input_key}: saved={saved[input_key]!r}, requested={config.composed_path!r}"
        )

    # output dir
    saved_out_norm = os.path.normcase(os.path.normpath(str(saved[output_key])))
    cfg_out_norm = os.path.normcase(os.path.normpath(config.compressed_path))
    if saved_out_norm != cfg_out_norm:
        mismatches.append(
            f"{output_key}: saved={saved[output_key]!r}, requested={config.compressed_path!r}"
        )

    if mismatches:
        raise ValueError(
            "Work directory args.json does not match the current job config.\n"
            "Delete or rename the work directory to start fresh, "
            "or fix the config so it matches the checkpoint.\n"
            "Mismatches:\n" + "\n".join(f"  {m}" for m in mismatches)
        )

    return True, "args.json found and matches current config; using -resume"


# ─── Output validation ────────────────────────────────────────────────────────

def _validate_converter_output(compressed_path: str) -> tuple[bool, str, dict]:
    """
    Verify converter produced a real EXL3-quantised model directory.

    Checks:
    - config.json present and parseable
    - config.json has ``quantization_config.quant_method == "exl3"``
    - quantization_config.json present and parseable
    - tokenizer.json present
    - At least one substantive model safetensors file (≥ ``_MODEL_FILE_MIN_BYTES``)

    A plain HF safetensors directory (no EXL3 quant config) will fail.
    Returns ``(ok, message, stats_dict)``.
    """
    p = Path(compressed_path)
    if not p.exists():
        return False, f"Compressed output directory does not exist: {compressed_path}", {}

    all_files = [f for f in p.rglob("*") if f.is_file()]
    lower_map = {f.name.lower(): f for f in all_files}

    # config.json must be present and parseable
    if "config.json" not in lower_map:
        return False, f"config.json missing from {compressed_path}", {}
    try:
        with open(lower_map["config.json"], "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as exc:
        return False, f"config.json is not parseable: {exc}", {}

    # quantization_config.quant_method must be exactly "exl3"
    qcfg = cfg.get("quantization_config") or {}
    quant_method = qcfg.get("quant_method", "")
    if quant_method != "exl3":
        return (
            False,
            f"config.json quantization_config.quant_method is {quant_method!r}, "
            f"expected 'exl3'. This does not appear to be an ExLlamaV3 output directory.",
            {},
        )

    # quantization_config.json must be present and parseable
    if "quantization_config.json" not in lower_map:
        return False, f"quantization_config.json missing from {compressed_path}", {}
    try:
        with open(lower_map["quantization_config.json"], "r", encoding="utf-8") as fh:
            json.load(fh)
    except Exception as exc:
        return False, f"quantization_config.json is not parseable: {exc}", {}

    # tokenizer metadata
    if "tokenizer.json" not in lower_map:
        return False, f"tokenizer.json missing from {compressed_path}", {}

    # At least one substantive model safetensors shard
    model_files = [
        f for f in all_files
        if f.suffix.lower() == ".safetensors"
        and f.stat().st_size >= _MODEL_FILE_MIN_BYTES
    ]
    if not model_files:
        return (
            False,
            f"No substantive model safetensors files (≥{_MODEL_FILE_MIN_BYTES} B) "
            f"found in {compressed_path}",
            {},
        )

    total_bytes = sum(f.stat().st_size for f in all_files)
    file_list = sorted(str(f.relative_to(p)) for f in all_files)

    return (
        True,
        "Output validated OK",
        {"output_size_bytes": total_bytes, "output_files": file_list[:500]},
    )


# ─── Composition wrapper (thin layer for testability) ─────────────────────────

def _compose_in_worker(recipe_path: str) -> None:
    """
    Call compose_svd from within the worker process.

    Isolated in its own function so tests can monkeypatch it without
    importing recipes at module load time.
    """
    from .recipes import compose_svd  # noqa: PLC0415
    compose_svd(recipe_path)


# ─── Composition completion validator ─────────────────────────────────────────

def _validate_composition_complete(
    config: "CompressionConfig",
    recipe: dict,
) -> tuple[bool, str]:
    """
    Verify that the composed output at ``config.composed_path`` is structurally
    complete **and** belongs to the current recipe.

    Returns ``(complete, reason)``.

    - ``complete=False`` → composition is not done; the caller should run it.
    - ``complete=True``  → composition is verified for this job; skip it.

    Raises ``ValueError`` on identity mismatch (different recipe or departure
    path), to prevent silently reusing or overwriting a different job's output.
    """
    composed = Path(config.composed_path)
    resolved_json = composed / "resolved_recipe.json"
    provenance_json = composed / "provenance.json"

    # Neither file → composition has not started
    if not resolved_json.exists() and not provenance_json.exists():
        return False, "No resolved_recipe.json or provenance.json; composition not started"

    # resolved_recipe.json missing while provenance.json exists → unexpected state
    if not resolved_json.exists():
        return False, "resolved_recipe.json missing; composition incomplete or inconsistent"

    # Parse resolved_recipe.json
    try:
        with open(resolved_json, "r", encoding="utf-8") as fh:
            resolved = json.load(fh)
    except Exception as exc:
        return False, f"resolved_recipe.json unreadable: {exc}"

    # Identity check: current recipe vs saved original_recipe (excluding transient cache)
    original = resolved.get("original_recipe", {})
    current = {k: v for k, v in recipe.items() if k != "_analysis_cache"}
    if original != current:
        raise ValueError(
            f"Composed directory {config.composed_path!r} contains a resolved_recipe.json "
            f"that does not match the current recipe. "
            f"This may be a different job's output.\n"
            f"Delete or rename the composed directory before proceeding."
        )

    # provenance.json missing → compose is in progress or failed after resolved was written
    if not provenance_json.exists():
        return False, "provenance.json missing; composition in progress or failed mid-run"

    # Parse provenance.json
    try:
        with open(provenance_json, "r", encoding="utf-8") as fh:
            provenance = json.load(fh)
    except Exception as exc:
        return False, f"provenance.json unreadable: {exc}"

    # Identity check: departure_path
    dep_path = recipe.get("departure_path", "")
    expected_base = os.path.normcase(os.path.normpath(os.path.abspath(dep_path)))
    saved_base = os.path.normcase(
        os.path.normpath(os.path.abspath(provenance.get("base_dir", "")))
    )
    if expected_base != saved_base:
        raise ValueError(
            f"provenance.base_dir {provenance.get('base_dir')!r} does not match "
            f"recipe departure_path {dep_path!r}. "
            f"Identity mismatch — delete the composed directory."
        )

    # dtype and missing_policy must match
    expected_dtype = recipe.get("dtype", "float16")
    expected_missing = recipe.get("missing_policy", "error")
    prov_dtype = provenance.get("dtype", "float16")
    prov_missing = provenance.get("missing_policy", "error")
    if prov_dtype != expected_dtype or prov_missing != expected_missing:
        raise ValueError(
            f"Provenance dtype/missing_policy ({prov_dtype!r}/{prov_missing!r}) "
            f"does not match recipe ({expected_dtype!r}/{expected_missing!r}). "
            f"Identity mismatch — delete the composed directory."
        )

    # Structural completeness: config.json must have been copied
    if not (composed / "config.json").exists():
        return False, "config.json missing from composed output; composition incomplete"

    # At least one safetensors shard must exist
    shard_files = [f for f in composed.rglob("*.safetensors") if f.is_file()]
    if not shard_files:
        return False, "No safetensors shards in composed output; composition incomplete"

    return True, f"Composition verified complete at {config.composed_path}"


# ─── Launcher ────────────────────────────────────────────────────────────────

def launch_worker(config: CompressionConfig) -> tuple[int, str]:
    """
    Save config and initial job state, then launch a detached worker subprocess.

    Returns ``(pid, config_path)``.

    Raises ``RuntimeError`` if a job for the same work directory is already
    running (live PID) or has already completed successfully.
    """
    work_path = config.work_path
    Path(work_path).mkdir(parents=True, exist_ok=True)

    # ── Duplicate-launch guard ────────────────────────────────────────────────
    state = _load_job_state(work_path)
    if state:
        status = state.get("status", "")
        pid = state.get("pid")
        if status in ("queued", "preflight", "composing", "compressing"):
            if pid and _check_pid_alive(pid):
                raise RuntimeError(
                    f"Job '{config.output_name}' is already running "
                    f"(PID {pid}, status={status}).\n"
                    f"Stop it first:  Stop-Process -Id {pid}\n"
                    f"Then relaunch once stopped (work checkpoint will be reused)."
                )
        elif status == "completed":
            raise RuntimeError(
                f"Job '{config.output_name}' has already completed successfully.\n"
                f"Output: {config.compressed_path}\n"
                f"To redo it, delete the work dir and compressed output, then relaunch."
            )

    # ── Persist config and initial state ─────────────────────────────────────
    jobs_dir = Path(config.jobs_root) / config.output_name
    jobs_dir.mkdir(parents=True, exist_ok=True)
    config_path = str(jobs_dir / "compression_config.json")
    config.save(config_path)

    initial_state = _make_initial_state(config)
    _save_job_state(work_path, initial_state)

    log_path = str(jobs_dir / "worker.log")

    # ── Build argv – no shell=True, quote-safe ────────────────────────────────
    argv = [
        sys.executable, "-m", "llmtraits",
        "compressed-job",
        "--config", config_path,
    ]

    # Launch from the package's project root so the source checkout remains
    # importable even when llmtraits is not installed into the virtualenv.
    project_root = Path(__file__).resolve().parent.parent
    worker_env = os.environ.copy()
    worker_env["PYTHONIOENCODING"] = "utf-8"

    # ── Launch detached child ─────────────────────────────────────────────────
    log_fh = open(log_path, "a", encoding="utf-8")
    try:
        log_fh.write(f"\n=== Worker started {_now_iso()} ===\n")
        log_fh.write(f"argv: {argv}\n\n")
        log_fh.flush()

        if sys.platform == "win32":
            _popen_kwargs: dict = {
                "creationflags": (
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                ),
            }
        else:
            _popen_kwargs = {"start_new_session": True}

        proc = subprocess.Popen(
            argv,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
            env=worker_env,
            **_popen_kwargs,
        )
    finally:
        log_fh.close()  # Parent closes; child keeps its inherited copy

    _update_job_state(work_path, pid=proc.pid)
    return proc.pid, config_path


# ─── Worker (foreground, launched as subprocess) ──────────────────────────────

def run_worker(config_path: str) -> None:
    """
    Two-stage worker: compose SVD model, then compress with ExLlamaV3.

    State transitions:
      queued → preflight → composing → compressing → completed
                                                   ↘ failed

    Called by the ``compressed-job`` CLI subcommand.
    Never called directly from the GUI; always via launch_worker().
    """
    config = CompressionConfig.from_file(config_path)
    work_path = config.work_path

    def _update(**kw):
        _update_job_state(work_path, **kw)

    try:
        _update(status="preflight", pid=os.getpid())
        print(
            f"[worker] PID={os.getpid()}  job={config.output_name}",
            flush=True,
        )

        # ── Preflight ────────────────────────────────────────────────────────
        ok, msg = run_preflight(config)
        if not ok:
            _update(status="failed", failed_at=_now_iso(), error=f"Preflight failed:\n{msg}")
            print(f"[worker] PREFLIGHT FAILED:\n{msg}", flush=True)
            sys.exit(1)
        print(f"[worker] Preflight OK", flush=True)

        # ── Stage 1: Compose ─────────────────────────────────────────────────
        # Load recipe for composition validation
        try:
            with open(config.recipe_path, "r", encoding="utf-8") as _fh:
                _recipe = json.load(_fh)
        except Exception as exc:
            _update(status="failed", failed_at=_now_iso(), error=f"Cannot load recipe: {exc}")
            print(f"[worker] RECIPE LOAD FAILED: {exc}", flush=True)
            sys.exit(1)

        # Validate composition status with identity check
        try:
            composed_done, compose_msg = _validate_composition_complete(config, _recipe)
        except ValueError as exc:
            _update(status="failed", failed_at=_now_iso(),
                    error=f"Composition identity mismatch:\n{exc}")
            print(f"[worker] COMPOSITION IDENTITY MISMATCH:\n{exc}", flush=True)
            sys.exit(1)

        state = _load_job_state(work_path)
        if not state.get("compose_started_at"):
            _update(status="composing", compose_started_at=_now_iso())
        else:
            _update(status="composing")

        if composed_done:
            print(
                f"[worker] Composition already complete at {config.composed_path}; skipping.",
                flush=True,
            )
        else:
            print(
                f"[worker] Composing: {config.recipe_path} → {config.composed_path}",
                flush=True,
            )
            _compose_in_worker(config.recipe_path)
            print("[worker] Composition complete.", flush=True)

            # Post-compose validation
            try:
                composed_done, compose_msg = _validate_composition_complete(config, _recipe)
            except ValueError as exc:
                _update(status="failed", failed_at=_now_iso(),
                        error=f"Post-compose identity mismatch:\n{exc}")
                sys.exit(1)
            if not composed_done:
                _update(status="failed", failed_at=_now_iso(),
                        error=f"Composition incomplete after running: {compose_msg}")
                print(f"[worker] COMPOSITION INCOMPLETE: {compose_msg}", flush=True)
                sys.exit(1)

        _update(compose_completed_at=_now_iso())

        # ── Stage 2: Compress ─────────────────────────────────────────────────
        state = _load_job_state(work_path)
        if not state.get("compress_started_at"):
            _update(status="compressing", compress_started_at=_now_iso())
        else:
            _update(status="compressing")

        # Check if already done (e.g., previous run completed conversion)
        already_done, _, existing_stats = _validate_converter_output(config.compressed_path)
        if already_done:
            print(
                f"[worker] Compressed output already valid at {config.compressed_path}; "
                f"skipping converter.",
                flush=True,
            )
            _update(
                status="completed",
                completed_at=_now_iso(),
                compress_completed_at=_now_iso(),
                output_size_bytes=existing_stats.get("output_size_bytes"),
                output_files=existing_stats.get("output_files"),
            )
            return

        # Determine initial vs resume
        try:
            do_resume, resume_reason = _should_resume_conversion(config)
        except ValueError as exc:
            _update(status="failed", failed_at=_now_iso(), error=str(exc))
            print(f"[worker] WORK-DIR MISMATCH:\n{exc}", flush=True)
            sys.exit(1)

        if do_resume:
            argv = _build_resume_convert_argv(config)
        else:
            argv = _build_initial_convert_argv(config)

        _update(compress_argv=argv)
        print(f"[worker] Running: {' '.join(repr(a) for a in argv)}", flush=True)

        env = _converter_env(config)
        script_dir = str(Path(config.convert_script).parent)

        Path(config.compressed_path).mkdir(parents=True, exist_ok=True)

        result = subprocess.run(argv, cwd=script_dir, env=env)

        if result.returncode != 0:
            _update(
                status="failed",
                failed_at=_now_iso(),
                error=f"Converter exited with code {result.returncode}",
            )
            print(
                f"[worker] CONVERTER FAILED (exit {result.returncode})", flush=True
            )
            sys.exit(1)

        # Validate output even on exit 0 — convert.py may emit errors and exit 0
        ok, val_msg, stats = _validate_converter_output(config.compressed_path)
        if not ok:
            _update(
                status="failed",
                failed_at=_now_iso(),
                error=f"Converter exited 0 but output validation failed: {val_msg}",
            )
            print(f"[worker] OUTPUT INVALID: {val_msg}", flush=True)
            sys.exit(1)

        size_gb = (stats.get("output_size_bytes") or 0) / 1024 ** 3
        _update(
            status="completed",
            completed_at=_now_iso(),
            compress_completed_at=_now_iso(),
            output_size_bytes=stats.get("output_size_bytes"),
            output_files=stats.get("output_files"),
        )
        print(
            f"[worker] SUCCESS  output={config.compressed_path}  size={size_gb:.2f} GB",
            flush=True,
        )

    except KeyboardInterrupt:
        _update(
            status="stopped",
            failed_at=_now_iso(),
            error="Worker stopped by KeyboardInterrupt",
        )
        print("[worker] Stopped by KeyboardInterrupt", flush=True)
        sys.exit(130)

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        _update(status="failed", failed_at=_now_iso(), error=f"{exc}\n{tb}")
        print(f"[worker] UNHANDLED ERROR: {exc}\n{tb}", flush=True)
        sys.exit(1)


# ─── Status helpers ───────────────────────────────────────────────────────────

def get_job_status(work_path: str) -> dict:
    """
    Load job state and enrich it with a live PID-liveness check.
    Returns ``{"status": "not_started"}`` when no state file exists.
    """
    state = _load_job_state(work_path)
    if not state:
        return {"status": "not_started"}
    pid = state.get("pid")
    state["pid_alive"] = _check_pid_alive(pid) if pid else False
    return state


def tail_log(jobs_root: str, output_name: str, n_lines: int = 60) -> str:
    """Return the last *n_lines* of the worker log, or a helpful message."""
    log_path = Path(jobs_root) / output_name / "worker.log"
    if not log_path.exists():
        return "(no log file yet — worker has not been launched)"
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return "".join(lines[-n_lines:])
    except Exception as exc:
        return f"(error reading log: {exc})"


def log_path_for(jobs_root: str, output_name: str) -> str:
    """Return the absolute path to the worker log file."""
    return str(Path(jobs_root) / output_name / "worker.log")
