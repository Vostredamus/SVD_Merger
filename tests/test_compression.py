"""
Tests for llmtraits.compression — no real GPU, no real ExLlamaV3, no real model files.

All subprocesses and heavy I/O are mocked or use tiny synthetic data.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llmtraits.compression import (
    BPW_MAX,
    BPW_MIN,
    DEFAULT_BPW,
    DEFAULT_CONVERT_SCRIPT,
    DEFAULT_CONVERTER_PYTHON,
    DEFAULT_CUDA_DEVICES,
    DEFAULT_JOBS_ROOT,
    CompressionConfig,
    _atomic_json_write,
    _build_initial_convert_argv,
    _build_resume_convert_argv,
    _check_pid_alive,
    _compose_in_worker,
    _load_job_state,
    _make_initial_state,
    _save_job_state,
    _should_resume_conversion,
    _update_job_state,
    _validate_composition_complete,
    _validate_converter_output,
    default_output_name,
    format_bpw,
    get_job_status,
    run_preflight,
    run_worker,
    sanitize_folder_name,
    tail_log,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def fake_python(tmp_path: Path) -> Path:
    """Create a tiny fake python executable (just a .exe stub that exists)."""
    exe = tmp_path / "fake_python.exe"
    exe.write_bytes(b"")
    return exe


@pytest.fixture()
def fake_script(tmp_path: Path) -> Path:
    script = tmp_path / "convert.py"
    script.write_text("# fake convert.py\n")
    return script


@pytest.fixture()
def fake_recipe(tmp_path: Path) -> Path:
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "schema": "svd_recipe",
        "version": "1.0",
        "analysis_dir": str(tmp_path),
        "departure_label": "ModelA",
        "departure_path": str(tmp_path / "departure"),
        "output_path": str(tmp_path / "composed"),
        "dtype": "float16",
        "missing_policy": "error",
        "adjustments": [],
    }))
    return recipe


@pytest.fixture()
def minimal_config(tmp_path: Path, fake_python: Path, fake_script: Path, fake_recipe: Path) -> CompressionConfig:
    return CompressionConfig(
        recipe_path=str(fake_recipe),
        model_name="TestModel",
        output_name="TestModel_5.5bw",
        composed_path=str(tmp_path / "composed" / "TestModel"),
        compressed_path=str(tmp_path / "compressed" / "TestModel_5.5bw"),
        work_path=str(tmp_path / "work" / "TestModel_5.5bw"),
        jobs_root=str(tmp_path / "jobs"),
        converter_python=str(fake_python),
        convert_script=str(fake_script),
        bpw=5.5,
        cuda_devices="0",
    )


# ─── Naming helpers ───────────────────────────────────────────────────────────

class TestNamingHelpers:
    def test_sanitize_strips_illegal_chars(self):
        assert "\\" not in sanitize_folder_name("a\\b")
        assert "/" not in sanitize_folder_name("a/b")
        assert ":" not in sanitize_folder_name("a:b")
        assert "*" not in sanitize_folder_name("a*b")
        assert "?" not in sanitize_folder_name("a?b")

    def test_sanitize_preserves_descriptive_name(self):
        name = "Example-L3.3-70B-Model-V1.5"
        result = sanitize_folder_name(name)
        assert "Example-L3.3-70B-Model-V1.5" == result

    def test_format_bpw_strips_trailing_zero(self):
        assert format_bpw(4.0) == "4"
        assert format_bpw(5.5) == "5.5"
        assert format_bpw(8.0) == "8"

    def test_default_output_name_pattern(self):
        name = default_output_name("MyModel", 5.5)
        assert name == "MyModel_5.5bw"

    def test_default_output_name_whole_bpw(self):
        name = default_output_name("MyModel", 4.0)
        assert name == "MyModel_4bw"

    def test_default_output_name_sanitizes(self):
        name = default_output_name("My/Model:A", 5.5)
        assert "/" not in name
        assert ":" not in name
        assert name.endswith("_5.5bw")

    # ── New sanitize tests ────────────────────────────────────────────────────

    def test_sanitize_dot_becomes_underscore(self):
        assert sanitize_folder_name(".") == "_"

    def test_sanitize_dotdot_becomes_underscore(self):
        assert sanitize_folder_name("..") == "_"

    def test_sanitize_many_dots_become_underscore(self):
        assert sanitize_folder_name("...") == "_"

    def test_sanitize_reserved_windows_name_prefixed(self):
        assert sanitize_folder_name("NUL") == "_NUL"
        assert sanitize_folder_name("CON") == "_CON"
        assert sanitize_folder_name("COM1") == "_COM1"
        assert sanitize_folder_name("LPT9") == "_LPT9"
        assert sanitize_folder_name("nul") == "_nul"  # case-insensitive

    def test_sanitize_trailing_dot_stripped(self):
        assert sanitize_folder_name("mymodel.") == "mymodel"

    def test_sanitize_trailing_space_stripped(self):
        assert sanitize_folder_name("mymodel ") == "mymodel"

    def test_sanitize_empty_string_becomes_underscore(self):
        assert sanitize_folder_name("") == "_"

    def test_sanitize_path_traversal_illegal_chars_replaced(self):
        # / and \ are replaced; the result should not be a traversal
        result = sanitize_folder_name("../../etc")
        assert "/" not in result
        assert "\\" not in result
        assert ".." in result  # dots remain but path chars are gone


# ─── Config round-trip ────────────────────────────────────────────────────────

class TestCompressionConfig:
    def test_roundtrip(self, minimal_config: CompressionConfig, tmp_path: Path):
        p = str(tmp_path / "cfg.json")
        minimal_config.save(p)
        loaded = CompressionConfig.from_file(p)
        assert loaded.bpw == minimal_config.bpw
        assert loaded.output_name == minimal_config.output_name
        assert loaded.converter_python == minimal_config.converter_python

    def test_from_dict_wrong_schema_raises(self):
        with pytest.raises(ValueError, match="schema"):
            CompressionConfig.from_dict({"schema": "wrong", "version": "1.0"})

    def test_from_dict_wrong_version_raises(self):
        with pytest.raises(ValueError, match="version"):
            CompressionConfig.from_dict({"schema": "compression_job", "version": "99.0"})

    def test_validate_missing_recipe(self, minimal_config: CompressionConfig):
        minimal_config.recipe_path = "/does/not/exist.json"
        errors = minimal_config.validate()
        assert any("recipe" in e.lower() for e in errors)

    def test_validate_bpw_out_of_range(self, minimal_config: CompressionConfig):
        minimal_config.bpw = 0.5
        errors = minimal_config.validate()
        assert any("bpw" in e.lower() for e in errors)

    def test_validate_bpw_max(self, minimal_config: CompressionConfig):
        minimal_config.bpw = 8.5
        errors = minimal_config.validate()
        assert any("bpw" in e.lower() for e in errors)

    def test_validate_missing_converter_python(self, minimal_config: CompressionConfig):
        minimal_config.converter_python = "/no/such/python.exe"
        errors = minimal_config.validate()
        assert any("converter" in e.lower() or "python" in e.lower() for e in errors)

    def test_validate_missing_convert_script(self, minimal_config: CompressionConfig):
        minimal_config.convert_script = "/no/such/convert.py"
        errors = minimal_config.validate()
        assert any("script" in e.lower() or "convert" in e.lower() for e in errors)

    def test_validate_empty_output_name(self, minimal_config: CompressionConfig):
        minimal_config.output_name = "   "
        errors = minimal_config.validate()
        assert any("output_name" in e for e in errors)

    def test_validate_unsafe_output_name_rejected(self, minimal_config: CompressionConfig):
        """output_name with illegal chars must be reported as unsafe."""
        minimal_config.output_name = "bad:name"
        errors = minimal_config.validate()
        assert any("output_name" in e for e in errors)

    def test_validate_reserved_output_name_rejected(self, minimal_config: CompressionConfig):
        """Reserved Windows names must be rejected."""
        minimal_config.output_name = "NUL"
        errors = minimal_config.validate()
        assert any("output_name" in e for e in errors)

    def test_validate_clean_config_no_errors(self, minimal_config: CompressionConfig):
        assert minimal_config.validate() == []


# ─── Atomic JSON write ────────────────────────────────────────────────────────

class TestAtomicJsonWrite:
    def test_writes_and_reads_back(self, tmp_path: Path):
        p = str(tmp_path / "test.json")
        _atomic_json_write(p, {"key": "value"})
        with open(p) as fh:
            data = json.load(fh)
        assert data["key"] == "value"

    def test_no_tmp_file_left_behind(self, tmp_path: Path):
        p = str(tmp_path / "test.json")
        _atomic_json_write(p, {"x": 1})
        assert not Path(p + ".tmp").exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        p = str(tmp_path / "a" / "b" / "test.json")
        _atomic_json_write(p, {"y": 2})
        assert Path(p).exists()


# ─── Job state ────────────────────────────────────────────────────────────────

class TestJobState:
    def test_round_trip(self, tmp_path: Path):
        work = str(tmp_path / "work")
        state = {"status": "queued", "pid": 1234}
        _save_job_state(work, state)
        loaded = _load_job_state(work)
        assert loaded["status"] == "queued"
        assert loaded["pid"] == 1234

    def test_missing_returns_empty(self, tmp_path: Path):
        assert _load_job_state(str(tmp_path / "no_such")) == {}

    def test_update_patches_fields(self, tmp_path: Path):
        work = str(tmp_path / "work")
        _save_job_state(work, {"status": "queued", "pid": None})
        _update_job_state(work, status="composing", pid=999)
        state = _load_job_state(work)
        assert state["status"] == "composing"
        assert state["pid"] == 999

    def test_initial_state_structure(self, minimal_config: CompressionConfig):
        state = _make_initial_state(minimal_config)
        assert state["status"] == "queued"
        assert state["bpw"] == 5.5
        assert state["output_name"] == "TestModel_5.5bw"
        assert "queued_at" in state
        assert state["pid"] is None


# ─── Preflight ────────────────────────────────────────────────────────────────

class TestPreflight:
    def _successful_run(self, *args, **kwargs):
        r = SimpleNamespace()
        r.returncode = 0
        r.stdout = "LLMTRAITS_PREFLIGHT_JSON=" + json.dumps(
            {
                "python_version": "3.10.11",
                "torch_found": True,
                "torch_version": "2.9.0+cu128",
                "exllamav3_found": True,
                "exllamav3_version": "0.0.26",
                "ninja_found": True,
                "ninja_version": "1.13.0",
                "ninja_executable": r"C:\venv\Scripts\ninja.exe",
                "converter_syntax": "ok",
                "cuda_available": True,
                "gpu_count": 1,
            }
        )
        r.stderr = ""
        return r

    def test_preflight_ok(self, minimal_config: CompressionConfig):
        with patch("subprocess.run", side_effect=self._successful_run) as run:
            with patch("shutil.disk_usage", return_value=(500e9, 100e9, 400e9)):
                ok, msg = run_preflight(minimal_config)
        assert ok
        assert "OK" in msg
        command = run.call_args.args[0]
        assert command[1] == "-c"
        assert "--help" not in command
        assert str(minimal_config.convert_script) == command[-1]
        assert "static checks only" in msg

    def test_preflight_missing_python(self, minimal_config: CompressionConfig, tmp_path: Path):
        minimal_config.converter_python = str(tmp_path / "no_python.exe")
        ok, msg = run_preflight(minimal_config)
        assert not ok
        assert "not found" in msg.lower() or "python" in msg.lower()

    def test_preflight_missing_script(self, minimal_config: CompressionConfig, tmp_path: Path):
        minimal_config.convert_script = str(tmp_path / "no_script.py")
        ok, msg = run_preflight(minimal_config)
        assert not ok
        assert "script" in msg.lower() or "convert" in msg.lower()

    def test_preflight_import_error_in_output(self, minimal_config: CompressionConfig):
        def bad_run(*args, **kwargs):
            r = SimpleNamespace()
            r.returncode = 1
            r.stdout = ""
            r.stderr = "ModuleNotFoundError: No module named 'torch'"
            return r

        with patch("subprocess.run", side_effect=bad_run):
            with patch("shutil.disk_usage", return_value=(500e9, 100e9, 400e9)):
                ok, msg = run_preflight(minimal_config)
        assert not ok
        assert "torch" in msg or "module" in msg.lower()

    def test_preflight_no_module_named_in_stdout(self, minimal_config: CompressionConfig):
        def bad_run(*args, **kwargs):
            r = SimpleNamespace()
            r.returncode = 0
            r.stdout = "No module named 'exllamav3'"
            r.stderr = ""
            return r

        with patch("subprocess.run", side_effect=bad_run):
            with patch("shutil.disk_usage", return_value=(500e9, 100e9, 400e9)):
                ok, msg = run_preflight(minimal_config)
        assert not ok
        assert "module" in msg.lower() or "exllamav3" in msg

    def test_preflight_timeout(self, minimal_config: CompressionConfig):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=[], timeout=30)):
            with patch("shutil.disk_usage", return_value=(500e9, 100e9, 400e9)):
                ok, msg = run_preflight(minimal_config)
        assert not ok
        assert "timed out" in msg.lower()

    def test_preflight_file_not_found(self, minimal_config: CompressionConfig):
        with patch("subprocess.run", side_effect=FileNotFoundError("not found")):
            with patch("shutil.disk_usage", return_value=(500e9, 100e9, 400e9)):
                ok, msg = run_preflight(minimal_config)
        assert not ok

    def test_preflight_low_disk_space(self, minimal_config: CompressionConfig):
        with patch("shutil.disk_usage", return_value=(1e12, 990e9, 5e9)):
            ok, msg = run_preflight(minimal_config)
        assert not ok
        assert "gb" in msg.lower() or "space" in msg.lower() or "free" in msg.lower()

    def test_preflight_config_validation_errors_block_subprocess(self, minimal_config: CompressionConfig):
        minimal_config.bpw = 999.0
        with patch("subprocess.run") as mock_run:
            ok, msg = run_preflight(minimal_config)
        assert not ok
        mock_run.assert_not_called()  # Never reaches subprocess if config is invalid


# ─── Converter argv builders ──────────────────────────────────────────────────

class TestConverterArgv:
    def test_initial_argv_contains_required_flags(self, minimal_config: CompressionConfig):
        argv = _build_initial_convert_argv(minimal_config)
        assert minimal_config.converter_python == argv[0]
        assert minimal_config.convert_script == argv[1]
        assert "-i" in argv
        assert "-o" in argv
        assert "-w" in argv
        assert "-b" in argv
        assert "-d" in argv
        assert "-v" in argv
        # Check values follow their flags
        i_idx = argv.index("-i")
        assert argv[i_idx + 1] == minimal_config.composed_path
        o_idx = argv.index("-o")
        assert argv[o_idx + 1] == minimal_config.compressed_path
        w_idx = argv.index("-w")
        assert argv[w_idx + 1] == minimal_config.work_path
        b_idx = argv.index("-b")
        assert argv[b_idx + 1] == str(minimal_config.bpw)
        d_idx = argv.index("-d")
        assert argv[d_idx + 1] == minimal_config.cuda_devices

    def test_initial_argv_no_resume_flag(self, minimal_config: CompressionConfig):
        argv = _build_initial_convert_argv(minimal_config)
        assert "-resume" not in argv
        assert "--resume" not in argv

    def test_resume_argv_contains_resume_flag(self, minimal_config: CompressionConfig):
        argv = _build_resume_convert_argv(minimal_config)
        assert "-resume" in argv
        assert "-w" in argv
        w_idx = argv.index("-w")
        assert argv[w_idx + 1] == minimal_config.work_path
        assert "-v" in argv

    def test_resume_argv_no_input_output(self, minimal_config: CompressionConfig):
        argv = _build_resume_convert_argv(minimal_config)
        assert "-i" not in argv
        assert "-o" not in argv
        assert "-b" not in argv


# ─── Resume decision ──────────────────────────────────────────────────────────

class TestShouldResumeConversion:
    def test_no_args_json_returns_false(self, minimal_config: CompressionConfig, tmp_path: Path):
        minimal_config.work_path = str(tmp_path / "empty_work")
        Path(minimal_config.work_path).mkdir()
        do_resume, reason = _should_resume_conversion(minimal_config)
        assert not do_resume
        assert "no args.json" in reason.lower()

    def test_matching_args_json_returns_true(self, minimal_config: CompressionConfig, tmp_path: Path):
        """Matching bpw/input_dir/out_dir aliases → resume."""
        minimal_config.work_path = str(tmp_path / "work")
        Path(minimal_config.work_path).mkdir()
        args_json = {
            "bpw": minimal_config.bpw,
            "input_dir": minimal_config.composed_path,
            "out_dir": minimal_config.compressed_path,
        }
        with open(Path(minimal_config.work_path) / "args.json", "w") as fh:
            json.dump(args_json, fh)
        do_resume, reason = _should_resume_conversion(minimal_config)
        assert do_resume

    def test_exllamav3_canonical_keys_resume(self, minimal_config: CompressionConfig, tmp_path: Path):
        """ExLlamaV3 actual keys bits/in_dir/out_dir → resume."""
        minimal_config.work_path = str(tmp_path / "work")
        Path(minimal_config.work_path).mkdir()
        args_json = {
            "bits": minimal_config.bpw,
            "in_dir": minimal_config.composed_path,
            "out_dir": minimal_config.compressed_path,
        }
        with open(Path(minimal_config.work_path) / "args.json", "w") as fh:
            json.dump(args_json, fh)
        do_resume, reason = _should_resume_conversion(minimal_config)
        assert do_resume

    def test_bpw_mismatch_raises(self, minimal_config: CompressionConfig, tmp_path: Path):
        """Partial identity (only bpw, no input/output) → ValueError mentioning bpw."""
        minimal_config.work_path = str(tmp_path / "work")
        Path(minimal_config.work_path).mkdir()
        args_json = {
            "bpw": 4.0,  # different from 5.5
        }
        with open(Path(minimal_config.work_path) / "args.json", "w") as fh:
            json.dump(args_json, fh)
        with pytest.raises(ValueError, match="bpw"):
            _should_resume_conversion(minimal_config)

    def test_input_dir_mismatch_raises(self, minimal_config: CompressionConfig, tmp_path: Path):
        """Partial identity (only input_dir, no bits/output) → ValueError mentioning input_dir."""
        minimal_config.work_path = str(tmp_path / "work")
        Path(minimal_config.work_path).mkdir()
        args_json = {
            "input_dir": r"D:\work\composed\OtherModel",
        }
        with open(Path(minimal_config.work_path) / "args.json", "w") as fh:
            json.dump(args_json, fh)
        with pytest.raises(ValueError, match="input_dir"):
            _should_resume_conversion(minimal_config)

    def test_corrupt_json_raises(
        self, minimal_config: CompressionConfig, tmp_path: Path
    ):
        """Corrupt args.json must raise ValueError (fail closed, not fall back to initial)."""
        minimal_config.work_path = str(tmp_path / "work")
        Path(minimal_config.work_path).mkdir()
        (Path(minimal_config.work_path) / "args.json").write_text("{invalid json}")
        with pytest.raises(ValueError, match="corrupt"):
            _should_resume_conversion(minimal_config)

    def test_unknown_keys_only_raises(
        self, minimal_config: CompressionConfig, tmp_path: Path
    ):
        """args.json with only unrecognised keys → fail closed (ValueError)."""
        minimal_config.work_path = str(tmp_path / "work")
        Path(minimal_config.work_path).mkdir()
        (Path(minimal_config.work_path) / "args.json").write_text(
            json.dumps({"unknown_key": 42, "another": "value"})
        )
        with pytest.raises(ValueError, match="recognized"):
            _should_resume_conversion(minimal_config)

    def test_partial_identity_bits_only_raises(
        self, minimal_config: CompressionConfig, tmp_path: Path
    ):
        """Only bits present without input/output → fail closed (partial identity)."""
        minimal_config.work_path = str(tmp_path / "work")
        Path(minimal_config.work_path).mkdir()
        (Path(minimal_config.work_path) / "args.json").write_text(
            json.dumps({"bits": minimal_config.bpw})
        )
        with pytest.raises(ValueError, match="partial"):
            _should_resume_conversion(minimal_config)

    def test_all_three_mismatch_raises(
        self, minimal_config: CompressionConfig, tmp_path: Path
    ):
        """All three fields present but bpw mismatches → ValueError."""
        minimal_config.work_path = str(tmp_path / "work")
        Path(minimal_config.work_path).mkdir()
        args_json = {
            "bits": 4.0,  # wrong bpw
            "in_dir": minimal_config.composed_path,
            "out_dir": minimal_config.compressed_path,
        }
        with open(Path(minimal_config.work_path) / "args.json", "w") as fh:
            json.dump(args_json, fh)
        with pytest.raises(ValueError, match="bpw"):
            _should_resume_conversion(minimal_config)


# ─── Composition validation ───────────────────────────────────────────────────

def _make_recipe_dict(tmp_path: Path) -> dict:
    """Build a minimal recipe dict for composition tests."""
    return {
        "schema": "svd_recipe",
        "version": "1.0",
        "analysis_dir": str(tmp_path),
        "departure_label": "ModelA",
        "departure_path": str(tmp_path / "departure"),
        "output_path": str(tmp_path / "composed"),
        "dtype": "float16",
        "missing_policy": "error",
        "adjustments": [],
    }


def _make_composed_dir(composed_path: Path, recipe: dict) -> None:
    """
    Create a structurally valid and matching composed directory.
    Used by both composition-validation tests and worker tests.
    """
    composed_path.mkdir(parents=True, exist_ok=True)
    resolved = {"original_recipe": recipe}
    (composed_path / "resolved_recipe.json").write_text(
        json.dumps(resolved, indent=2)
    )
    provenance = {
        "base_dir": os.path.abspath(recipe.get("departure_path", str(composed_path))),
        "deltas": [],
        "dtype": recipe.get("dtype", "float16"),
        "missing_policy": recipe.get("missing_policy", "error"),
        "created_at": "2025-01-01T00:00:00+00:00",
    }
    (composed_path / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (composed_path / "config.json").write_text("{}")
    (composed_path / "model.safetensors").write_bytes(b"fake_shard_data" * 100)


class TestValidateCompositionComplete:
    def test_no_composed_dir_returns_not_done(
        self, minimal_config: CompressionConfig, tmp_path: Path
    ):
        recipe = _make_recipe_dict(tmp_path)
        done, msg = _validate_composition_complete(minimal_config, recipe)
        assert not done

    def test_only_resolved_recipe_returns_not_done(
        self, minimal_config: CompressionConfig, tmp_path: Path
    ):
        recipe = _make_recipe_dict(tmp_path)
        composed = Path(minimal_config.composed_path)
        composed.mkdir(parents=True, exist_ok=True)
        (composed / "resolved_recipe.json").write_text(
            json.dumps({"original_recipe": recipe})
        )
        done, msg = _validate_composition_complete(minimal_config, recipe)
        assert not done
        assert "provenance" in msg.lower()

    def test_only_provenance_returns_not_done(
        self, minimal_config: CompressionConfig, tmp_path: Path
    ):
        recipe = _make_recipe_dict(tmp_path)
        composed = Path(minimal_config.composed_path)
        composed.mkdir(parents=True, exist_ok=True)
        (composed / "provenance.json").write_text("{}")
        done, msg = _validate_composition_complete(minimal_config, recipe)
        assert not done

    def test_valid_composition_returns_done(
        self, minimal_config: CompressionConfig, tmp_path: Path
    ):
        recipe = _make_recipe_dict(tmp_path)
        _make_composed_dir(Path(minimal_config.composed_path), recipe)
        done, msg = _validate_composition_complete(minimal_config, recipe)
        assert done, msg

    def test_wrong_recipe_raises_value_error(
        self, minimal_config: CompressionConfig, tmp_path: Path
    ):
        """A composed dir from a different recipe must raise ValueError."""
        original_recipe = _make_recipe_dict(tmp_path)
        _make_composed_dir(Path(minimal_config.composed_path), original_recipe)

        different_recipe = dict(original_recipe)
        different_recipe["departure_label"] = "DifferentModel"

        with pytest.raises(ValueError, match="different recipe|does not match"):
            _validate_composition_complete(minimal_config, different_recipe)

    def test_departure_path_mismatch_raises(
        self, minimal_config: CompressionConfig, tmp_path: Path
    ):
        """provenance.base_dir ≠ recipe departure_path → ValueError."""
        recipe = _make_recipe_dict(tmp_path)
        composed = Path(minimal_config.composed_path)
        composed.mkdir(parents=True, exist_ok=True)
        (composed / "resolved_recipe.json").write_text(
            json.dumps({"original_recipe": recipe})
        )
        # Provenance with a wrong base_dir
        bad_provenance = {
            "base_dir": str(tmp_path / "wrong_departure"),
            "dtype": "float16",
            "missing_policy": "error",
            "created_at": "2025-01-01T00:00:00+00:00",
        }
        (composed / "provenance.json").write_text(json.dumps(bad_provenance))
        (composed / "config.json").write_text("{}")
        (composed / "model.safetensors").write_bytes(b"x" * 100)

        with pytest.raises(ValueError, match="base_dir|Identity mismatch|departure_path"):
            _validate_composition_complete(minimal_config, recipe)

    def test_missing_config_json_returns_not_done(
        self, minimal_config: CompressionConfig, tmp_path: Path
    ):
        recipe = _make_recipe_dict(tmp_path)
        composed = Path(minimal_config.composed_path)
        _make_composed_dir(composed, recipe)
        (composed / "config.json").unlink()  # remove config
        done, msg = _validate_composition_complete(minimal_config, recipe)
        assert not done
        assert "config.json" in msg.lower()

    def test_incomplete_no_shards_returns_not_done(
        self, minimal_config: CompressionConfig, tmp_path: Path
    ):
        recipe = _make_recipe_dict(tmp_path)
        composed = Path(minimal_config.composed_path)
        _make_composed_dir(composed, recipe)
        (composed / "model.safetensors").unlink()  # remove the shard
        done, msg = _validate_composition_complete(minimal_config, recipe)
        assert not done


# ─── Output validation ────────────────────────────────────────────────────────

def _make_exl3_config() -> str:
    """Return a config.json string representing an EXL3 quantised model."""
    return json.dumps({"quantization_config": {"quant_method": "exl3"}})


class TestValidateConverterOutput:
    def test_missing_dir_fails(self, tmp_path: Path):
        ok, msg, stats = _validate_converter_output(str(tmp_path / "no_such_dir"))
        assert not ok
        assert "not exist" in msg.lower() or "does not exist" in msg.lower()

    def test_empty_dir_fails(self, tmp_path: Path):
        out = tmp_path / "empty"
        out.mkdir()
        ok, msg, stats = _validate_converter_output(str(out))
        assert not ok

    def test_missing_config_json_fails(self, tmp_path: Path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "tokenizer.json").write_text("{}")
        (out / "model.safetensors").write_bytes(b"x" * 2000)
        ok, msg, stats = _validate_converter_output(str(out))
        assert not ok
        assert "config.json" in msg.lower() or "missing" in msg.lower()

    def test_missing_tokenizer_fails(self, tmp_path: Path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "config.json").write_text(_make_exl3_config())
        (out / "quantization_config.json").write_text("{}")
        (out / "model.safetensors").write_bytes(b"x" * 2000)
        ok, msg, stats = _validate_converter_output(str(out))
        assert not ok
        assert "tokenizer" in msg.lower()

    def test_no_substantive_model_file_fails(self, tmp_path: Path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "config.json").write_text(_make_exl3_config())
        (out / "quantization_config.json").write_text("{}")
        (out / "tokenizer.json").write_text("{}")
        # Only a tiny file, below the 1 KB threshold
        (out / "model.safetensors").write_bytes(b"x" * 100)
        ok, msg, stats = _validate_converter_output(str(out))
        assert not ok
        assert "substantive" in msg.lower() or "model" in msg.lower()

    def test_hf_safetensors_no_quant_config_fails(self, tmp_path: Path):
        """A plain HF safetensors dir (no EXL3 quantization_config) must fail."""
        out = tmp_path / "hf_model"
        out.mkdir()
        # HF config without quantization_config.quant_method == "exl3"
        (out / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (out / "tokenizer.json").write_text("{}")
        (out / "model.safetensors").write_bytes(b"y" * 5000)
        ok, msg, stats = _validate_converter_output(str(out))
        assert not ok
        assert "exl3" in msg.lower() or "quant_method" in msg.lower()

    def test_unparseable_config_json_fails(self, tmp_path: Path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "config.json").write_text("{not valid json")
        (out / "tokenizer.json").write_text("{}")
        (out / "model.safetensors").write_bytes(b"x" * 2000)
        ok, msg, stats = _validate_converter_output(str(out))
        assert not ok
        assert "parseable" in msg.lower() or "config.json" in msg.lower()

    def test_valid_exl3_safetensors_output_succeeds(self, tmp_path: Path):
        """A real EXL3 output with safetensors shards must pass."""
        out = tmp_path / "out"
        out.mkdir()
        (out / "config.json").write_text(_make_exl3_config())
        (out / "quantization_config.json").write_text(json.dumps({"bits": 5.5}))
        (out / "tokenizer.json").write_text("{}")
        big = out / "model-00001-of-00001.safetensors"
        big.write_bytes(b"x" * 5000)
        ok, msg, stats = _validate_converter_output(str(out))
        assert ok, msg
        assert stats["output_size_bytes"] > 0
        assert any("safetensors" in f.lower() for f in stats["output_files"])




# ─── get_job_status / tail_log ────────────────────────────────────────────────

class TestStatusHelpers:
    def test_not_started(self, tmp_path: Path):
        state = get_job_status(str(tmp_path / "no_job"))
        assert state["status"] == "not_started"

    def test_reads_existing_state(self, tmp_path: Path):
        work = str(tmp_path / "work")
        _save_job_state(work, {"status": "composing", "pid": 1111})
        state = get_job_status(work)
        assert state["status"] == "composing"
        assert "pid_alive" in state  # enriched with liveness check

    def test_tail_log_no_file(self, tmp_path: Path):
        result = tail_log(str(tmp_path), "no_job")
        assert "no log" in result.lower()

    def test_tail_log_returns_last_n_lines(self, tmp_path: Path):
        jobs_root = str(tmp_path)
        output_name = "test_job"
        log_dir = tmp_path / output_name
        log_dir.mkdir()
        lines = [f"line {i}\n" for i in range(200)]
        (log_dir / "worker.log").write_text("".join(lines))
        result = tail_log(jobs_root, output_name, n_lines=10)
        returned_lines = result.strip().splitlines()
        assert len(returned_lines) == 10
        assert "line 199" in result


# ─── Duplicate launch prevention ──────────────────────────────────────────────

class TestLaunchWorker:
    def test_duplicate_running_job_raises(self, minimal_config: CompressionConfig):
        """A job with a live PID and active status must block a second launch."""
        _save_job_state(minimal_config.work_path, {
            "status": "compressing",
            "pid": 99999,
        })
        with patch("llmtraits.compression._check_pid_alive", return_value=True):
            from llmtraits.compression import launch_worker
            with pytest.raises(RuntimeError, match="already running"):
                launch_worker(minimal_config)

    def test_completed_job_raises(self, minimal_config: CompressionConfig):
        """A completed job must block relaunch without manual cleanup."""
        _save_job_state(minimal_config.work_path, {
            "status": "completed",
            "pid": 12345,
        })
        with patch("llmtraits.compression._check_pid_alive", return_value=False):
            from llmtraits.compression import launch_worker
            with pytest.raises(RuntimeError, match="completed"):
                launch_worker(minimal_config)

    def test_dead_pid_allows_relaunch(self, minimal_config: CompressionConfig):
        """A job with a dead PID (stopped/failed) allows a new launch."""
        _save_job_state(minimal_config.work_path, {
            "status": "failed",
            "pid": 77777,
        })

        mock_proc = MagicMock()
        mock_proc.pid = 42

        with patch("llmtraits.compression._check_pid_alive", return_value=False):
            with patch("subprocess.Popen", return_value=mock_proc):
                with patch("builtins.open", side_effect=_passthrough_open):
                    from llmtraits.compression import launch_worker
                    pid, cfg_path = launch_worker(minimal_config)
        assert pid == 42

    def test_windows_creation_flags(self, minimal_config: CompressionConfig):
        """On Windows, DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP must both be set."""
        mock_proc = MagicMock()
        mock_proc.pid = 99
        captured: dict = {}

        def capturing_popen(argv, **kwargs):
            captured.update(kwargs)
            return mock_proc

        with patch("sys.platform", "win32"):
            with patch("subprocess.Popen", side_effect=capturing_popen):
                with patch("builtins.open", side_effect=_passthrough_open):
                    from llmtraits.compression import launch_worker
                    launch_worker(minimal_config)

        flags = captured.get("creationflags", 0)
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP, "CREATE_NEW_PROCESS_GROUP not set"
        assert flags & subprocess.DETACHED_PROCESS, "DETACHED_PROCESS not set"

    def test_launches_from_importable_project_root(
        self, minimal_config: CompressionConfig
    ):
        """The detached worker must be able to import the local source package."""
        mock_proc = MagicMock()
        mock_proc.pid = 99
        captured: dict = {}

        def capturing_popen(argv, **kwargs):
            captured.update(kwargs)
            return mock_proc

        with patch("subprocess.Popen", side_effect=capturing_popen):
            with patch("builtins.open", side_effect=_passthrough_open):
                from llmtraits.compression import launch_worker
                launch_worker(minimal_config)

        project_root = Path(__file__).resolve().parent.parent
        assert Path(captured["cwd"]).resolve() == project_root
        assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
        result = subprocess.run(
            [sys.executable, "-c", "import llmtraits"],
            cwd=captured["cwd"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_non_windows_no_detached_process(self, minimal_config: CompressionConfig):
        """On non-Windows, DETACHED_PROCESS must NOT be set; start_new_session used instead."""
        mock_proc = MagicMock()
        mock_proc.pid = 99
        captured: dict = {}

        def capturing_popen(argv, **kwargs):
            captured.update(kwargs)
            return mock_proc

        with patch("sys.platform", "linux"):
            with patch("subprocess.Popen", side_effect=capturing_popen):
                with patch("builtins.open", side_effect=_passthrough_open):
                    from llmtraits.compression import launch_worker
                    launch_worker(minimal_config)

        flags = captured.get("creationflags", 0)
        assert not (flags & subprocess.DETACHED_PROCESS), "DETACHED_PROCESS must not be set on non-Windows"
        assert captured.get("start_new_session") is True, "start_new_session must be True on non-Windows"


def _passthrough_open(path, *args, **kwargs):
    """Allow real file opens to work in tests that patch subprocess.Popen."""
    return open.__wrapped__(path, *args, **kwargs) if hasattr(open, "__wrapped__") else _real_open(path, *args, **kwargs)


_real_open = open


# ─── Worker state machine ─────────────────────────────────────────────────────

class TestRunWorker:
    """Test the foreground worker with mocked compose and converter subprocess."""

    @staticmethod
    def _make_valid_composition(composed_path: str, recipe_path: str) -> None:
        """
        Create a structurally valid and identity-matching composed directory.
        Reads recipe from recipe_path and writes resolved_recipe.json + provenance.json.
        """
        with open(recipe_path, "r", encoding="utf-8") as fh:
            recipe = json.load(fh)
        _make_composed_dir(Path(composed_path), recipe)

    def _make_valid_output(self, compressed_path: str) -> None:
        p = Path(compressed_path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "config.json").write_text(_make_exl3_config())
        (p / "quantization_config.json").write_text("{}")
        (p / "tokenizer.json").write_text("{}")
        (p / "model-00001-of-00001.safetensors").write_bytes(b"fake_model_data" * 200)

    def _run_worker_mocked(
        self,
        minimal_config: CompressionConfig,
        tmp_path: Path,
        *,
        compose_side_effect=None,
        subprocess_side_effect=None,
        preflight_ok: bool = True,
    ) -> dict:
        """Run the worker with heavy parts mocked; return final job state."""
        config_path = str(tmp_path / "cfg.json")
        minimal_config.save(config_path)
        _save_job_state(minimal_config.work_path, _make_initial_state(minimal_config))

        def mock_compose(recipe_path: str) -> None:
            # Create a fully valid composed directory that matches the recipe
            self._make_valid_composition(minimal_config.composed_path, recipe_path)
            if compose_side_effect is not None:
                raise compose_side_effect

        def mock_subprocess_run(argv, **kwargs) -> SimpleNamespace:
            if subprocess_side_effect is not None:
                raise subprocess_side_effect
            self._make_valid_output(minimal_config.compressed_path)
            r = SimpleNamespace()
            r.returncode = 0
            return r

        preflight_result = (preflight_ok, "Preflight OK" if preflight_ok else "Preflight FAILED")

        with patch("llmtraits.compression._compose_in_worker", side_effect=mock_compose):
            with patch("subprocess.run", side_effect=mock_subprocess_run):
                with patch("llmtraits.compression.run_preflight", return_value=preflight_result):
                    try:
                        run_worker(config_path)
                    except SystemExit:
                        pass

        return get_job_status(minimal_config.work_path)

    def test_happy_path_reaches_completed(self, minimal_config: CompressionConfig, tmp_path: Path):
        state = self._run_worker_mocked(minimal_config, tmp_path)
        assert state["status"] == "completed"
        assert state["output_size_bytes"] is not None
        assert state["completed_at"] is not None

    def test_preflight_failure_blocks_compose(self, minimal_config: CompressionConfig, tmp_path: Path):
        state = self._run_worker_mocked(minimal_config, tmp_path, preflight_ok=False)
        assert state["status"] == "failed"
        assert "Preflight" in state["error"]
        # Composed dir must not exist
        assert not Path(minimal_config.composed_path).exists()

    def test_compose_exception_sets_failed(self, minimal_config: CompressionConfig, tmp_path: Path):
        state = self._run_worker_mocked(
            minimal_config,
            tmp_path,
            compose_side_effect=RuntimeError("Compose boom"),
        )
        assert state["status"] == "failed"
        assert "Compose boom" in state["error"]

    def test_converter_exit_nonzero_fails(self, minimal_config: CompressionConfig, tmp_path: Path):
        config_path = str(tmp_path / "cfg.json")
        minimal_config.save(config_path)
        _save_job_state(minimal_config.work_path, _make_initial_state(minimal_config))

        def mock_compose(recipe_path):
            self._make_valid_composition(minimal_config.composed_path, recipe_path)

        def bad_run(argv, **kwargs):
            r = SimpleNamespace()
            r.returncode = 1
            return r

        with patch("llmtraits.compression._compose_in_worker", side_effect=mock_compose):
            with patch("subprocess.run", side_effect=bad_run):
                with patch("llmtraits.compression.run_preflight", return_value=(True, "ok")):
                    try:
                        run_worker(config_path)
                    except SystemExit:
                        pass

        state = get_job_status(minimal_config.work_path)
        assert state["status"] == "failed"
        assert "exit" in state["error"].lower() or "code" in state["error"].lower()

    def test_converter_exit_0_missing_output_fails(self, minimal_config: CompressionConfig, tmp_path: Path):
        config_path = str(tmp_path / "cfg.json")
        minimal_config.save(config_path)
        _save_job_state(minimal_config.work_path, _make_initial_state(minimal_config))

        def mock_compose(recipe_path):
            self._make_valid_composition(minimal_config.composed_path, recipe_path)

        def run_that_creates_nothing(argv, **kwargs):
            # Output directory exists but is empty — mimics converter that exits 0 on error
            Path(minimal_config.compressed_path).mkdir(parents=True, exist_ok=True)
            r = SimpleNamespace()
            r.returncode = 0
            return r

        with patch("llmtraits.compression._compose_in_worker", side_effect=mock_compose):
            with patch("subprocess.run", side_effect=run_that_creates_nothing):
                with patch("llmtraits.compression.run_preflight", return_value=(True, "ok")):
                    try:
                        run_worker(config_path)
                    except SystemExit:
                        pass

        state = get_job_status(minimal_config.work_path)
        assert state["status"] == "failed"
        assert "validation" in state["error"].lower() or "invalid" in state["error"].lower()

    def test_compose_already_done_skips_compose(self, minimal_config: CompressionConfig, tmp_path: Path):
        """If composition is already verified complete, compose is skipped."""
        # Pre-create a fully valid composed directory
        with open(minimal_config.recipe_path, "r", encoding="utf-8") as fh:
            recipe = json.load(fh)
        _make_composed_dir(Path(minimal_config.composed_path), recipe)

        config_path = str(tmp_path / "cfg.json")
        minimal_config.save(config_path)
        _save_job_state(minimal_config.work_path, _make_initial_state(minimal_config))

        compose_called = []

        def mock_compose(recipe_path):
            compose_called.append(True)
            raise AssertionError("compose should not be called")

        def mock_run(argv, **kwargs):
            self._make_valid_output(minimal_config.compressed_path)
            r = SimpleNamespace()
            r.returncode = 0
            return r

        with patch("llmtraits.compression._compose_in_worker", side_effect=mock_compose):
            with patch("subprocess.run", side_effect=mock_run):
                with patch("llmtraits.compression.run_preflight", return_value=(True, "ok")):
                    try:
                        run_worker(config_path)
                    except SystemExit:
                        pass

        assert not compose_called
        state = get_job_status(minimal_config.work_path)
        assert state["status"] == "completed"

    def test_wrong_recipe_in_composed_dir_causes_failure(
        self, minimal_config: CompressionConfig, tmp_path: Path
    ):
        """Composed dir with a different recipe's resolved_recipe.json → job fails."""
        # Create composed dir with a DIFFERENT recipe
        different_recipe = {
            "schema": "svd_recipe", "version": "1.0",
            "analysis_dir": str(tmp_path),
            "departure_label": "OtherModel",
            "departure_path": str(tmp_path / "other_departure"),
            "output_path": str(tmp_path / "composed"),
            "dtype": "float16", "missing_policy": "error", "adjustments": [],
        }
        _make_composed_dir(Path(minimal_config.composed_path), different_recipe)

        config_path = str(tmp_path / "cfg.json")
        minimal_config.save(config_path)
        _save_job_state(minimal_config.work_path, _make_initial_state(minimal_config))

        with patch("llmtraits.compression._compose_in_worker"):
            with patch("llmtraits.compression.run_preflight", return_value=(True, "ok")):
                try:
                    run_worker(config_path)
                except SystemExit:
                    pass

        state = get_job_status(minimal_config.work_path)
        assert state["status"] == "failed"
        assert "identity" in state["error"].lower() or "recipe" in state["error"].lower() or "mismatch" in state["error"].lower()

    def test_resume_used_when_args_json_matches(self, minimal_config: CompressionConfig, tmp_path: Path):
        """When a matching args.json exists in work dir, -resume argv is used."""
        with open(minimal_config.recipe_path, "r", encoding="utf-8") as fh:
            recipe = json.load(fh)
        _make_composed_dir(Path(minimal_config.composed_path), recipe)

        Path(minimal_config.work_path).mkdir(parents=True, exist_ok=True)
        args_json = {
            "bpw": minimal_config.bpw,
            "input_dir": minimal_config.composed_path,
            "out_dir": minimal_config.compressed_path,
        }
        (Path(minimal_config.work_path) / "args.json").write_text(json.dumps(args_json))

        config_path = str(tmp_path / "cfg.json")
        minimal_config.save(config_path)
        _save_job_state(minimal_config.work_path, _make_initial_state(minimal_config))

        captured_argv = []

        def mock_run(argv, **kwargs):
            captured_argv.extend(argv)
            self._make_valid_output(minimal_config.compressed_path)
            r = SimpleNamespace()
            r.returncode = 0
            return r

        with patch("llmtraits.compression._compose_in_worker"):
            with patch("subprocess.run", side_effect=mock_run):
                with patch("llmtraits.compression.run_preflight", return_value=(True, "ok")):
                    try:
                        run_worker(config_path)
                    except SystemExit:
                        pass

        assert "-resume" in captured_argv

    def test_mismatched_args_json_causes_failure(self, minimal_config: CompressionConfig, tmp_path: Path):
        """A work dir with mismatched bpw must fail rather than proceed."""
        with open(minimal_config.recipe_path, "r", encoding="utf-8") as fh:
            recipe = json.load(fh)
        _make_composed_dir(Path(minimal_config.composed_path), recipe)

        Path(minimal_config.work_path).mkdir(parents=True, exist_ok=True)
        # Only 'bpw' → partial identity (no in_dir/out_dir) → ValueError
        (Path(minimal_config.work_path) / "args.json").write_text(
            json.dumps({"bpw": 4.0})  # conflicts with config bpw=5.5
        )

        config_path = str(tmp_path / "cfg.json")
        minimal_config.save(config_path)
        _save_job_state(minimal_config.work_path, _make_initial_state(minimal_config))

        with patch("llmtraits.compression._compose_in_worker"):
            with patch("llmtraits.compression.run_preflight", return_value=(True, "ok")):
                try:
                    run_worker(config_path)
                except SystemExit:
                    pass

        state = get_job_status(minimal_config.work_path)
        assert state["status"] == "failed"
        assert "bpw" in state["error"].lower() or "mismatch" in state["error"].lower() or "partial" in state["error"].lower()


# ─── GUI importability ────────────────────────────────────────────────────────

class TestGuiImport:
    def test_gui_module_importable(self):
        """gui.py must be importable without Streamlit installed/running."""
        import importlib
        # If already imported, just re-check attributes
        import llmtraits.gui as gui_mod
        assert hasattr(gui_mod, "main")
        assert hasattr(gui_mod, "_render_compression_section")


# ─── CLI help ─────────────────────────────────────────────────────────────────

class TestCliHelp:
    def test_main_help_exit(self):
        from llmtraits.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_compressed_job_in_parser(self):
        from llmtraits.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["compressed-job", "--help"])

    def test_compression_preflight_in_parser(self):
        from llmtraits.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["compression-preflight", "--help"])

    def test_compression_status_in_parser(self):
        from llmtraits.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["compression-status", "--help"])

    def test_cli_main_via_module(self):
        """Ensure ``python -m llmtraits --help`` exits cleanly."""
        result = subprocess.run(
            [sys.executable, "-m", "llmtraits", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "compressed-job" in result.stdout or "compression-preflight" in result.stdout


# ─── compression-preflight CLI command ───────────────────────────────────────

class TestCompressionPreflightCommand:
    def test_preflight_command_ok(self, tmp_path: Path, fake_python: Path, fake_script: Path):
        def mock_run_preflight(config):
            return True, "All good"

        from llmtraits.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "compression-preflight",
            "--converter-python", str(fake_python),
            "--convert-script", str(fake_script),
        ])
        with patch("llmtraits.compression.run_preflight", return_value=(True, "All good")):
            # Should not raise
            args.func(args)

    def test_preflight_command_failure_exits_1(self, tmp_path: Path, fake_python: Path, fake_script: Path):
        from llmtraits.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "compression-preflight",
            "--converter-python", str(fake_python),
            "--convert-script", str(fake_script),
        ])
        with patch("llmtraits.compression.run_preflight", return_value=(False, "No torch")):
            with pytest.raises(SystemExit) as exc_info:
                args.func(args)
            assert exc_info.value.code == 1


# ─── Preflight disk check — composed drive ────────────────────────────────────

class TestPreflightDiskComposedDrive:
    """Verify that the composed_path drive is included in disk checks."""

    def test_composed_drive_insufficient_causes_failure(self, minimal_config: CompressionConfig, tmp_path: Path):
        """If the composed path drive has insufficient space, preflight must report failure."""
        # Mock all paths to exist, but disk_usage returns very little free space
        with patch("os.path.exists", return_value=True):
            with patch("shutil.disk_usage", return_value=(20 * 1024**3, 10 * 1024**3, 5 * 1024**3)):
                ok, msg = run_preflight(minimal_config)
        assert not ok
        assert "disk" in msg.lower() or "space" in msg.lower() or "gb" in msg.lower() or "drive" in msg.lower()

    def test_composed_drive_ample_does_not_block(self, minimal_config: CompressionConfig, tmp_path: Path):
        """With ample disk space, disk checks should not block preflight (other checks may)."""
        # We only care disk checks pass; other checks (Python/script) are expected to fail in test env
        with patch("shutil.disk_usage", return_value=(2000 * 1024**3, 100 * 1024**3, 1900 * 1024**3)):
            ok, msg = run_preflight(minimal_config)
        # The result may be False due to Python/converter issues, but NOT due to disk
        assert "space" not in msg.lower() or "disk" not in msg.lower() or ok


# ─── GUI helper tests ─────────────────────────────────────────────────────────

class TestGuiPreflightSnapshot:
    """Test the _preflight_config_snapshot helper from gui.py."""

    _COMMON = dict(
        departure_path="D:\\work\\models\\base",
        composed_path="D:\\work\\composed\\base_5.5bw",
        compressed_path="D:\\work\\compressed\\base_5.5bw",
        work_path="D:\\work\\compression_work\\base_5.5bw",
        bpw=5.5,
        cuda_devices="0",
        converter_python="C:\\venv\\python.exe",
        convert_script="C:\\venv\\convert.py",
        output_name="base_5.5bw",
    )

    def test_snapshot_changes_when_bpw_changes(self):
        from llmtraits.gui import _preflight_config_snapshot
        snap1 = _preflight_config_snapshot(**self._COMMON)
        snap2 = _preflight_config_snapshot(**{**self._COMMON, "bpw": 4.0})
        assert snap1 != snap2

    def test_snapshot_changes_when_departure_changes(self):
        from llmtraits.gui import _preflight_config_snapshot
        snap1 = _preflight_config_snapshot(**self._COMMON)
        snap2 = _preflight_config_snapshot(**{**self._COMMON, "departure_path": "D:\\work\\models\\different"})
        assert snap1 != snap2

    def test_identical_config_produces_same_snapshot(self):
        from llmtraits.gui import _preflight_config_snapshot
        assert _preflight_config_snapshot(**self._COMMON) == _preflight_config_snapshot(**self._COMMON)
