import ctypes
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


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
        return exit_code.value == 259  # STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(process)


def launch_svd(registry: str, output_dir: str) -> tuple[int, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "svd_job.json"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        pid = int(state.get("pid", 0))
        if pid and _pid_alive(pid):
            raise RuntimeError(f"SVD is already running with PID {pid}")

    progress_path = output / "svd_progress.json"
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as handle:
            if json.load(handle).get("phase") == "complete":
                raise RuntimeError("SVD analysis is already complete")

    log_path = output / "svd_worker.log"
    argv = [
        sys.executable,
        "-m",
        "llmtraits",
        "svd",
        "--registry",
        os.path.abspath(registry),
        "--output-dir",
        os.path.abspath(output_dir),
    ]
    with open(log_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"\n=== SVD worker launch {datetime.now(timezone.utc).isoformat()} ===\n"
        )
        log_handle.write(f"argv: {argv}\n")
        log_handle.flush()
        kwargs = (
            {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
            }
            if sys.platform == "win32"
            else {"start_new_session": True}
        )
        process = subprocess.Popen(
            argv,
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            **kwargs,
        )

    _atomic_json(
        state_path,
        {
            "pid": process.pid,
            "registry": os.path.abspath(registry),
            "output_dir": os.path.abspath(output_dir),
            "log_path": str(log_path),
            "launched_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return process.pid, str(log_path)


def svd_launch_command(args) -> None:
    pid, log_path = launch_svd(args.registry, args.output_dir)
    print(json.dumps({"pid": pid, "log_path": log_path}, indent=2))
