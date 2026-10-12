import argparse
import json
import logging
import os
from pathlib import Path

from .batch import batch_command
from .compose import compose_command
from .config import DEFAULT_BASE, DEFAULT_DTYPE, DEFAULT_OUTPUT, SCAN_DIRS
from .cross import cross_command
from .delta import DeltaComputer
from .manifest import load_manifest
from .scan import scan_command
from .svd import svd_command
from .svd_job import svd_launch_command
from .visualize import visualize_command


def _configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _delta_output_dir(base_output: str, target: str) -> str:
    return os.path.join(base_output, Path(target).name)


def delta_command(args) -> None:
    output_dir = args.output_dir or _delta_output_dir(DEFAULT_OUTPUT, args.target)
    computer = DeltaComputer(
        base_dir=args.base,
        target_dir=args.target,
        output_dir=output_dir,
        dtype=args.delta_dtype,
        force=args.force,
        skip_landscapes=args.skip_landscapes,
    )
    manifest = computer.run()
    print(json.dumps({"output_dir": output_dir, "n_tensors": len(manifest.tensors)}, indent=2))


def status_command(args) -> None:
    delta_dirs = args.delta_dirs or []
    for delta_dir in delta_dirs:
        manifest = load_manifest(delta_dir)
        if manifest is None:
            print(f"{delta_dir}: missing manifest")
            continue
        completed = sum(1 for shard in manifest.shards.values() if shard.status == "complete")
        print(
            f"{delta_dir}: {completed}/{len(manifest.shards)} shards complete, "
            f"{len(manifest.tensors)} tensors, "
            f"{len(manifest.incompatible_tensors)} incompatible"
        )


def compose_svd_command_wrapper(args):
    from .recipes import compose_svd_command

    compose_svd_command(args)


def route_preview_command(args):
    from .recipes import load_analysis
    from .routing import preview_routed_recipe, validate_routed_recipe

    with open(args.recipe, "r", encoding="utf-8") as handle:
        recipe = json.load(handle)
    analysis = load_analysis(recipe["analysis_dir"])
    validate_routed_recipe(recipe, analysis)
    summary = preview_routed_recipe(recipe)
    print(json.dumps(summary, indent=2, default=str))


def atlas_scan_command(args):
    from .atlas import scan_local_atlas

    result = scan_local_atlas(
        args.analysis,
        args.output_dir,
        block_size=args.block_size,
        checkpoint_blocks=args.checkpoint_blocks,
    )
    print(json.dumps(result, indent=2, default=str))


def atlas_launch_command(args):
    from .atlas import launch_local_atlas

    pid, log_path = launch_local_atlas(
        args.analysis,
        args.output_dir,
        block_size=args.block_size,
        checkpoint_blocks=args.checkpoint_blocks,
    )
    print(json.dumps({"pid": pid, "log_path": log_path}, indent=2))


def atlas_analyze_command(args):
    from .atlas_analysis import analyze_local_atlas

    result = analyze_local_atlas(
        args.atlas_dir,
        args.analysis,
        output_dir=args.output_dir,
        departure_label=args.departure_label,
        prior_recipe=args.prior_recipe,
    )
    print(json.dumps(result, indent=2, default=str))


def atlas_analyze_launch_command(args):
    from .atlas_analysis import launch_atlas_analysis

    pid, log_path = launch_atlas_analysis(
        args.atlas_dir,
        args.analysis,
        output_dir=args.output_dir,
        departure_label=args.departure_label,
        prior_recipe=args.prior_recipe,
    )
    print(json.dumps({"pid": pid, "log_path": log_path}, indent=2))


def gui_command(args):
    import subprocess
    import sys

    gui_module = str(Path(__file__).parent / "gui.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            gui_module,
            "--server.port",
            str(args.port),
            "--",
            "--analysis",
            args.analysis,
        ],
        check=True,
    )


def compressed_job_command(args) -> None:
    """Internal: foreground worker launched by launch_worker()."""
    from .compression import run_worker
    run_worker(args.config)


def compression_preflight_command(args) -> None:
    """Check ExLlamaV3 environment without starting any real work."""
    from .compression import (
        CompressionConfig,
        DEFAULT_BPW,
        DEFAULT_COMPOSED_ROOT,
        DEFAULT_COMPRESSED_ROOT,
        DEFAULT_CONVERT_SCRIPT,
        DEFAULT_CONVERTER_PYTHON,
        DEFAULT_CUDA_DEVICES,
        DEFAULT_WORK_ROOT,
        run_preflight,
    )

    config = CompressionConfig(
        recipe_path=args.recipe_path or "placeholder.json",
        model_name="preflight",
        output_name="preflight",
        composed_path=os.path.join(DEFAULT_COMPOSED_ROOT, "preflight"),
        compressed_path=os.path.join(DEFAULT_COMPRESSED_ROOT, "preflight"),
        work_path=os.path.join(DEFAULT_WORK_ROOT, "preflight"),
        converter_python=args.converter_python or DEFAULT_CONVERTER_PYTHON,
        convert_script=args.convert_script or DEFAULT_CONVERT_SCRIPT,
        bpw=float(args.bpw or DEFAULT_BPW),
        cuda_devices=args.cuda_devices or DEFAULT_CUDA_DEVICES,
    )
    # Skip recipe/path validation for pure environment check
    ok, msg = run_preflight(config, environment_only=True)
    print("PREFLIGHT OK" if ok else "PREFLIGHT FAILED")
    print(msg)
    if not ok:
        import sys
        sys.exit(1)


def compression_status_command(args) -> None:
    """Show status of a compression job by work directory."""
    from .compression import get_job_status, tail_log

    state = get_job_status(args.work_path)
    print(json.dumps(state, indent=2, default=str))
    if args.log and state.get("status") not in ("not_started",):
        output_name = state.get("output_name", "")
        jobs_root = args.jobs_root or str(
            __import__("pathlib").Path(args.work_path).parent
        )
        print("\n--- log tail ---")
        print(tail_log(jobs_root, output_name))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m llmtraits")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan")
    scan_parser.add_argument("--scan-dirs", nargs="*", default=None)
    scan_parser.add_argument("--base", default=DEFAULT_BASE)
    scan_parser.set_defaults(func=scan_command)

    delta_parser = subparsers.add_parser("delta")
    delta_parser.add_argument("target")
    delta_parser.add_argument("--base", default=DEFAULT_BASE)
    delta_parser.add_argument("--output-dir", default=None)
    delta_parser.add_argument("--delta-dtype", choices=["float16", "float32"], default=DEFAULT_DTYPE)
    delta_parser.add_argument("--skip-landscapes", action="store_true")
    delta_parser.add_argument("--force", action="store_true")
    delta_parser.set_defaults(func=delta_command)

    visualize_parser = subparsers.add_parser("visualize")
    visualize_parser.add_argument("delta_dir")
    visualize_parser.add_argument("--skip-landscapes", action="store_true")
    visualize_parser.add_argument("--force", action="store_true")
    visualize_parser.set_defaults(func=visualize_command)

    cross_parser = subparsers.add_parser("cross")
    cross_parser.add_argument("delta_dirs", nargs="+")
    cross_parser.add_argument("--output-dir", default=None)
    cross_parser.add_argument("--label", nargs="*", default=None)
    cross_parser.set_defaults(func=cross_command)

    svd_parser = subparsers.add_parser("svd")
    svd_parser.add_argument("--registry", required=True)
    svd_parser.add_argument("--output-dir", required=True)
    svd_parser.set_defaults(func=svd_command)

    svd_launch_parser = subparsers.add_parser("svd-launch")
    svd_launch_parser.add_argument("--registry", required=True)
    svd_launch_parser.add_argument("--output-dir", required=True)
    svd_launch_parser.set_defaults(func=svd_launch_command)

    compose_parser = subparsers.add_parser("compose")
    compose_parser.add_argument("--base", required=True)
    compose_parser.add_argument("--delta", action="append", required=True)
    compose_parser.add_argument("--output", required=True)
    compose_parser.add_argument(
        "--delta-dtype",
        choices=["source", "float16", "float32"],
        default=DEFAULT_DTYPE,
    )
    compose_parser.add_argument("--missing-policy", choices=["error", "skip", "zero"], default="error")
    compose_parser.set_defaults(func=compose_command)

    compose_svd_parser = subparsers.add_parser("compose-svd")
    compose_svd_parser.add_argument("--recipe", required=True, help="Path to recipe JSON file")
    compose_svd_parser.add_argument("--dry-run", action="store_true", help="Resolve recipe without composing")
    compose_svd_parser.set_defaults(func=compose_svd_command_wrapper)

    route_preview_parser = subparsers.add_parser(
        "route-preview",
        help="Build or resume a routed-recipe block allocation preview",
    )
    route_preview_parser.add_argument("--recipe", required=True)
    route_preview_parser.set_defaults(func=route_preview_command)

    atlas_scan_parser = subparsers.add_parser(
        "atlas-scan",
        help="Build or resume a recipe-independent block-local Gram atlas",
    )
    atlas_scan_parser.add_argument("--analysis", required=True)
    atlas_scan_parser.add_argument("--output-dir", required=True)
    atlas_scan_parser.add_argument("--block-size", type=int, default=262144)
    atlas_scan_parser.add_argument("--checkpoint-blocks", type=int, default=32)
    atlas_scan_parser.set_defaults(func=atlas_scan_command)

    atlas_launch_parser = subparsers.add_parser(
        "atlas-launch",
        help="Launch a detached recipe-independent atlas scan",
    )
    atlas_launch_parser.add_argument("--analysis", required=True)
    atlas_launch_parser.add_argument("--output-dir", required=True)
    atlas_launch_parser.add_argument("--block-size", type=int, default=262144)
    atlas_launch_parser.add_argument("--checkpoint-blocks", type=int, default=32)
    atlas_launch_parser.set_defaults(func=atlas_launch_command)

    atlas_analyze_parser = subparsers.add_parser(
        "atlas-analyze",
        help="Derive metrics, plots, candidates, and atlas-only route preplans",
    )
    atlas_analyze_parser.add_argument("--atlas-dir", required=True)
    atlas_analyze_parser.add_argument("--analysis", required=True)
    atlas_analyze_parser.add_argument("--output-dir", default=None)
    atlas_analyze_parser.add_argument(
        "--departure-label",
        required=True,
        help="Label of the departure model (the one whose identity is protected)",
    )
    atlas_analyze_parser.add_argument("--prior-recipe", default=None)
    atlas_analyze_parser.set_defaults(func=atlas_analyze_command)

    atlas_analyze_launch_parser = subparsers.add_parser(
        "atlas-analyze-launch",
        help="Launch detached derived-atlas analysis and preplanning",
    )
    atlas_analyze_launch_parser.add_argument("--atlas-dir", required=True)
    atlas_analyze_launch_parser.add_argument("--analysis", required=True)
    atlas_analyze_launch_parser.add_argument("--output-dir", default=None)
    atlas_analyze_launch_parser.add_argument(
        "--departure-label",
        required=True,
        help="Label of the departure model (the one whose identity is protected)",
    )
    atlas_analyze_launch_parser.add_argument("--prior-recipe", default=None)
    atlas_analyze_launch_parser.set_defaults(func=atlas_analyze_launch_command)

    gui_parser = subparsers.add_parser("gui")
    gui_parser.add_argument("--analysis", required=True, help="Path to analysis directory containing svd_analysis.npz")
    gui_parser.add_argument("--port", type=int, default=8501)
    gui_parser.set_defaults(func=gui_command)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("delta_dirs", nargs="*")
    status_parser.set_defaults(func=status_command)

    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("--scan-dirs", nargs="+", default=SCAN_DIRS)
    batch_parser.add_argument("--base", default=DEFAULT_BASE)
    batch_parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    batch_parser.add_argument("--delta-dtype", choices=["float16", "float32"], default=DEFAULT_DTYPE)
    batch_parser.add_argument("--skip-landscapes", action="store_true")
    batch_parser.add_argument("--skip-visuals", action="store_true")
    batch_parser.add_argument("--skip-cross", action="store_true")
    batch_parser.set_defaults(func=batch_command)

    cjob_parser = subparsers.add_parser(
        "compressed-job",
        help="[Internal] foreground worker: compose then compress",
    )
    cjob_parser.add_argument(
        "--config", required=True, help="Path to compression_config.json"
    )
    cjob_parser.set_defaults(func=compressed_job_command)

    preflight_parser = subparsers.add_parser(
        "compression-preflight",
        help="Check the ExLlamaV3 converter environment without launching any work",
    )
    preflight_parser.add_argument("--converter-python", default=None)
    preflight_parser.add_argument("--convert-script", default=None)
    preflight_parser.add_argument("--bpw", type=float, default=None)
    preflight_parser.add_argument("--cuda-devices", default=None)
    preflight_parser.add_argument("--recipe-path", default=None)
    preflight_parser.set_defaults(func=compression_preflight_command)

    cstatus_parser = subparsers.add_parser(
        "compression-status",
        help="Show status of a compression job by work directory",
    )
    cstatus_parser.add_argument("work_path", help="Per-job work directory")
    cstatus_parser.add_argument("--jobs-root", default=None)
    cstatus_parser.add_argument("--log", action="store_true", help="Show log tail")
    cstatus_parser.set_defaults(func=compression_status_command)

    return parser


def main(argv=None):
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0
