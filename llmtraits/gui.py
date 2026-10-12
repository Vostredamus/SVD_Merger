import argparse
import hashlib
import io
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_STREAMLIT_AVAILABLE = False
try:
    import streamlit as st

    _STREAMLIT_AVAILABLE = True
except ImportError:
    pass

from llmtraits.recipes import load_analysis
from llmtraits.routing import default_routed_recipe, load_route_preview
from llmtraits.steering import coefficient_table, steering_recipe_adjustment
from llmtraits.support import departure_overlap_table, load_block_diagnostics
from llmtraits.compression import (
    CompressionConfig,
    DEFAULT_BPW,
    DEFAULT_COMPOSED_ROOT,
    DEFAULT_COMPRESSED_ROOT,
    DEFAULT_CONVERT_SCRIPT,
    DEFAULT_CONVERTER_PYTHON,
    DEFAULT_CUDA_DEVICES,
    DEFAULT_JOBS_ROOT,
    DEFAULT_WORK_ROOT,
    _atomic_json_write,
    default_output_name,
    get_job_status,
    launch_worker,
    log_path_for,
    run_preflight,
    sanitize_folder_name,
    tail_log,
)


def _preflight_config_snapshot(
    departure_path: str,
    composed_path: str,
    compressed_path: str,
    work_path: str,
    bpw: float,
    cuda_devices: str,
    converter_python: str,
    convert_script: str,
    output_name: str,
) -> str:
    """
    Return a stable hex digest of all preflight-relevant config fields.

    If any field changes, the digest changes, allowing the GUI to detect a
    stale preflight result.
    """
    data = {
        "departure_path": departure_path,
        "composed_path": composed_path,
        "compressed_path": compressed_path,
        "work_path": work_path,
        "bpw": float(bpw),
        "cuda_devices": cuda_devices,
        "converter_python": converter_python,
        "convert_script": convert_script,
        "output_name": output_name,
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()


def _save_recipe_for_job(
    recipe: dict,
    jobs_root: str,
    output_name: str,
    departure_path: str,
    composed_path: str,
    analysis_dir: str,
) -> str:
    """
    Atomically save the full recipe (with departure_path, output_path, analysis_dir)
    to ``<jobs_root>/<output_name>/recipe.json``.

    Returns the recipe file path.  Used by both the preflight button and the
    launch button so the recipe and CompressionConfig can never diverge.
    """
    import pathlib as _pl
    jobs_dir = _pl.Path(jobs_root) / output_name
    jobs_dir.mkdir(parents=True, exist_ok=True)
    full_recipe = dict(recipe)
    full_recipe.pop("_analysis_cache", None)
    full_recipe["departure_path"] = departure_path.strip()
    full_recipe["output_path"] = composed_path
    full_recipe["analysis_dir"] = analysis_dir
    if (
        full_recipe.get("schema") == "routed_recipe"
        and not full_recipe.get("routing_path")
    ):
        full_recipe["routing_path"] = str(jobs_dir / "routing")
    recipe_save_path = str(jobs_dir / "recipe.json")
    _atomic_json_write(recipe_save_path, full_recipe)
    return recipe_save_path


def _parse_args():
    args_after_sep = (
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--port", type=int, default=8501)
    return parser.parse_known_args(args_after_sep)[0]


def _fig_bytes(fig):
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def _bar_chart(labels, values, title, color="#4472c4", ylim=None):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(np.arange(len(values)), values, color=color)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    return _fig_bytes(fig)


def _status_badge(status: str) -> str:
    icons = {
        "not_started": "⬜",
        "queued": "🕐",
        "preflight": "🔍",
        "composing": "🔄",
        "compressing": "⚙️",
        "completed": "✅",
        "failed": "❌",
        "stopped": "🛑",
    }
    return f"{icons.get(status, '❓')} **{status}**"


def _render_compression_section(analysis: dict, recipe: dict, args) -> None:
    """
    Render the 'Compression job' section of the GUI.

    Builds a CompressionConfig from the current field values, provides
    preflight and launch buttons, and shows live job status and log tail.

    Never launches a job on page load — only on explicit button click.
    """
    st.subheader("Compression job (compose → ExLlamaV3)")
    st.info(
        "This section creates a full composed model, then quantises it with "
        "ExLlamaV3. The composed model is **never auto-deleted**. Preflight uses "
        "static environment checks and does not import or execute ExLlamaV3."
    )

    departure_label: str = recipe.get("departure_label", "")
    departure_path = str(recipe.get("departure_path", "")).strip()

    # ── Model name ────────────────────────────────────────────────────────────
    default_model_name = sanitize_folder_name(
        recipe.get("model_name") or departure_label
    ) if departure_label else ""
    model_name = st.text_input(
        "Model name (used to build default paths)",
        key="cjob_model_name",
        value=default_model_name,
        help="Safe folder name for this model.  Auto-filled from departure label.",
    )

    # ── BPW ───────────────────────────────────────────────────────────────────
    bpw = st.number_input(
        "Bits per weight (BPW)",
        min_value=1.0,
        max_value=8.0,
        value=DEFAULT_BPW,
        step=0.5,
        key="cjob_bpw",
        help="ExLlamaV3 quantisation target.  5.5 is a good balance of quality/size.",
    )

    suggested_output = default_output_name(model_name or departure_label or "model", bpw)

    st.caption(f"Suggested output name: `{suggested_output}`")
    col_name, col_reset = st.columns([5, 1])
    with col_name:
        output_name = st.text_input(
            "Output name (compressed folder suffix)",
            key="cjob_output_name",
            placeholder=suggested_output,
            help="Leave blank to use the suggested name.  Edit freely; only folder-name "
            "characters are sanitised.",
        )
    with col_reset:
        st.write("")
        st.write("")
        if st.button("↺", key="cjob_reset_name", help="Reset to suggested name"):
            st.session_state["cjob_output_name"] = suggested_output
            st.rerun()

    effective_output_name_raw = (output_name or suggested_output).strip()
    # Sanitize and display the result if changed
    effective_output_name = sanitize_folder_name(effective_output_name_raw) if effective_output_name_raw else suggested_output
    if effective_output_name_raw and effective_output_name != effective_output_name_raw:
        st.warning(
            f"Output name sanitized: `{effective_output_name_raw}` → `{effective_output_name}`"
        )
    effective_model_name = (model_name or departure_label or "model").strip()

    # ── Path fields ───────────────────────────────────────────────────────────
    default_composed = str(
        __import__("pathlib").Path(DEFAULT_COMPOSED_ROOT) / effective_model_name
    ) if effective_model_name else DEFAULT_COMPOSED_ROOT

    default_compressed = str(
        __import__("pathlib").Path(DEFAULT_COMPRESSED_ROOT) / effective_output_name
    )
    default_work = str(
        __import__("pathlib").Path(DEFAULT_WORK_ROOT) / effective_output_name
    )

    with st.expander("Paths (click to edit)", expanded=False):
        composed_path = st.text_input(
            "Full composed model output",
            key="cjob_composed_path",
            value=default_composed,
        )
        compressed_path = st.text_input(
            "Compressed model output",
            key="cjob_compressed_path",
            value=default_compressed,
        )
        work_path = st.text_input(
            "Per-job converter work dir",
            key="cjob_work_path",
            value=default_work,
            help="ExLlamaV3 writes args.json / ckpt / qtensors here.  "
            "Each job must have its own directory.",
        )
        jobs_root = st.text_input(
            "Jobs root (recipe + log storage)",
            key="cjob_jobs_root",
            value=DEFAULT_JOBS_ROOT,
        )

    effective_composed = (composed_path or default_composed).strip()
    effective_compressed = (compressed_path or default_compressed).strip()
    effective_work = (work_path or default_work).strip()
    effective_jobs_root = (jobs_root or DEFAULT_JOBS_ROOT).strip()

    if recipe.get("schema") == "routed_recipe":
        route_path = str(
            recipe.get("routing_path")
            or (
                __import__("pathlib").Path(effective_jobs_root)
                / effective_output_name
                / "routing"
            )
        )
        progress_path = __import__("pathlib").Path(route_path) / "routing_progress.json"
        preview = load_route_preview(route_path)
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                phase = progress.get("phase", "unknown")
                current = progress.get(
                    "completed_tensors", progress.get("completed_shards", 0)
                )
                total = progress.get(
                    "tensor_total", progress.get("shard_total", 0)
                )
                st.info(
                    f"Routing phase: **{phase}**"
                    + (f" — {current}/{total}" if total else "")
                )
            except (OSError, json.JSONDecodeError):
                st.warning(f"Routing progress is unreadable: `{progress_path}`")
        if preview is not None:
            st.caption(
                f"Routing preview complete: {preview.get('accepted', 0):,} accepted "
                f"blocks; C4 beta {preview.get('c4_beta', 0.0):.6g}."
            )
            winner_counts = preview.get("winner_counts", {})
            if winner_counts:
                st.dataframe(
                    [
                        {"donor": donor, "winning blocks": count}
                        for donor, count in winner_counts.items()
                    ],
                    hide_index=True,
                    width="stretch",
                )

    # ── Converter settings ────────────────────────────────────────────────────
    with st.expander("Converter settings", expanded=False):
        cuda_devices = st.text_input(
            "CUDA_VISIBLE_DEVICES / converter devices (-d)",
            key="cjob_cuda_devices",
            value=DEFAULT_CUDA_DEVICES,
        )
        converter_python = st.text_input(
            "Converter Python executable",
            key="cjob_converter_python",
            value=DEFAULT_CONVERTER_PYTHON,
        )
        convert_script = st.text_input(
            "ExLlamaV3 convert.py path",
            key="cjob_convert_script",
            value=DEFAULT_CONVERT_SCRIPT,
        )

    # ── Build current config for preflight / launch ───────────────────────────
    def _current_config(recipe_save_path: str) -> CompressionConfig:
        return CompressionConfig(
            recipe_path=recipe_save_path,
            model_name=effective_model_name,
            output_name=effective_output_name,
            composed_path=effective_composed,
            compressed_path=effective_compressed,
            work_path=effective_work,
            jobs_root=effective_jobs_root,
            converter_python=converter_python or DEFAULT_CONVERTER_PYTHON,
            convert_script=convert_script or DEFAULT_CONVERT_SCRIPT,
            bpw=bpw,
            cuda_devices=cuda_devices or DEFAULT_CUDA_DEVICES,
        )

    # Compute current config snapshot for staleness detection
    _current_snapshot = _preflight_config_snapshot(
        departure_path=departure_path,
        composed_path=effective_composed,
        compressed_path=effective_compressed,
        work_path=effective_work,
        bpw=bpw,
        cuda_devices=cuda_devices or DEFAULT_CUDA_DEVICES,
        converter_python=converter_python or DEFAULT_CONVERTER_PYTHON,
        convert_script=convert_script or DEFAULT_CONVERT_SCRIPT,
        output_name=effective_output_name,
    )

    # ── Preflight staleness check ─────────────────────────────────────────────
    _saved_snapshot = st.session_state.get("cjob_preflight_snapshot")
    if _saved_snapshot is not None and _saved_snapshot != _current_snapshot:
        st.session_state.pop("cjob_preflight_result", None)
        st.session_state.pop("cjob_preflight_snapshot", None)
        st.info("⚠️ Configuration changed — re-run preflight check before launching.")

    # ── Preflight check button ────────────────────────────────────────────────
    st.write("---")
    if st.button("🔍 Check ExLlamaV3 environment", key="cjob_preflight"):
        if not departure_path.strip():
            st.error("⚠️ Departure model path is required before running preflight.")
        else:
            # Save recipe atomically before preflight so recipe_path exists
            recipe_save_path = _save_recipe_for_job(
                recipe, effective_jobs_root, effective_output_name,
                departure_path, effective_composed, args.analysis,
            )
            cfg = _current_config(recipe_save_path)
            with st.spinner("Running preflight…"):
                ok, msg = run_preflight(cfg)
            st.session_state["cjob_preflight_result"] = (ok, msg)
            st.session_state["cjob_preflight_snapshot"] = _current_snapshot

    pf_result = st.session_state.get("cjob_preflight_result")
    if pf_result is not None:
        ok, msg = pf_result
        if ok:
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ Preflight failed:\n\n{msg}")

    # ── Launch button ─────────────────────────────────────────────────────────
    preflight_ok = pf_result is not None and pf_result[0]
    fields_ok = bool(departure_path.strip() and effective_output_name)
    can_launch = preflight_ok and fields_ok

    if not preflight_ok:
        st.caption("Run a successful preflight check before launching.")
    if not departure_path.strip():
        st.caption("⚠️ Departure model path is required.")

    launch_label = "🚀 Start composed + compressed model"
    if st.button(launch_label, key="cjob_launch", disabled=not can_launch):
        # Save recipe atomically via the same helper used by preflight
        recipe_save_path = _save_recipe_for_job(
            recipe, effective_jobs_root, effective_output_name,
            departure_path, effective_composed, args.analysis,
        )
        cfg = _current_config(recipe_save_path)

        # Guard against duplicate launch
        state = get_job_status(effective_work)
        pid = state.get("pid")
        status = state.get("status", "")
        if status in ("queued", "preflight", "composing", "compressing") and state.get("pid_alive"):
            st.error(
                f"Job is already running (PID {pid}, status={status}).\n"
                f"Use `Stop-Process -Id {pid}` in PowerShell to stop it."
            )
        elif status == "completed":
            st.error(
                "Job has already completed. Delete the work dir and compressed output to rerun."
            )
        else:
            try:
                worker_pid, config_path = launch_worker(cfg)
                st.success(
                    f"✅ Worker launched!  PID={worker_pid}\n\n"
                    f"Config: `{config_path}`\n\n"
                    f"To stop:  `Stop-Process -Id {worker_pid}`"
                )
                st.session_state["cjob_last_launched_pid"] = worker_pid
            except Exception as exc:
                st.error(f"Launch failed: {exc}")

    # ── Status section ────────────────────────────────────────────────────────
    st.write("---")
    st.subheader("Job status")

    col_refresh, col_log = st.columns([2, 8])
    with col_refresh:
        if st.button("🔄 Refresh", key="cjob_refresh"):
            st.rerun()

    state = get_job_status(effective_work)
    status = state.get("status", "not_started")

    st.write(_status_badge(status))

    if state.get("pid"):
        pid = state["pid"]
        alive = state.get("pid_alive", False)
        pid_line = f"PID **{pid}** — {'🟢 alive' if alive else '⚫ not running'}"
        st.write(pid_line)
        if alive and status not in ("completed",):
            st.code(f"Stop-Process -Id {pid}", language="powershell")
            st.caption(
                "⚠️ Stopping during conversion relies on ExLlamaV3 work checkpoints.  "
                "The converter can be resumed from the work directory."
            )

    # Timestamps
    ts_fields = [
        ("queued_at", "Queued"),
        ("compose_started_at", "Compose started"),
        ("compose_completed_at", "Compose completed"),
        ("compress_started_at", "Compress started"),
        ("compress_completed_at", "Compress completed"),
        ("completed_at", "Completed"),
        ("failed_at", "Failed / stopped"),
    ]
    rows = [
        {"event": label, "timestamp": state[key]}
        for key, label in ts_fields
        if state.get(key)
    ]
    if rows:
        st.table(rows)

    # Paths
    if state.get("composed_path"):
        st.write(f"**Composed model:** `{state['composed_path']}`")
    if state.get("compressed_path"):
        st.write(f"**Compressed model:** `{state['compressed_path']}`")
    if state.get("output_size_bytes"):
        gb = state["output_size_bytes"] / 1024 ** 3
        st.write(f"**Output size:** {gb:.2f} GB")

    if state.get("error"):
        st.error(f"Error:\n\n```\n{state['error']}\n```")

    # Resume guidance
    if status in ("failed", "stopped") and state.get("pid") and not state.get("pid_alive", True):
        st.info(
            "To resume: click **🚀 Start composed + compressed model** again.  "
            "Completed composition will be skipped; the converter will attempt to resume "
            "from the work directory checkpoint."
        )

    # Log tail
    log_text = tail_log(effective_jobs_root, effective_output_name)
    with st.expander("📋 Worker log (last 60 lines)", expanded=status not in ("not_started", "completed")):
        st.code(log_text, language="text")
        log_fp = log_path_for(effective_jobs_root, effective_output_name)
        st.caption(f"Full log: `{log_fp}`")

    # Manual command equivalent
    with st.expander("🖥️ Equivalent manual commands", expanded=False):
        _disp_py = converter_python or DEFAULT_CONVERTER_PYTHON
        _disp_script = convert_script or DEFAULT_CONVERT_SCRIPT
        initial_argv = [
            _disp_py, _disp_script,
            "-i", effective_composed,
            "-o", effective_compressed,
            "-w", effective_work,
            "-b", str(bpw),
            "-d", cuda_devices or DEFAULT_CUDA_DEVICES,
            "-v",
        ]
        resume_argv = [
            _disp_py, _disp_script,
            "-resume",
            "-w", effective_work,
            "-v",
        ]
        st.write("**Initial conversion run:**")
        st.code(" ".join(f'"{a}"' if " " in a else a for a in initial_argv), language="powershell")
        st.write("**Resume after interruption:**")
        st.code(" ".join(f'"{a}"' if " " in a else a for a in resume_argv), language="powershell")
        st.caption(
            f"Set `$env:CUDA_VISIBLE_DEVICES = '{cuda_devices or DEFAULT_CUDA_DEVICES}'` "
            f"before running, or prefix the command."
        )


def _render_donor_tab(analysis, block_diag, departure_label, departure_idx, labels):
    st.subheader("Donor transfer")
    st.caption(
        "Move a component's coefficient toward a donor model. Global scope blends "
        "the whole component; novel-support scope applies the component delta only "
        "where the departure model barely moved."
    )
    adjustments = []
    n_components = analysis["n_components"]
    for k in range(n_components):
        current = float(analysis["model_coefficients"][departure_idx, k])
        with st.expander(f"Component {k + 1}  (current coeff `{current:.4f}`)", expanded=False):
            enabled = st.checkbox("Enable", key=f"don_enable_{k}")
            if not enabled:
                st.image(
                    _bar_chart(
                        labels,
                        analysis["model_coefficients"][:, k],
                        f"Model loadings for component {k + 1}",
                    ),
                    use_container_width=True,
                )
                continue
            donor_label = st.selectbox(
                "Donor model", labels, key=f"don_donor_{k}"
            )
            gamma = st.slider("Gamma", 0.0, 2.0, 1.0, 0.05, key=f"don_gamma_{k}")
            scope = st.radio(
                "Scope", ["global", "novel_support"], horizontal=True, key=f"don_scope_{k}"
            )
            donor_idx = labels.index(donor_label)
            donor_coeff = float(analysis["model_coefficients"][donor_idx, k])
            proposed = current + gamma * (donor_coeff - current)
            st.write(f"Proposed new coefficient: `{proposed:.6f}`")
            if gamma > 1.0:
                st.warning("Extrapolation risk: gamma > 1.0")
            adjustment = {"component": k + 1, "donor_label": donor_label, "gamma": float(gamma)}
            if scope == "novel_support":
                nr = st.number_input(
                    "Novelty ratio", 0.0, 1.0, 0.10, 0.01, key=f"don_nr_{k}"
                )
                st.warning(
                    "Novel-support scope only transfers blocks where the departure "
                    "model barely moved. Blocks are not literal concepts."
                )
                ack = st.checkbox(
                    "I understand novel-support is heuristic", key=f"don_ack_{k}"
                )
                if not ack:
                    st.info("Acknowledge to include this novel-support adjustment.")
                    continue
                adjustment["scope"] = "novel_support"
                adjustment["novelty_ratio"] = float(nr)
            if gamma > 0.0:
                adjustments.append(adjustment)
            st.image(
                _bar_chart(
                    labels,
                    analysis["model_coefficients"][:, k],
                    f"Model loadings for component {k + 1}",
                ),
                use_container_width=True,
            )
    st.session_state["don_adjustments"] = adjustments
    st.subheader("Donor adjustments")
    st.table(adjustments if adjustments else [{"note": "No donor adjustments"}])
    return adjustments


def _render_steering_tab(analysis, departure_label, departure_idx, labels):
    st.subheader("Direct steering")
    st.caption(
        "Set component coordinates directly by delta, absolute target, or "
        "multiplier. Produces target_coefficient adjustments."
    )
    n_components = analysis["n_components"]
    overrides = st.session_state.get("steer_overrides", {})
    component = st.selectbox(
        "Component", list(range(1, n_components + 1)), key="steer_component"
    )
    k = component - 1
    current = float(analysis["model_coefficients"][departure_idx, k])
    st.write(f"Starting coefficient: `{current:.6f}`")
    mode = st.radio(
        "Mode", ["delta", "target", "multiplier"], horizontal=True, key="steer_mode"
    )
    default_value = {"delta": 0.0, "target": current, "multiplier": 1.0}[mode]
    value = st.number_input("Value", value=float(default_value), step=0.05, key="steer_value")
    adjustment = steering_recipe_adjustment(component, current, mode, float(value))
    if adjustment is not None:
        proposed = adjustment["target_coefficient"]
        st.write(f"Proposed coefficient: `{proposed:.6f}`")
    else:
        st.write("No change.")
    col_apply, col_clear = st.columns(2)
    with col_apply:
        if st.button("Apply override", key="steer_apply"):
            if adjustment is not None:
                overrides[component] = adjustment["target_coefficient"]
            else:
                overrides.pop(component, None)
            st.session_state["steer_overrides"] = overrides
            st.rerun()
    with col_clear:
        if st.button("Clear all overrides", key="steer_clear"):
            st.session_state["steer_overrides"] = {}
            st.rerun()

    st.subheader("Coefficient table")
    st.table(coefficient_table(analysis, departure_label, overrides))
    adjustments = [
        {"component": comp, "target_coefficient": float(target)}
        for comp, target in sorted(overrides.items())
    ]
    st.session_state["steer_adjustments"] = adjustments
    return adjustments


def _render_overlap_tab(analysis, block_diag, departure_label, departure_idx, labels):
    st.subheader("Component overlap & islands")
    st.caption(
        "Departure overlap describes how the departure model's delta projects onto "
        "each SVD component. Components are derived from all models, so high novel "
        "support does not mean the component is independent of the departure model."
    )
    overlap_rows = departure_overlap_table(
        analysis, block_diag, departure_label
    )
    display_rows = []
    for row in overlap_rows:
        display_rows.append(
            {
                "component": row["component"],
                "coefficient": row["coefficient"],
                "departure from BASE (energy)": row["departure_energy_share"],
                "explained ensemble energy": row["explained_ensemble_energy"],
                "shared support": row["shared_support_fraction"],
                "novel support": row["novel_support_fraction"],
                "sign conflict": row["sign_conflict_fraction"],
                "dominant layers": row["dominant_layers"],
                "dominant architecture": row["dominant_arch"],
            }
        )
    st.dataframe(
        display_rows,
        column_config={
            "departure from BASE (energy)": st.column_config.NumberColumn(
                format="percent"
            ),
            "explained ensemble energy": st.column_config.NumberColumn(
                format="percent"
            ),
            "shared support": st.column_config.NumberColumn(format="percent"),
            "novel support": st.column_config.NumberColumn(format="percent"),
            "sign conflict": st.column_config.NumberColumn(format="percent"),
        },
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        """
**Column guide**

- **component:** one global pattern of weight changes found across the model family.
- **coefficient:** the signed amount of that component in the departure model. It is a coordinate, not a percentage.
- **departure from BASE (energy):** how much of the departure model's total change from BASE is explained by this component. This column sums to approximately 100%.
- **explained ensemble energy:** how much of all analyzed models' combined change is represented by this component. This column also sums to approximately 100%.
- **shared support:** component energy in material blocks where the departure model also changed substantially.
- **novel support:** component energy in material blocks where the departure model remained relatively close to BASE.
- **sign conflict:** component energy in material blocks where the local departure change points against the component.
- **dominant layers / architecture:** where the recorded novel-support energy is concentrated.

Novel support is a statistical candidate for lower-interference transfer, not proof that a block contains an isolated behavior or skill.
"""
    )

    if block_diag is None:
        st.info(
            "Run SVD analysis to generate block diagnostics "
            "(svd_block_diagnostics.npz not found)."
        )
        return
    st.subheader("Novel support heatmap")
    matrix = np.asarray(block_diag["novel_support_frac"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f"svd_{i + 1}" for i in range(matrix.shape[1])], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax)
    st.image(_fig_bytes(fig), use_container_width=True)

    st.subheader("Top novel blocks")
    rows = [
        row
        for row in block_diag.get("top_blocks", [])
        if row.get("departure_label") == departure_label
    ]
    rows = sorted(rows, key=lambda r: r["comp_block_energy"], reverse=True)[:50]
    if rows:
        st.table(rows)
    else:
        st.write("No novel blocks recorded for this departure model.")
    st.caption(
        "Blocks are flat parameter windows, not literal semantic concepts. "
        "Treat them as localized statistical evidence only."
    )


def _parse_int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _render_routed_tab(
    analysis, args, departure_label, departure_idx, inferred_departure_path
):
    st.subheader("Protected routed composition")
    st.caption(
        "Build a donor-exclusive, block-routed mixture while protecting the "
        "departure model's strongest changes and its selected SVD component. "
        "Routing preview and composition are resumable."
    )
    defaults = default_routed_recipe(
        analysis,
        departure_label,
        inferred_departure_path,
        "",
    )
    defaults["analysis_dir"] = args.analysis
    defaults["model_name"] = "Routed-Model-v1"

    routing = dict(defaults["routing"])
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        routing["protection_energy_fraction"] = float(
            st.slider(
                "Protect departure energy",
                0.0,
                1.0,
                float(routing["protection_energy_fraction"]),
                0.01,
                key="route_protection",
            )
        )
        routing["protected_component"] = int(
            st.number_input(
                "Protected SVD component",
                1,
                int(analysis["n_components"]),
                int(routing["protected_component"]),
                key="route_component",
            )
        )
    with col_b:
        routing["max_rms_ratio"] = float(
            st.number_input(
                "Maximum update / weight RMS",
                min_value=0.0001,
                max_value=0.05,
                value=float(routing["max_rms_ratio"]),
                step=0.0005,
                format="%.4f",
                key="route_rms_cap",
            )
        )
        routing["conflict_threshold"] = float(
            st.slider(
                "Minimum local cosine",
                -1.0,
                1.0,
                float(routing["conflict_threshold"]),
                0.05,
                key="route_conflict",
            )
        )
    with col_c:
        routing["winner_only"] = bool(
            st.checkbox(
                "Winner-only block allocation",
                value=bool(routing["winner_only"]),
                key="route_winner_only",
            )
        )
        routing["require_coarse_stability"] = bool(
            st.checkbox(
                "Require 1M/262K route stability",
                value=bool(routing["require_coarse_stability"]),
                key="route_stability",
            )
        )

    donors = []
    for index, donor_default in enumerate(defaults["donors"]):
        label = donor_default["donor_label"]
        with st.expander(f"{label} routed contribution", expanded=True):
            enabled = st.checkbox(
                "Enable donor", value=True, key=f"route_enable_{index}"
            )
            alpha = st.number_input(
                "Strength",
                min_value=0.0,
                max_value=2.0,
                value=float(donor_default["alpha"]),
                step=0.005,
                format="%.3f",
                key=f"route_alpha_{index}",
            )
            components_text = st.text_input(
                "SVD affinity components",
                value=", ".join(str(value) for value in donor_default["components"]),
                key=f"route_components_{index}",
            )
            architectures = st.multiselect(
                "Allowed tensor types",
                [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                default=list(donor_default.get("architectures", [])),
                key=f"route_arch_{index}",
            )
            layers_text = st.text_input(
                "Restrict to layers (blank = all)",
                value=", ".join(
                    str(value) for value in donor_default.get("layers", [])
                ),
                key=f"route_layers_{index}",
            )
            require_novel = st.checkbox(
                "Require novel support",
                value=bool(donor_default.get("require_novel", False)),
                key=f"route_novel_{index}",
            )
            novelty_ratio = st.number_input(
                "Novelty ratio",
                min_value=0.0,
                max_value=1.0,
                value=float(
                    donor_default.get(
                        "novelty_ratio", routing.get("novelty_ratio", 0.10)
                    )
                ),
                step=0.01,
                key=f"route_novelty_ratio_{index}",
            )
            if enabled and alpha > 0:
                donor = dict(donor_default)
                donor["alpha"] = float(alpha)
                donor["components"] = _parse_int_list(components_text)
                donor["architectures"] = list(architectures)
                donor["layers"] = _parse_int_list(layers_text)
                donor["require_novel"] = bool(require_novel)
                donor["novelty_ratio"] = float(novelty_ratio)
                donors.append(donor)

    departure_path = st.text_input(
        "Departure model path",
        key="routed_departure_path",
        value=inferred_departure_path,
        help="The complete departure checkpoint used as the starting model.",
    )
    recipe = dict(defaults)
    recipe["departure_path"] = departure_path
    recipe["donors"] = donors
    recipe["routing"] = routing

    st.subheader("Resolved routed recipe")
    st.table(
        [
            {
                "order": index + 1,
                "donor": donor["donor_label"],
                "strength": donor["alpha"],
                "components": ", ".join(map(str, donor["components"])),
                "layers": ", ".join(map(str, donor.get("layers", []))) or "all",
                "tensor types": ", ".join(donor.get("architectures", [])),
                "novel only": donor.get("require_novel", False),
            }
            for index, donor in enumerate(donors)
        ]
        or [{"note": "No routed donors enabled"}]
    )
    recipe_json = json.dumps(recipe, indent=2)
    st.code(recipe_json, language="json")
    st.download_button(
        "Download routed recipe JSON",
        recipe_json.encode("utf-8"),
        file_name="routed_recipe.json",
        mime="application/json",
    )
    st.caption(
        "The final output and routing paths are inserted when preflight saves "
        "the job recipe. Use the standalone route-preview command only with a "
        "saved recipe whose output_path is populated."
    )
    st.session_state["routed_recipe"] = recipe
    st.info(
        "The routed recipe is ready. Open **Build & compress**, select "
        "**Routed composition**, then run preflight and launch."
    )
    return recipe


def _render_build_tab(analysis, args, departure_label, departure_idx, inferred_departure_path):
    st.subheader("Build & compress")
    donor_adjustments = st.session_state.get("don_adjustments", [])
    steer_adjustments = st.session_state.get("steer_adjustments", [])

    source = st.radio(
        "Active recipe source",
        ["Donor transfer", "Direct steering", "Combined", "Routed composition"],
        horizontal=True,
        key="build_source",
    )
    previous_source = st.session_state.get("build_previous_source")
    if previous_source != source:
        for key in (
            "cjob_model_name",
            "cjob_output_name",
            "cjob_composed_path",
            "cjob_compressed_path",
            "cjob_work_path",
            "cjob_preflight_result",
            "cjob_preflight_snapshot",
        ):
            st.session_state.pop(key, None)
        st.session_state["build_previous_source"] = source
    if source == "Donor transfer":
        adjustments = list(donor_adjustments)
    elif source == "Direct steering":
        adjustments = list(steer_adjustments)
    elif source == "Combined":
        seen = {a["component"] for a in donor_adjustments}
        combined = list(donor_adjustments)
        for a in steer_adjustments:
            if a["component"] in seen:
                st.warning(
                    f"Component {a['component']} appears in both sources; "
                    "using the donor-transfer version."
                )
                continue
            combined.append(a)
        adjustments = combined
    else:
        adjustments = []

    st.subheader("Recipe adjustments")
    if source == "Routed composition":
        routed_recipe = st.session_state.get("routed_recipe")
        routed_donors = routed_recipe.get("donors", []) if routed_recipe else []
        st.table(
            [
                {
                    "donor": donor["donor_label"],
                    "strength": donor["alpha"],
                    "components": ", ".join(map(str, donor.get("components", []))),
                }
                for donor in routed_donors
            ]
            or [{"note": "Configure donors in the Routed composition tab first"}]
        )
    else:
        st.table(adjustments if adjustments else [{"note": "No adjustments selected"}])

    departure_path = st.text_input(
        "Departure model path",
        key="build_departure_path",
        value=inferred_departure_path,
        help="Auto-inferred from analysis target_paths; edit if needed.",
    )

    if source == "Routed composition":
        recipe = dict(
            st.session_state.get("routed_recipe")
            or default_routed_recipe(
                analysis, departure_label, departure_path, ""
            )
        )
        recipe["analysis_dir"] = args.analysis
        recipe["departure_label"] = departure_label
        recipe["departure_path"] = departure_path
        recipe["output_path"] = ""
        recipe["routing_path"] = ""
    else:
        recipe = {
            "schema": "svd_recipe",
            "version": "1.0",
            "analysis_dir": args.analysis,
            "departure_label": departure_label,
            "departure_path": departure_path,
            "output_path": "",
            "dtype": "float16",
            "missing_policy": "error",
            "adjustments": adjustments,
        }
    recipe_json = json.dumps(recipe, indent=2)
    st.subheader("Recipe JSON")
    st.code(recipe_json, language="json")
    st.download_button(
        "Download recipe JSON",
        recipe_json.encode("utf-8"),
        file_name=(
            "routed_recipe.json"
            if source == "Routed composition"
            else "svd_recipe.json"
        ),
        mime="application/json",
    )
    st.code("python -m llmtraits compose-svd --recipe path/to/recipe.json", language="powershell")

    st.write("---")
    with st.expander("🗜️ Compression job (compose → ExLlamaV3 EXL3)", expanded=True):
        _render_compression_section(analysis, recipe, args)
    return recipe


def main():
    if not _STREAMLIT_AVAILABLE:
        raise RuntimeError("streamlit is required for the GUI")

    args = _parse_args()
    analysis = load_analysis(args.analysis)
    block_diag = load_block_diagnostics(args.analysis)
    labels = analysis["labels"]

    default_departure = labels[0]
    departure_label = st.sidebar.selectbox(
        "Departure model", labels, index=labels.index(default_departure)
    )
    departure_idx = labels.index(departure_label)

    target_paths = analysis.get("target_paths", [])
    inferred_departure_path = ""
    if target_paths and departure_idx < len(target_paths):
        inferred_departure_path = target_paths[departure_idx] or ""

    st.title("LLMTraits SVD recipe builder")

    coeff_rows = [
        {"component": f"svd_{idx + 1}", "coefficient": float(value)}
        for idx, value in enumerate(analysis["model_coefficients"][departure_idx])
    ]
    st.sidebar.subheader(f"Departure coefficients: {departure_label}")
    st.sidebar.table(coeff_rows)

    tab_donor, tab_steer, tab_overlap, tab_routed, tab_build = st.tabs(
        [
            "🎯 Donor transfer",
            "🔧 Direct steering",
            "📊 Overlap & islands",
            "🧭 Routed composition",
            "🏗️ Build & compress",
        ]
    )
    with tab_donor:
        _render_donor_tab(analysis, block_diag, departure_label, departure_idx, labels)
    with tab_steer:
        _render_steering_tab(analysis, departure_label, departure_idx, labels)
    with tab_overlap:
        _render_overlap_tab(analysis, block_diag, departure_label, departure_idx, labels)
    with tab_routed:
        _render_routed_tab(
            analysis,
            args,
            departure_label,
            departure_idx,
            inferred_departure_path,
        )
    with tab_build:
        _render_build_tab(analysis, args, departure_label, departure_idx, inferred_departure_path)


if __name__ == "__main__":
    main()
