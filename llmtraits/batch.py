import json
import logging
import os
import re
from argparse import Namespace
from pathlib import Path

from .cross import cross_command
from .delta import DeltaComputer
from .manifest import load_manifest
from .scan import cached_model_fingerprint, dedup_models, find_models
from .svd import svd_command
from .visualize import visualize_command

LOGGER = logging.getLogger(__name__)


def _slug(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    return value or "model"


def _is_complete(delta_dir: Path, base_fingerprint: str, target_fingerprint: str, dtype: str) -> bool:
    manifest = load_manifest(str(delta_dir))
    if manifest is None:
        return False
    if (
        manifest.base_fingerprint != base_fingerprint
        or manifest.target_fingerprint != target_fingerprint
        or manifest.delta_dtype != dtype
    ):
        return False
    if not manifest.shards or any(status.status != "complete" for status in manifest.shards.values()):
        return False
    if any(status.file_sha256 and not (delta_dir / name).exists() for name, status in manifest.shards.items()):
        return False
    return bool(manifest.stats_json and (delta_dir / manifest.stats_json).exists())


def batch_command(args) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base_path = os.path.abspath(args.base)
    models = find_models(args.scan_dirs, skip_paths={base_path})
    unique, duplicates = dedup_models(models)
    targets = [model for model in unique if model.is_valid]
    base_fingerprint = cached_model_fingerprint(base_path)

    used_slugs: set[str] = set()
    registry = {"base": base_path, "models": [], "duplicates": []}
    selected = []
    for model in targets:
        slug = _slug(model.name)
        if slug.lower() in used_slugs:
            slug = f"{slug}-{model.fingerprint[:8]}"
        used_slugs.add(slug.lower())
        delta_dir = output_root / slug
        selected.append((model, slug, delta_dir))
        registry["models"].append(
            {"name": model.name, "path": model.path, "slug": slug, "fingerprint": model.fingerprint}
        )
    for kept, duplicate in duplicates:
        registry["duplicates"].append({"path": duplicate.path, "duplicate_of": kept.path})

    registry_tmp = output_root / "registry.json.tmp"
    with open(registry_tmp, "w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
    os.replace(registry_tmp, output_root / "registry.json")

    delta_dirs = []
    labels = []
    for index, (model, slug, delta_dir) in enumerate(selected, start=1):
        LOGGER.info("[%d/%d] Processing %s", index, len(selected), model.path)
        if not _is_complete(delta_dir, base_fingerprint, model.fingerprint, args.delta_dtype):
            DeltaComputer(
                base_dir=base_path,
                target_dir=model.path,
                output_dir=str(delta_dir),
                dtype=args.delta_dtype,
                force=False,
                skip_landscapes=args.skip_landscapes,
            ).run()
        else:
            LOGGER.info("Delta already complete: %s", delta_dir)

        if not args.skip_visuals:
            visualize_command(
                Namespace(delta_dir=str(delta_dir), skip_landscapes=args.skip_landscapes, force=False)
            )
        delta_dirs.append(str(delta_dir))
        labels.append(model.name)

    if not args.skip_cross and len(delta_dirs) >= 2:
        cross_output = output_root / "cross_model"
        cross_json = cross_output / "cross_analysis.json"
        reuse_cross = False
        if cross_json.exists():
            with open(cross_json, "r", encoding="utf-8") as handle:
                reuse_cross = json.load(handle).get("labels", []) == labels
        if reuse_cross:
            progress_path = cross_output / "svd_progress.json"
            svd_complete = False
            if progress_path.exists():
                with open(progress_path, "r", encoding="utf-8") as handle:
                    svd_complete = json.load(handle).get("phase") == "complete"
            if not svd_complete:
                svd_command(
                    Namespace(
                        registry=str(output_root / "registry.json"),
                        output_dir=str(cross_output),
                    )
                )
            else:
                LOGGER.info("Cross and SVD analyses already complete: %s", cross_output)
        else:
            cross_command(
                Namespace(
                    delta_dirs=delta_dirs,
                    output_dir=str(cross_output),
                    label=labels,
                )
            )
