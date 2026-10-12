import json
from pathlib import Path

import numpy as np

from llmtraits.atlas import atlas_identity, scan_local_atlas
from llmtraits.cli import build_parser
from llmtraits.manifest import load_manifest
from llmtraits.recipes import load_analysis
from tests.test_routing import _case


def _artifact(output: Path) -> Path:
    return next((output / "tensors").glob("*.npz"))


def _multiblock_artifact(output: Path) -> Path:
    for path in (output / "tensors").glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            if len(data["starts"]) > 1:
                return path
    raise AssertionError("No multi-block atlas artifact")


def test_atlas_math_status_and_resume(tmp_path):
    analysis_dir, analysis, labels, delta_dirs, _ = _case(tmp_path / "case")
    output = tmp_path / "atlas"
    result = scan_local_atlas(
        str(analysis_dir), str(output), block_size=17, checkpoint_blocks=2
    )
    assert result["tensor_count"] == analysis["shared_tensor_count"]
    assert result["labels"] == labels
    assert json.loads((output / "atlas_progress.json").read_text())["status"] == "complete"

    statuses = list((output / "models").glob("*/analysis_status.json"))
    assert len(statuses) == len(labels)
    assert all(json.loads(path.read_text())["status"] == "complete" for path in statuses)

    with np.load(_artifact(output), allow_pickle=False) as data:
        name = str(data["tensor_name"].item())
        gram = np.asarray(data["gram"])
        starts = np.asarray(data["starts"])
        ends = np.asarray(data["ends"])
    deltas = []
    from tests.test_pipeline import load_delta_tensors

    for delta_dir in delta_dirs:
        deltas.append(
            load_delta_tensors(delta_dir, load_manifest(delta_dir))[name]
            .reshape(-1)
            .numpy()
        )
    matrix = np.stack(deltas)
    for index, (start, end) in enumerate(zip(starts, ends)):
        expected = matrix[:, int(start) : int(end)].astype(np.float64)
        np.testing.assert_allclose(gram[index], expected @ expected.T, rtol=1e-6, atol=1e-8)

    mtimes = {path: path.stat().st_mtime_ns for path in (output / "tensors").glob("*.npz")}
    resumed = scan_local_atlas(
        str(analysis_dir), str(output), block_size=17, checkpoint_blocks=2
    )
    assert resumed["identity_hash"] == result["identity_hash"]
    assert mtimes == {
        path: path.stat().st_mtime_ns for path in (output / "tensors").glob("*.npz")
    }


def test_atlas_partial_checkpoint_and_identity_guard(tmp_path):
    analysis_dir, analysis, _, _, _ = _case(tmp_path / "case")
    output = tmp_path / "atlas"
    scan_local_atlas(str(analysis_dir), str(output), block_size=17, checkpoint_blocks=2)
    artifact = _multiblock_artifact(output)
    with np.load(artifact, allow_pickle=False) as data:
        payload = {key: np.array(data[key], copy=True) for key in data.files}
    original_gram = payload["gram"].copy()
    payload["completed_blocks"] = np.array(1, dtype=np.int64)
    payload["gram"][1:] = 123.0
    with open(artifact, "wb") as handle:
        np.savez_compressed(handle, **payload)
    scan_local_atlas(str(analysis_dir), str(output), block_size=17, checkpoint_blocks=2)
    with np.load(artifact, allow_pickle=False) as data:
        assert int(data["completed_blocks"].item()) == len(data["starts"])
        np.testing.assert_allclose(data["gram"], original_gram)

    changed = atlas_identity(load_analysis(str(analysis_dir)), 19)
    assert changed["identity_hash"] != atlas_identity(analysis, 17)["identity_hash"]
    try:
        scan_local_atlas(str(analysis_dir), str(output), block_size=19)
    except ValueError as error:
        assert "identity mismatch" in str(error).lower()
    else:
        raise AssertionError("Expected atlas identity mismatch")


def test_atlas_cli_parsers():
    parser = build_parser()
    scan = parser.parse_args(
        ["atlas-scan", "--analysis", "a", "--output-dir", "o"]
    )
    launch = parser.parse_args(
        ["atlas-launch", "--analysis", "a", "--output-dir", "o"]
    )
    assert scan.block_size == 262144
    assert scan.checkpoint_blocks == 32
    assert launch.block_size == 262144
    assert launch.checkpoint_blocks == 32
