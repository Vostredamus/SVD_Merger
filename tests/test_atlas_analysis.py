import json
from pathlib import Path

import numpy as np
import pytest

from llmtraits.atlas import scan_local_atlas
from llmtraits.atlas_analysis import analyze_local_atlas, derive_block_metrics
from llmtraits.cli import build_parser
from tests.test_routing import _case


def test_gram_metrics_match_direct_vectors():
    vectors = np.asarray(
        [
            [[1.0, 2.0, 0.0], [0.0, 2.0, 1.0], [1.0, -1.0, 2.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 4.0], [-1.0, 2.0, 2.0]],
        ],
        dtype=np.float64,
    )
    gram = np.einsum("bmi,bni->bmn", vectors, vectors)
    left = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=np.float64
    )
    singular = np.asarray([2.0, 3.0], dtype=np.float64)
    metrics = derive_block_metrics(gram, 0, left, singular)

    np.testing.assert_allclose(
        metrics["energy"], np.einsum("bmi,bmi->bm", vectors, vectors)
    )
    for block in range(len(vectors)):
        departure = vectors[block, 0]
        for model in range(vectors.shape[1]):
            candidate = vectors[block, model]
            if departure @ departure:
                residual = candidate - departure * (
                    (departure @ candidate) / (departure @ departure)
                )
            else:
                residual = candidate
            assert metrics["novelty_energy_departure"][block, model] == pytest.approx(
                residual @ residual
            )
        for component in range(left.shape[1]):
            coefficient = left[:, component] / singular[component]
            vector = coefficient @ vectors[block]
            assert metrics["svd_component_energy"][
                block, component
            ] == pytest.approx(vector @ vector)
            np.testing.assert_allclose(
                metrics["svd_model_component_dot"][block, :, component],
                vectors[block] @ vector,
            )
    assert np.all(metrics["pair_cosine"][1, 0] == 0)
    np.testing.assert_array_equal(
        metrics["pair_conflict"], metrics["pair_cosine"] < 0
    )
    assert np.all(np.isfinite(metrics["svd_model_component_affinity"]))


def test_analysis_candidates_preplans_resume_and_identity(tmp_path):
    analysis_dir, _, _, _, _ = _case(tmp_path / "case")
    atlas_dir = tmp_path / "atlas"
    scan_local_atlas(str(analysis_dir), str(atlas_dir), block_size=17)
    output = tmp_path / "derived"

    first = analyze_local_atlas(
        str(atlas_dir),
        str(analysis_dir),
        str(output),
        departure_label="departure",
    )
    assert first["status"] == "complete"
    assert first["candidate_count"] == 6
    assert len(list((output / "plots").glob("*.png"))) == 5
    recipes = sorted((output / "candidate_recipes").glob("*.json"))
    assert len(recipes) == 6
    recipe_bytes = {path.name: path.read_bytes() for path in recipes}
    route_indexes = sorted((output / "route_plans").glob("*/route_index.json"))
    assert len(route_indexes) == 6
    for path in route_indexes:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["atlas_identity_hash"] == first["atlas_identity_hash"]
        assert payload["scale_status"] == "nominal_one_not_full_weight_rms_capped"
        assert payload["requires_finalization"]["departure_weight_rms_scale"] is True

    derived = next((output / "derived_tensors").glob("*.npz"))
    with np.load(derived, allow_pickle=False) as data:
        payload = {key: np.array(data[key], copy=True) for key in data.files}
    payload["energy"] = payload["energy"][:-1]
    with open(derived, "wb") as handle:
        np.savez_compressed(handle, **payload)

    plan = next((output / "route_plans").glob("*/plans/*.npz"))
    with np.load(plan, allow_pickle=False) as data:
        plan_payload = {key: np.array(data[key], copy=True) for key in data.files}
        expected_winner = np.array(data["winner"], copy=True)
    plan_payload["winner"] = plan_payload["winner"][:-1]
    with open(plan, "wb") as handle:
        np.savez_compressed(handle, **plan_payload)

    second = analyze_local_atlas(
        str(atlas_dir),
        str(analysis_dir),
        str(output),
        departure_label="departure",
    )
    assert second["identity_hash"] == first["identity_hash"]
    assert {
        path.name: path.read_bytes()
        for path in sorted((output / "candidate_recipes").glob("*.json"))
    } == recipe_bytes
    with np.load(derived, allow_pickle=False) as data:
        assert len(data["energy"]) == int(data["ends"].shape[0])
    with np.load(plan, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["winner"], expected_winner)

    with pytest.raises(ValueError, match="identity mismatch"):
        analyze_local_atlas(
            str(atlas_dir),
            str(analysis_dir),
            str(output),
            departure_label="donor_a",
        )


def test_atlas_analysis_cli_parsers():
    parser = build_parser()
    foreground = parser.parse_args(
        ["atlas-analyze", "--atlas-dir", "atlas", "--analysis", "svd",
         "--departure-label", "Example-L3.3-70B-Model-V1.5"]
    )
    detached = parser.parse_args(
        ["atlas-analyze-launch", "--atlas-dir", "atlas", "--analysis", "svd",
         "--departure-label", "Example-L3.3-70B-Model-V1.5"]
    )
    assert foreground.output_dir is None
    assert detached.departure_label == "Example-L3.3-70B-Model-V1.5"
