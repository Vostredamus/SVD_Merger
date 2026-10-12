from pathlib import Path

from llmtraits.compose import archive_model_metadata


def test_archive_model_metadata_excludes_weight_files(tmp_path):
    model = tmp_path / "model"
    delta = tmp_path / "delta"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "README.md").write_text("model", encoding="utf-8")
    (model / "model-00001.safetensors").write_bytes(b"weights")
    (model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

    result = archive_model_metadata(str(model), str(delta))

    assert (result / "config.json").exists()
    assert (result / "README.md").exists()
    assert not (result / "model-00001.safetensors").exists()
    assert not (result / "model.safetensors.index.json").exists()
