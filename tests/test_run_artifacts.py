import json
import pytest

from llm_pruning.run_artifacts import RunArtifacts, file_fingerprint


def artifacts(path, root):
    return RunArtifacts(path, root, {"seed": 17}, model={"identifier": "random-init"})


def test_run_is_self_describing_and_never_overwrites(tmp_path):
    output = tmp_path / "run"
    with artifacts(output, tmp_path) as run:
        run.write_json("metrics.json", {"accuracy": 0.5})
        run.complete(["# Run", "Status: complete."])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["seed"] == 17
    assert manifest["config"] == {"seed": 17}
    assert "python" in manifest["versions"]
    assert set(manifest["git"]) == {"commit", "dirty"}
    assert manifest["finished_at"] >= manifest["started_at"]
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        with artifacts(output, tmp_path):
            pass
    assert (output / "manifest.json").read_bytes() == before


def test_failure_preserves_error_and_original_exception(tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="model unavailable"):
        with artifacts(output, tmp_path):
            raise RuntimeError("model unavailable")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == {"type": "RuntimeError", "message": "model unavailable"}
    assert "Status: failed." in (output / "summary.md").read_text()


def test_incomplete_context_and_summary_limit(tmp_path):
    with artifacts(tmp_path / "incomplete", tmp_path) as run:
        with pytest.raises(ValueError, match="40 single lines"):
            run.write_summary(["one\ntwo"])
        with pytest.raises(ValueError, match="40 single lines"):
            run.write_summary(["line"] * 41)
    manifest = json.loads((tmp_path / "incomplete" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "IncompleteRun"


def test_fingerprint_changes_with_file_content(tmp_path):
    dataset = tmp_path / "input.txt"
    dataset.write_text("one")
    first = file_fingerprint(dataset)
    dataset.write_text("two")
    second = file_fingerprint(dataset)
    assert first["sha256"] != second["sha256"]
    assert first["size_bytes"] == second["size_bytes"] == 3
    assert first["path"] == str(dataset.resolve())
