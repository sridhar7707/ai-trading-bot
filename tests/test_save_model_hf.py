"""Tests for scripts/save_model_hf.py — file-size guard and upload logic."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _patch_hf_env(monkeypatch):
    """Provide minimal env so push() does not raise on startup."""
    import scripts.save_model_hf as mod
    monkeypatch.setattr(mod, "HF_TOKEN", "test-token")
    monkeypatch.setattr(mod, "HF_REPO_ID", "test/repo")


def _fs_stubs(file_map: dict):
    """
    Return (exists_fn, getsize_fn) that consult `file_map` {str_path: int_bytes}.
    Any path NOT in the map is treated as non-existent.
    """
    def exists_fn(path):
        return path in file_map

    def getsize_fn(path):
        return file_map[path]

    return exists_fn, getsize_fn


# --- file size guard ---

def test_push_skips_file_smaller_than_1024_bytes(monkeypatch):
    import scripts.save_model_hf as mod

    file_map = {
        "small_model.zip": 512,   # too small — should be skipped
        "good_regime.pkl": 2048,  # valid
    }
    monkeypatch.setattr(mod, "MODEL_SAVE_PATH", "small_model")  # .zip appended inside push()
    monkeypatch.setattr(mod, "REGIME_MODEL_PATH", "good_regime.pkl")

    exists_fn, getsize_fn = _fs_stubs(file_map)
    mock_api = MagicMock()
    queued = []

    def _fake_add(**kwargs):
        queued.append(kwargs.get("path_or_fileobj"))
        return MagicMock()

    with patch("scripts.save_model_hf.os.path.exists", side_effect=exists_fn), \
         patch("scripts.save_model_hf.os.path.getsize", side_effect=getsize_fn), \
         patch("scripts.save_model_hf.HfApi", return_value=mock_api), \
         patch("scripts.save_model_hf.CommitOperationAdd", side_effect=_fake_add):
        mod.push()

    assert "small_model.zip" not in queued, "Small file must not be queued"
    assert "good_regime.pkl" in queued, "Valid file must be queued"
    mock_api.create_commit.assert_called_once()


def test_push_accepts_file_at_exactly_1024_bytes(monkeypatch):
    import scripts.save_model_hf as mod

    file_map = {"boundary_model.zip": 1024}
    monkeypatch.setattr(mod, "MODEL_SAVE_PATH", "boundary_model")
    monkeypatch.setattr(mod, "REGIME_MODEL_PATH", "nonexistent.pkl")

    exists_fn, getsize_fn = _fs_stubs(file_map)
    mock_api = MagicMock()
    queued = []

    def _fake_add(**kwargs):
        queued.append(kwargs.get("path_or_fileobj"))
        return MagicMock()

    with patch("scripts.save_model_hf.os.path.exists", side_effect=exists_fn), \
         patch("scripts.save_model_hf.os.path.getsize", side_effect=getsize_fn), \
         patch("scripts.save_model_hf.HfApi", return_value=mock_api), \
         patch("scripts.save_model_hf.CommitOperationAdd", side_effect=_fake_add):
        mod.push()

    assert "boundary_model.zip" in queued, "Exactly 1024-byte file should be accepted"


def test_push_aborts_when_all_files_too_small(monkeypatch):
    import scripts.save_model_hf as mod

    file_map = {
        "sm_model.zip": 100,
        "sm_regime.pkl": 200,
        "models/saved/xgb_predictor.pkl": 50,
        "models/saved/lstm_predictor.pt": 50,
    }
    monkeypatch.setattr(mod, "MODEL_SAVE_PATH", "sm_model")
    monkeypatch.setattr(mod, "REGIME_MODEL_PATH", "sm_regime.pkl")

    exists_fn, getsize_fn = _fs_stubs(file_map)
    mock_api = MagicMock()

    with patch("scripts.save_model_hf.os.path.exists", side_effect=exists_fn), \
         patch("scripts.save_model_hf.os.path.getsize", side_effect=getsize_fn), \
         patch("scripts.save_model_hf.HfApi", return_value=mock_api):
        mod.push()

    mock_api.create_commit.assert_not_called()


def test_push_skips_missing_file(monkeypatch):
    import scripts.save_model_hf as mod

    # Only regime file exists and is large enough; everything else missing
    file_map = {"present_regime.pkl": 4096}
    monkeypatch.setattr(mod, "MODEL_SAVE_PATH", "missing_model")
    monkeypatch.setattr(mod, "REGIME_MODEL_PATH", "present_regime.pkl")

    exists_fn, getsize_fn = _fs_stubs(file_map)
    mock_api = MagicMock()

    with patch("scripts.save_model_hf.os.path.exists", side_effect=exists_fn), \
         patch("scripts.save_model_hf.os.path.getsize", side_effect=getsize_fn), \
         patch("scripts.save_model_hf.HfApi", return_value=mock_api), \
         patch("scripts.save_model_hf.CommitOperationAdd", return_value=MagicMock()):
        mod.push()

    mock_api.create_commit.assert_called_once()


def test_push_aborts_when_all_files_missing(monkeypatch):
    import scripts.save_model_hf as mod

    file_map = {}  # nothing exists
    monkeypatch.setattr(mod, "MODEL_SAVE_PATH", "ghost_model")
    monkeypatch.setattr(mod, "REGIME_MODEL_PATH", "ghost_regime.pkl")

    exists_fn, getsize_fn = _fs_stubs(file_map)
    mock_api = MagicMock()

    with patch("scripts.save_model_hf.os.path.exists", side_effect=exists_fn), \
         patch("scripts.save_model_hf.os.path.getsize", side_effect=getsize_fn), \
         patch("scripts.save_model_hf.HfApi", return_value=mock_api):
        mod.push()

    mock_api.create_commit.assert_not_called()


def test_push_raises_when_no_token(monkeypatch):
    import scripts.save_model_hf as mod
    monkeypatch.setattr(mod, "HF_TOKEN", "")

    with pytest.raises(EnvironmentError, match="HF_TOKEN"):
        mod.push()


def test_push_raises_when_no_repo_id(monkeypatch):
    import scripts.save_model_hf as mod
    monkeypatch.setattr(mod, "HF_TOKEN", "tok")
    monkeypatch.setattr(mod, "HF_REPO_ID", "")

    with pytest.raises(EnvironmentError):
        mod.push()


def test_push_commit_includes_model_info_json(monkeypatch):
    import scripts.save_model_hf as mod

    file_map = {"valid_regime.pkl": 2048}
    monkeypatch.setattr(mod, "MODEL_SAVE_PATH", "ghost_model")
    monkeypatch.setattr(mod, "REGIME_MODEL_PATH", "valid_regime.pkl")

    exists_fn, getsize_fn = _fs_stubs(file_map)
    mock_api = MagicMock()
    repo_paths = []

    def _fake_add(**kwargs):
        repo_paths.append(kwargs.get("path_in_repo"))
        return MagicMock()

    with patch("scripts.save_model_hf.os.path.exists", side_effect=exists_fn), \
         patch("scripts.save_model_hf.os.path.getsize", side_effect=getsize_fn), \
         patch("scripts.save_model_hf.HfApi", return_value=mock_api), \
         patch("scripts.save_model_hf.CommitOperationAdd", side_effect=_fake_add):
        mod.push()

    assert "model_info.json" in repo_paths, "Commit must include model_info.json metadata"
