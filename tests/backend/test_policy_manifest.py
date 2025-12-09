import json
import pickle
from pathlib import Path

import numpy as np

from backend.axiom_engine import policy as policy_mod


def test_load_policy_manifest_from_directory(tmp_path):
    manifest = {
        "schema_version": "policy_manifest@v2",
        "policy": {
            "hash": "abc123",
            "version": "v-test",
            "model_type": "mock",
            "serialization": "pickle@3.11",
            "byte_size": 10,
            "created_at": "2025-12-06T00:00:00Z",
        },
    }
    (tmp_path / "policy.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded = policy_mod.load_policy_manifest(str(tmp_path))
    assert loaded["policy"]["hash"] == "abc123"
    assert loaded["policy"]["version"] == "v-test"


def test_resolve_weight_path_falls_back_to_default(tmp_path):
    weights = tmp_path / "policy.weights.bin"
    dummy = policy_mod.MockPolicy()
    with weights.open("wb") as handle:
        pickle.dump(dummy, handle)
    resolved = policy_mod._resolve_weight_path(weights, tmp_path)  # type: ignore[attr-defined]
    assert resolved == weights


def test_load_policy_scores_with_manifest(tmp_path):
    weights = tmp_path / "policy.weights.bin"
    with weights.open("wb") as handle:
        pickle.dump(policy_mod.MockPolicy(), handle)
    manifest = {
        "schema_version": "policy_manifest@v2",
        "policy": {
            "hash": "ff" * 32,
            "version": "v1",
            "model_type": "mock",
            "serialization": "pickle@3.11",
            "byte_size": weights.stat().st_size,
            "created_at": "2025-12-06T00:00:00Z",
        },
    }
    (tmp_path / "policy.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded_policy = policy_mod.load_policy(str(weights))
    feats = np.array([[1, 0, 1]])
    scores = policy_mod.score_batch(loaded_policy, feats)
    assert scores.shape == (1,)
