from pathlib import Path

from onecard.store.footprints import FootprintStore


def test_unknown_model_returns_none(tmp_path: Path):
    assert FootprintStore(tmp_path / "f.db").get("ollama", "ghost") is None


def test_recorded_value_is_returned(tmp_path: Path):
    s = FootprintStore(tmp_path / "f.db")
    s.record("ollama", "llama3.1:8b", 4768)
    assert s.get("ollama", "llama3.1:8b") == 4768


def test_recording_again_overwrites(tmp_path: Path):
    s = FootprintStore(tmp_path / "f.db")
    s.record("ollama", "m", 100)
    s.record("ollama", "m", 200)
    assert s.get("ollama", "m") == 200


def test_values_persist_across_instances(tmp_path: Path):
    db = tmp_path / "f.db"
    FootprintStore(db).record("ollama", "m", 321)
    assert FootprintStore(db).get("ollama", "m") == 321


def test_same_key_under_different_consumers_is_distinct(tmp_path: Path):
    s = FootprintStore(tmp_path / "f.db")
    s.record("ollama", "x", 100)
    s.record("comfyui", "x", 4000)
    assert s.get("ollama", "x") == 100
    assert s.get("comfyui", "x") == 4000


def test_all_returns_every_row(tmp_path: Path):
    s = FootprintStore(tmp_path / "f.db")
    s.record("ollama", "a", 1)
    s.record("comfyui", "b", 2)
    assert s.all() == {("ollama", "a"): 1, ("comfyui", "b"): 2}


def test_parent_directory_is_created(tmp_path: Path):
    s = FootprintStore(tmp_path / "nested" / "deeper" / "f.db")
    s.record("ollama", "m", 5)
    assert s.get("ollama", "m") == 5
