from pathlib import Path

from moss_slider_node.prompt_files import load_prompt_file


PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


def test_realistic_breast_preset_is_available_for_photographic_evaluation():
    assert (PROMPTS / "breast_size_slider_realistic.yaml").is_file()
    records = load_prompt_file(PROMPTS, "breast_size_slider_realistic.yaml").records
    assert len(records) == 12
    assert len({record["target"] for record in records}) == len(records)
    for record in records:
        assert record["target"].startswith("A photorealistic color photograph")
        assert "exactly one fully clothed adult woman" in record["target"]
        assert "opaque fitted" in record["target"]
        assert "anime" not in record["target"].lower()


def test_realistic_breast_pairs_change_only_breast_size():
    assert (PROMPTS / "breast_size_slider_realistic.yaml").is_file()
    records = load_prompt_file(PROMPTS, "breast_size_slider_realistic.yaml").records
    for record in records:
        target = record["target"]
        assert target.count("medium-sized breasts") == 1
        assert record["positive"].count("large breasts") == 1
        assert record["negative"].count("small breasts") == 1
        assert record["positive"].replace("large breasts", "medium-sized breasts") == target
        assert record["negative"].replace("small breasts", "medium-sized breasts") == target
        assert record["neutral"] == target
