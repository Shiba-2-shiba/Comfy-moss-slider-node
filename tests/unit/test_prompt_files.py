import hashlib
from pathlib import Path

import pytest
import torch


def test_yaml_snapshot_supports_block_text_anchors_and_content_fingerprint(tmp_path):
    from moss_slider_node.prompt_files import list_prompt_files, load_prompt_file

    path = tmp_path / "aging.yaml"
    data = b"\xef\xbb\xbf- target: &base >-\n    An adult woman\n    standing in a studio.\n  positive: An elderly adult woman standing in a studio.\n  negative: *base\n"
    path.write_bytes(data)
    (tmp_path / "README.md").write_text("not a preset")
    assert list_prompt_files(tmp_path) == ["aging.yaml"]
    snapshot = load_prompt_file(tmp_path, "aging.yaml")
    assert snapshot.name == "aging.yaml"
    assert snapshot.sha256 == hashlib.sha256(data).hexdigest()
    assert snapshot.records[0]["target"] == "An adult woman standing in a studio."
    assert snapshot.records[0]["negative"] == snapshot.records[0]["target"]
    path.write_bytes(data.replace(b"elderly", b"older"))
    assert load_prompt_file(tmp_path, "aging.yaml").sha256 != snapshot.sha256


@pytest.mark.parametrize("body", [
    "[]", "{}", "- target: t\n  positive: p", "- target: t\n  positive: p\n  negative: ' '",
    "- target: t\n  positive: p\n  negative: n\n  anchor: a",
    "- target: t\n  positive: p\n  negative: p",
    "- target: t\n  positive: p\n  negative: n\n  target: overwritten",
    "!!python/object/apply:os.system ['echo unsafe']",
])
def test_invalid_yaml_never_silently_drops_records_or_roles(tmp_path, body):
    from moss_slider_node.prompt_files import load_prompt_file

    (tmp_path / "bad.yaml").write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompt_file(tmp_path, "bad.yaml")


@pytest.mark.parametrize("name", ["../outside.yaml", "..\\outside.yaml", "/outside.yaml", "missing.yaml"])
def test_selection_stays_in_the_prompt_directory(tmp_path, name):
    from moss_slider_node.prompt_files import load_prompt_file

    with pytest.raises(ValueError):
        load_prompt_file(tmp_path, name)


def test_missing_directory_lists_no_presets(tmp_path):
    from moss_slider_node.prompt_files import list_prompt_files

    assert list_prompt_files(tmp_path / "missing") == []


def test_encoder_keeps_distinct_role_lengths_and_uses_native_clip_template():
    from moss_slider_node.prompt_files import encode_prompt_records

    class Clip:
        def tokenize(self, text):
            return text

        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.ones(1, len(tokens), 4096), {"attention_mask": torch.ones(1, len(tokens))}]]

    records = [{"target": "tt", "positive": "ppp", "negative": "n"},
               {"target": "tttt", "positive": "ppppp", "negative": "nn"}]
    encoded = encode_prompt_records(Clip(), records)
    assert len(encoded) == 2
    assert [encoded[0][role][0][0].shape[1] for role in ("target", "positive", "negative")] == [2, 3, 1]
    assert encoded[1]["target"][0][0].shape == (1, 4, 4096)
    assert all(value[0][0].device.type == "cpu" and not value[0][0].requires_grad
               for record in encoded for value in record.values())


def test_packaged_presets_are_loadable_training_records():
    from moss_slider_node.prompt_files import list_prompt_files, load_prompt_file

    directory = Path(__file__).resolve().parents[2] / "prompts"
    names = list_prompt_files(directory)
    assert len(names) == 6
    for name in names:
        snapshot = load_prompt_file(directory, name)
        assert len(snapshot.records) == 6
        assert all(set(record) == {"target", "positive", "negative", "neutral"} for record in snapshot.records)
        assert all(record["neutral"] == record["target"] for record in snapshot.records)


def test_optional_neutral_snapshot_cache_and_mixed_progress(tmp_path):
    from moss_slider_node.prompt_files import encode_prompt_records, load_prompt_file

    body = "- target: tt\n  positive: p\n  negative: n\n- target: tt\n  positive: p\n  negative: n\n  neutral: rr\n"
    (tmp_path / "mixed.yaml").write_bytes(body.encode())
    snapshot = load_prompt_file(tmp_path, "mixed.yaml")
    assert "neutral" not in snapshot.records[0] and snapshot.records[1]["neutral"] == "rr"
    assert snapshot.sha256 == hashlib.sha256(body.encode()).hexdigest()
    calls, progress = [], []

    class Clip:
        def tokenize(self, text):
            calls.append(text)
            return text

        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.ones(1, len(tokens), 4096), {}]]

    encoded = encode_prompt_records(Clip(), snapshot.records, progress=lambda c, t: progress.append((c, t)))
    assert calls == ["tt", "p", "n", "rr"]
    assert progress == [(n, 7) for n in range(1, 8)]
    assert set(encoded[0]) == {"target", "positive", "negative"}
    assert encoded[1]["neutral"][0][0].shape == (1, 2, 4096)


@pytest.mark.parametrize("value", ["null", "''", "'  '", "42"])
def test_explicit_invalid_neutral_is_rejected(tmp_path, value):
    from moss_slider_node.prompt_files import load_prompt_file

    (tmp_path / "bad.yaml").write_text(f"- target: t\n  positive: p\n  negative: n\n  neutral: {value}\n")
    with pytest.raises(ValueError, match="nonempty"):
        load_prompt_file(tmp_path, "bad.yaml")
