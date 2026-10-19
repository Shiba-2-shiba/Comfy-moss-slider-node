"""Local YAML prompt snapshots and native MOSS CLIP encoding."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from .model_adapter import normalize_conditioning


_ROLES = ("target", "positive", "negative")


@dataclass(frozen=True)
class PromptFile:
    name: str
    sha256: str
    records: list[dict[str, str]]


class _PromptLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ValueError(f"Duplicate prompt YAML key: {key}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def list_prompt_files(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir()
                  if path.is_file() and path.suffix.lower() in (".yaml", ".yml") and path.resolve().parent == root)


def load_prompt_file(directory, name):
    root = Path(directory).resolve()
    if not isinstance(name, str) or "/" in name or "\\" in name or name not in list_prompt_files(root):
        raise ValueError("Select a YAML file from this repository's prompts list")
    path = (root / name).resolve()
    if path.parent != root or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Prompt YAML must stay inside prompts and be at most 2 MiB")
    data = path.read_bytes()
    try:
        records = yaml.load(data.decode("utf-8-sig"), Loader=_PromptLoader)
    except (yaml.YAMLError, UnicodeError, TypeError) as error:
        raise ValueError(f"Invalid prompt YAML: {name}") from error
    if not isinstance(records, list) or not 1 <= len(records) <= 64:
        raise ValueError("Prompt YAML must be a list of 1 to 64 records")
    validated = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != set(_ROLES):
            raise ValueError(f"Record {index + 1} requires only target, positive and negative; anchor/neutral are unsupported")
        if any(not isinstance(record[role], str) or not record[role].strip() for role in _ROLES):
            raise ValueError(f"Record {index + 1} requires three nonempty prompt strings")
        record = {role: record[role].strip() for role in _ROLES}
        if record["positive"] == record["negative"]:
            raise ValueError(f"Record {index + 1} requires distinct positive and negative concepts")
        validated.append(record)
    return PromptFile(name, hashlib.sha256(data).hexdigest(), validated)


def encode_prompt_records(clip, records, *, cancel=None, progress=None):
    encoded, cache = [], {}
    total = len(records) * len(_ROLES)
    for index, record in enumerate(records):
        current = {}
        for offset, role in enumerate(_ROLES, 1):
            if cancel:
                cancel()
            text = record[role]
            if text not in cache:
                conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(text))
                # Trim each role's own padding and retain a detached CPU tensor.
                cache[text] = [[normalize_conditioning(conditioning), {}]]
            current[role] = cache[text]
            if progress:
                progress(index * len(_ROLES) + offset, total)
        encoded.append(current)
    return encoded
