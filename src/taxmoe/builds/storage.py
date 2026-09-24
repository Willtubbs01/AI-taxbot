from pathlib import Path


def build_dir(root: str | Path, build_id: str) -> Path:
    return Path(root) / build_id
