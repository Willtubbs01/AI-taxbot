from pathlib import Path
from .hashing import sha256_file

def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)

def verify_file(path: str | Path, expected_sha256: str) -> bool:
    return sha256_file(path) == expected_sha256
