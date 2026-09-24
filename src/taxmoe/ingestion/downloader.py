from __future__ import annotations
from pathlib import Path
from urllib.request import Request, urlopen
from .....taxmoe.ingestion.storage import atomic_write_bytes


def download(url: str, destination: str | Path, user_agent: str = "TaxMoE/0.1") -> Path:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=60) as response:  # nosec - caller controls source policy
        data = response.read()
    destination = Path(destination)
    atomic_write_bytes(destination, data)
    return destination
