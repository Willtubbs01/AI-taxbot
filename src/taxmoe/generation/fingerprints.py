from taxmoe.schemas.common import stable_hash


def semantic_fingerprint(*parts) -> str:
    return stable_hash("semantic", *parts)


def structural_fingerprint(*parts) -> str:
    return stable_hash("structural", *parts)
