from __future__ import annotations

from functools import lru_cache
from hashlib import blake2b

from pydantic import Field, model_validator
from taxmoe.modeling.base import Stage4Model
from .models import NGramOccurrence
from .namespaces import NGRAM_NAMESPACE_ORDER_V1, NGramNamespace, namespace_for


class NamespaceHashConfig(Stage4Model):
    bucket_count: int = Field(gt=0)
    offset: int = Field(ge=0)


class NGramHashConfig(Stage4Model):
    version: str = "1"
    algorithm: str = "blake2b-64"
    key_id: str = "taxmoe-ngram-v1"
    namespaces: dict[NGramNamespace, NamespaceHashConfig]

    @classmethod
    def v1(cls):
        counts = {
            NGramNamespace.WORD_1: 16_384,
            NGramNamespace.WORD_2: 32_768,
            NGramNamespace.WORD_3: 32_768,
            NGramNamespace.WORD_4: 16_384,
            NGramNamespace.CHAR_3: 16_384,
            NGramNamespace.CHAR_4: 16_384,
            NGramNamespace.CHAR_5: 16_384,
        }
        offset = 0
        ns = {}
        for name in NGRAM_NAMESPACE_ORDER_V1:
            ns[name] = NamespaceHashConfig(bucket_count=counts[name], offset=offset)
            offset += counts[name]
        return cls(namespaces=ns)

    @property
    def real_bucket_count(self) -> int:
        return sum(x.bucket_count for x in self.namespaces.values())

    @property
    def embedding_rows(self) -> int:
        return 1 + self.real_bucket_count


class HashedNGramOccurrence(Stage4Model):
    feature_id: int = Field(gt=0)
    namespace: NGramNamespace
    bucket: int = Field(ge=0)
    segment_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    order: int = Field(gt=0)
    derived_alias: bool = False
    normalized_key: str | None = None


class NGramHasher:
    def __init__(self, config: NGramHashConfig | None = None):
        self.config = config or NGramHashConfig.v1()

    @lru_cache(maxsize=262_144)
    def feature_id(self, namespace: NGramNamespace, normalized_key: str) -> tuple[int, int]:
        if not normalized_key:
            raise ValueError("cannot hash empty normalized key")
        payload = namespace.value.encode() + b"\0" + normalized_key.encode("utf-8")
        digest = blake2b(payload, digest_size=8, key=self.config.key_id.encode("utf-8")).digest()
        raw = int.from_bytes(digest, "little", signed=False)
        ns = self.config.namespaces[namespace]
        bucket = raw % ns.bucket_count
        feature_id = 1 + ns.offset + bucket
        return feature_id, bucket

    def hash_occurrence(self, occurrence: NGramOccurrence, *, keep_key: bool = False) -> HashedNGramOccurrence:
        namespace = namespace_for(occurrence.kind, occurrence.order)
        fid, bucket = self.feature_id(namespace, occurrence.normalized_key)
        return HashedNGramOccurrence(
            feature_id=fid,
            namespace=namespace,
            bucket=bucket,
            segment_id=occurrence.segment_id,
            start_char=occurrence.start_char,
            end_char=occurrence.end_char,
            order=occurrence.order,
            derived_alias=occurrence.derived_alias,
            normalized_key=occurrence.normalized_key if keep_key else None,
        )
