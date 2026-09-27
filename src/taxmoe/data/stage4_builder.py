from __future__ import annotations

from pathlib import Path

import yaml

from taxmoe.data.cache_builder import ArrowCacheBuilder, example_content_hash
from taxmoe.data.cache_manifest import compute_cache_content_hash
from taxmoe.data.cache_models import CachedTrainingExample, TrainingCacheManifest
from taxmoe.data.tokenization_pipeline import DatasetTokenizationPipeline
from taxmoe.ingestion.cpt import build_cpt_records, write_cpt_corpus
from taxmoe.ingestion.hashing import stable_hash
from taxmoe.modeling.tokenization_config import TokenizationConfig
from taxmoe.modeling.tokenizer import TaxMoETokenizer


def _load_tokenization_config(path: str | Path) -> TokenizationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TokenizationConfig.model_validate(raw)


def _for_cpt_tokenization(config: TokenizationConfig) -> TokenizationConfig:
    """Return the Stage 4 CPT view of the shared tokenization config.

    CPT uses full-causal labels over plain source text, so an assistant-token mask
    is neither consumed nor required.  The shared Stage 4 config keeps the SFT
    capability requirement enabled for conversation caches; enforcing it while
    building CPT would incorrectly reject tokenizers whose chat template does not
    expose Hugging Face ``{% generation %}`` spans.
    """
    return config.model_copy(update={"require_assistant_mask": False})


def _to_cached(example) -> CachedTrainingExample:
    content_hash = example_content_hash(example.record_id, example.input_ids, example.labels)
    return CachedTrainingExample(
        record_id=example.record_id,
        split=example.split,
        source_kind=example.source_kind,
        input_ids=example.input_ids,
        labels=example.labels,
        token_count=example.token_count,
        supervised_token_count=example.supervised_token_count,
        content_hash=content_hash,
    )


def build_cpt_token_cache(
    *,
    source_manifest: str | Path,
    tokenization_config: str | Path,
    output_dir: str | Path,
    project_root: str | Path = ".",
    chunk_max_chars: int = 6000,
    validation_percent: int = 10,
) -> tuple[Path, TrainingCacheManifest]:
    output_dir = Path(output_dir)
    source_dir = output_dir / "source" / "cpt"
    cache_dir = output_dir / "caches" / "cpt" / "token_only"
    manifests_dir = output_dir / "manifests"
    source_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    records, corpus_manifest = build_cpt_records(
        source_manifest,
        project_root=project_root,
        chunk_max_chars=chunk_max_chars,
        validation_percent=validation_percent,
    )
    write_cpt_corpus(source_dir, records, corpus_manifest)

    shared_token_cfg = _load_tokenization_config(tokenization_config)
    token_cfg = _for_cpt_tokenization(shared_token_cfg)
    tokenizer = TaxMoETokenizer(token_cfg)
    tokenizer_manifest = tokenizer.manifest()
    tokenizer_manifest_hash = stable_hash(tokenizer_manifest.model_dump(mode="json"))
    tokenization_manifest_hash = stable_hash(token_cfg.model_dump(mode="json"))
    (manifests_dir / "tokenizer_manifest.json").write_text(tokenizer_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (manifests_dir / "tokenization_config.resolved.json").write_text(token_cfg.model_dump_json(indent=2) + "\n", encoding="utf-8")

    pipeline = DatasetTokenizationPipeline(tokenizer)
    by_split: dict[str, list[CachedTrainingExample]] = {}
    for record in records:
        tokenized = pipeline.tokenize_cpt(record)
        by_split.setdefault(record.split, []).append(_to_cached(tokenized))

    builder = ArrowCacheBuilder()
    shards = []
    for split in sorted(by_split):
        examples = sorted(by_split[split], key=lambda x: x.record_id)
        if not examples:
            continue
        shard_path = cache_dir / f"{split}.arrow"
        shard = builder.write_shard(examples, shard_path, split)
        shards.append(shard.model_copy(update={"path": str(shard_path.relative_to(output_dir)).replace("\\", "/")}))

    cache_id = "CACHE-QWEN-CPT-" + corpus_manifest.content_hash[:16]
    manifest = TrainingCacheManifest(
        cache_id=cache_id,
        cache_kind="token_only",
        dataset_release_id=corpus_manifest.corpus_id,
        dataset_release_hash=corpus_manifest.content_hash,
        tokenizer_manifest_hash=tokenizer_manifest_hash,
        tokenization_manifest_hash=tokenization_manifest_hash,
        files=shards,
        total_records=sum(x.records for x in shards),
        total_tokens=sum(x.tokens for x in shards),
    )
    manifest = manifest.model_copy(update={"content_hash": compute_cache_content_hash(manifest)})
    manifest_path = output_dir / "cache_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest_path, manifest
