from taxmoe.model_inputs.release_models import InputReleaseStatus,ModelInputReleaseManifest
from taxmoe.model_inputs.verifier import release_manifest_hash,verify_release_manifest

def test_release_hash_verification():
    kw=dict(release_id='R',name='TaxMoE-Inputs-Qwen3-v0.1',version='0.1',status=InputReleaseStatus.FROZEN,dataset_release_id='D',dataset_release_hash='1',tokenizer_manifest_hash='2',tokenization_manifest_hash='3',ngram_normalization_manifest_hash='4',ngram_hash_manifest_hash='5',token_alignment_manifest_hash='6',ngram_alignment_manifest_hash='7',token_cache_hash='8',ngram_cache_hash='9',encoder_architecture_hash='a',router_interface_hash='b',validation_report_hash='c',benchmark_report_hash='d')
    m=ModelInputReleaseManifest(**kw)
    m=m.model_copy(update={'release_content_hash':release_manifest_hash(m)})
    assert verify_release_manifest(m)
