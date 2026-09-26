from taxmoe.model_releases.models import TaxDenseReleaseManifest

def test_manifest_requires_fields():
    m=TaxDenseReleaseManifest(release_id='R',architecture='Qwen3ForCausalLM',parent_model_id='Qwen/Qwen3-0.6B',parent_model_revision='abc',parent_model_fingerprint='p',tokenizer_manifest_hash='t',input_release_id='I',input_release_hash='i',source_run_id='run',source_checkpoint_id='c',source_checkpoint_hash='ch',training_data_manifest_hash='d',objective_config_hash='o',optimization_config_hash='opt',evaluation_hash='e',reproducibility_hash='r',parameter_count=1,architecture_hash='a',weight_manifest_hash='w')
    assert m.name=='TaxDense-0.6B'
