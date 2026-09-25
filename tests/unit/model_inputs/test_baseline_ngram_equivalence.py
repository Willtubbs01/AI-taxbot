from taxmoe.model_inputs.validation import validate_baseline_ngram_equivalence

def test_equal_baseline_inputs_pass():
    a=[{'record_id':'x','input_ids':[1,2],'labels':[-100,2]}]
    b=[{'record_id':'x','input_ids':[1,2],'labels':[-100,2],'ngram_feature_ids':[[0]*24,[1]+[0]*23]}]
    assert validate_baseline_ngram_equivalence(a,b).passed
