import torch
from taxmoe.data.generation_collator import TaxMoEGenerationCollator

def test_left_padding_keeps_ngram_alignment():
    c=TaxMoEGenerationCollator(99,include_ngram_features=True,pad_to_multiple_of=4)
    rows=[[1]+[0]*23,[2]+[0]*23]
    b=c([{'prompt_input_ids':[10,11],'prompt_ngram_feature_ids':rows}])
    assert b['input_ids'].tolist()==[[99,99,10,11]]
    assert b['ngram_feature_ids'][0,2,0].item()==1
    assert b['ngram_feature_ids'][0,3,0].item()==2
