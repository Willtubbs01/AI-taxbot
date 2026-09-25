import torch
from taxmoe.data.collator import TaxMoETrainingCollator

def test_padding_and_ngram_alignment():
    c=TaxMoETrainingCollator(pad_token_id=99,include_ngram_features=True,pad_to_multiple_of=8)
    ex=[{'input_ids':[1,2,3],'labels':[-100,-100,3],'ngram_feature_ids':[[0]*24,[5]+[0]*23,[6,7]+[0]*22]}]
    b=c(ex)
    assert b['input_ids'].shape==(1,8)
    assert b['ngram_feature_ids'].shape==(1,8,24)
    assert torch.all(b['labels'][0,3:]==-100)
    assert torch.all(b['ngram_feature_ids'][0,3:]==0)
