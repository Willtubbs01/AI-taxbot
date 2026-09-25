import torch
from taxmoe.modeling.ngram_encoder import NGramEncoderConfig,NGramFeatureEncoder

def test_shapes_and_zero_feature_behavior():
    m=NGramFeatureEncoder(NGramEncoderConfig(num_features=100,embedding_dim=32,dropout=0.0)).eval()
    ids=torch.tensor([[[0,0,0],[1,2,0]]])
    out=m(ids)
    assert out.shape==(1,2,32)
    assert torch.equal(out[0,0],torch.zeros(32))
    assert not torch.equal(out[0,1],torch.zeros(32))
