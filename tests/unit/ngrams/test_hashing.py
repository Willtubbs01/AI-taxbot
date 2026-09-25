from taxmoe.ngrams.hashing import NGramHashConfig,NGramHasher
from taxmoe.ngrams.namespaces import NGramNamespace

def test_hash_stable_and_in_bounds():
    h=NGramHasher(NGramHashConfig.v1())
    a=h.feature_id(NGramNamespace.WORD_2,'cost basis')
    b=h.feature_id(NGramNamespace.WORD_2,'cost basis')
    assert a==b
    assert 1<=a[0]<h.config.embedding_rows

def test_namespace_separation():
    h=NGramHasher()
    a,_=h.feature_id(NGramNamespace.WORD_1,'basis')
    b,_=h.feature_id(NGramNamespace.CHAR_5,'basis')
    assert a!=b
