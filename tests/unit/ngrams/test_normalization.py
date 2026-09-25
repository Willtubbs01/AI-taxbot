from taxmoe.ngrams.config import NGramNormalizationConfig
from taxmoe.ngrams.normalization import normalize_atom

def test_tax_specific_normalization():
    c=NGramNormalizationConfig()
    assert normalize_atom('Cost',c).key=='cost'
    assert normalize_atom('$5,000',c).key=='<money>'
    assert normalize_atom('37',c).key=='<number>'
    assert normalize_atom('2025',c).key=='2025'
    x=normalize_atom('1099-B',c)
    assert x.key=='1099-b' and '1099b' in x.aliases
