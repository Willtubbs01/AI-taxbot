from taxmoe.ngrams.extraction import NGramExtractor

def test_cost_basis_bigram():
    r=NGramExtractor().extract_text('The cost basis is unknown.')
    keys={(o.order,o.normalized_key) for o in r.occurrences}
    assert (2,'cost basis') in keys
