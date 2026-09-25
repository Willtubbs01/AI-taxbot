from taxmoe.ngrams.extraction import NGramExtractor
from taxmoe.ngrams.hashing import NGramHasher
from taxmoe.ngrams.token_alignment import TokenNGramAligner
from taxmoe.alignment.models import AlignmentResult,AlignmentStatistics,CharacterSpan,RenderedSegment,TokenCharacterAlignment

def test_extraction_hash_alignment_smoke():
    text='cost basis'
    ext=NGramExtractor().extract_text(text)
    hasher=NGramHasher()
    hashed=[hasher.hash_occurrence(x) for x in ext.occurrences]
    align=AlignmentResult(record_id='x',rendered_text_hash='h',segments=[RenderedSegment(kind='message_content',segment_id='text:0',rendered_span=CharacterSpan(start=0,end=len(text)),source_span=CharacterSpan(start=0,end=len(text)))],statistics=AlignmentStatistics(total_tokens=2,content_tokens=2),token_alignments=[TokenCharacterAlignment(token_index=0,token_id=1,rendered_span=CharacterSpan(start=0,end=4),segment_id='text:0',segment_span=CharacterSpan(start=0,end=4)),TokenCharacterAlignment(token_index=1,token_id=2,rendered_span=CharacterSpan(start=5,end=10),segment_id='text:0',segment_span=CharacterSpan(start=5,end=10))])
    out=TokenNGramAligner().align(align,hashed)
    assert any(f for f in out.tokens[1].feature_ids)
