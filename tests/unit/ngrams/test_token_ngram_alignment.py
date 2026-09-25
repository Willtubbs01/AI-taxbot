from taxmoe.alignment.models import AlignmentResult,AlignmentStatistics,CharacterSpan,RenderedSegment,TokenCharacterAlignment
from taxmoe.ngrams.hashing import HashedNGramOccurrence
from taxmoe.ngrams.namespaces import NGramNamespace
from taxmoe.ngrams.token_alignment import TokenNGramAligner

def test_bigram_goes_to_right_edge():
    align=AlignmentResult(record_id='x',rendered_text_hash='h',segments=[RenderedSegment(kind='message_content',segment_id='text:0',rendered_span=CharacterSpan(start=0,end=10),source_span=CharacterSpan(start=0,end=10))],statistics=AlignmentStatistics(total_tokens=2,content_tokens=2),token_alignments=[
      TokenCharacterAlignment(token_index=0,token_id=1,rendered_span=CharacterSpan(start=0,end=4),segment_id='text:0',segment_span=CharacterSpan(start=0,end=4)),
      TokenCharacterAlignment(token_index=1,token_id=2,rendered_span=CharacterSpan(start=4,end=10),segment_id='text:0',segment_span=CharacterSpan(start=4,end=10)),
    ])
    gram=HashedNGramOccurrence(feature_id=123,namespace=NGramNamespace.WORD_2,bucket=1,segment_id='text:0',start_char=0,end_char=10,order=2)
    result=TokenNGramAligner().align(align,[gram])
    assert result.tokens[0].feature_ids==[]
    assert result.tokens[1].feature_ids==[123]
