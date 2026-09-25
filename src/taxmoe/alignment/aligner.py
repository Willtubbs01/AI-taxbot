from __future__ import annotations
import hashlib
from taxmoe.modeling.inputs import ChatMessage
from taxmoe.modeling.tokenizer import TaxMoETokenizer
from .exceptions import AlignmentError
from .message_mapping import map_rendered_span_to_segment
from .models import AlignmentResult, AlignmentStatistics, CharacterSpan, RenderedSegment, TokenCharacterAlignment
from .rendered_segments import derive_segments, instrument_messages
from .token_offsets import span_from_offset, validate_offsets


def _sha(text: str): return hashlib.sha256(text.encode()).hexdigest()

class TokenCharacterAligner:
    def __init__(self, tokenizer: TaxMoETokenizer):
        self.wrapper=tokenizer
        self.tokenizer=tokenizer.tokenizer

    def align_text(self, record_id: str, text: str) -> AlignmentResult:
        enc=self.tokenizer(text, add_special_tokens=True, return_offsets_mapping=True, return_attention_mask=True)
        ids=list(enc['input_ids']); offsets=list(enc['offset_mapping'])
        validate_offsets(offsets,len(text))
        specials=set(self.tokenizer.all_special_ids or [])
        segment=RenderedSegment(kind='message_content',segment_id='text:0',rendered_span=CharacterSpan(start=0,end=len(text)),source_span=CharacterSpan(start=0,end=len(text)))
        rows=[]
        for i,(tid,off) in enumerate(zip(ids,offsets)):
            special=int(tid) in specials
            rspan=span_from_offset(off,special=special)
            sspan=rspan.model_copy() if rspan else None
            rows.append(TokenCharacterAlignment(token_index=i,token_id=int(tid),rendered_span=rspan,segment_id='text:0' if sspan else None,segment_span=sspan,is_special=special,is_template_control=False))
        return AlignmentResult(record_id=record_id,rendered_text_hash=_sha(text),token_alignments=rows,segments=[segment],statistics=AlignmentStatistics(total_tokens=len(rows),content_tokens=sum(r.segment_id is not None for r in rows),special_tokens=sum(r.is_special for r in rows)))

    def align_messages(self, record_id: str, messages: list[ChatMessage], *, add_generation_prompt: bool=False) -> AlignmentResult:
        canonical=self.wrapper.render_messages(messages,add_generation_prompt=add_generation_prompt)
        instrumented=self.tokenizer.apply_chat_template(instrument_messages(messages),tokenize=False,add_generation_prompt=add_generation_prompt,enable_thinking=self.wrapper.config.chat.enable_thinking)
        segments=derive_segments(messages,canonical,instrumented)
        # Authoritative direct token stream.
        direct=self.tokenizer.apply_chat_template([m.model_dump() for m in messages],tokenize=True,add_generation_prompt=add_generation_prompt,enable_thinking=self.wrapper.config.chat.enable_thinking)
        direct_ids=list(direct['input_ids'] if isinstance(direct,dict) else direct)
        # Offset stream from rendered text. No duplicate special-token injection.
        enc=self.tokenizer(canonical,add_special_tokens=False,return_offsets_mapping=True,return_attention_mask=True)
        ids=list(enc['input_ids']); offsets=list(enc['offset_mapping'])
        if ids != direct_ids:
            raise AlignmentError('ALIGN-TOKEN-STREAM-MISMATCH')
        validate_offsets(offsets,len(canonical))
        specials=set(self.tokenizer.all_special_ids or [])
        rows=[]; content=control=special_count=0
        for i,(tid,off) in enumerate(zip(ids,offsets)):
            is_special=int(tid) in specials
            rspan=span_from_offset(off,special=is_special)
            seg=local=None
            if rspan is not None:
                seg,local=map_rendered_span_to_segment(rspan,segments)
            is_control=(seg is None)
            if seg: content+=1
            if is_control: control+=1
            if is_special: special_count+=1
            rows.append(TokenCharacterAlignment(token_index=i,token_id=int(tid),rendered_span=rspan,segment_id=seg.segment_id if seg else None,segment_span=local,role=seg.role if seg else None,is_special=is_special,is_template_control=is_control))
        return AlignmentResult(record_id=record_id,rendered_text_hash=_sha(canonical),token_alignments=rows,segments=segments,statistics=AlignmentStatistics(total_tokens=len(rows),content_tokens=content,template_control_tokens=control,special_tokens=special_count,unmapped_tokens=0,cross_segment_tokens=0))
