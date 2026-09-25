from __future__ import annotations
import hashlib, json
from taxmoe.modeling.inputs import TokenizedExample
from taxmoe.modeling.tokenizer import TaxMoETokenizer
from .readers import CPTInputRecord, SFTInputRecord


def _hash_obj(obj) -> str:
    raw=json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()

class DatasetTokenizationPipeline:
    def __init__(self, tokenizer: TaxMoETokenizer):
        self.tokenizer=tokenizer

    def tokenize_cpt(self, record: CPTInputRecord) -> TokenizedExample:
        tok=self.tokenizer.tokenize_text(record.text)
        labels=list(tok.input_ids)
        return TokenizedExample(record_id=record.record_id,source_kind='cpt',split=record.split,input_ids=tok.input_ids,attention_mask=tok.attention_mask,labels=labels,token_count=len(labels),supervised_token_count=len(labels))

    def tokenize_sft(self, record: SFTInputRecord) -> TokenizedExample:
        record.validate_v01_conversation()
        ids,attn,assistant_mask=self.tokenizer.tokenize_messages_with_mask(record.messages)
        labels=[tid if int(m)==1 else -100 for tid,m in zip(ids,assistant_mask)]
        supervised=sum(x!=-100 for x in labels)
        if supervised == 0:
            raise ValueError('TOKENIZE-NO-ASSISTANT-TOKENS')
        return TokenizedExample(record_id=record.record_id,source_kind='sft',split=record.split,input_ids=ids,attention_mask=attn,labels=labels,token_count=len(ids),supervised_token_count=supervised)
