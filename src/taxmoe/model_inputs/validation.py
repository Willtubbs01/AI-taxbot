from __future__ import annotations
from dataclasses import dataclass,field
import torch

@dataclass
class ValidationIssue:
    code:str
    message:str
    fatal:bool=True

@dataclass
class Stage4ValidationReport:
    issues:list[ValidationIssue]=field(default_factory=list)
    metrics:dict=field(default_factory=dict)
    @property
    def passed(self):return not any(x.fatal for x in self.issues)
    def require(self,condition:bool,code:str,message:str):
        if not condition:self.issues.append(ValidationIssue(code,message,True))


def validate_baseline_ngram_equivalence(token_examples,ngram_examples):
    report=Stage4ValidationReport()
    report.require(len(token_examples)==len(ngram_examples),'INPUT-FREEZE-RECORD-COUNT-MISMATCH','cache record counts differ')
    for a,b in zip(token_examples,ngram_examples):
        report.require(a['record_id']==b['record_id'],'INPUT-FREEZE-RECORD-ID-MISMATCH',f"{a['record_id']} != {b['record_id']}")
        report.require(a['input_ids']==b['input_ids'],'INPUT-FREEZE-TOKEN-DIVERGENCE',a['record_id'])
        report.require(a['labels']==b['labels'],'INPUT-FREEZE-LABEL-DIVERGENCE',a['record_id'])
    return report


def validate_collated_batch(batch,ngram_pad_id=0,label_pad_id=-100):
    report=Stage4ValidationReport()
    mask=batch['attention_mask'].eq(0)
    report.require(torch.all(batch['labels'][mask].eq(label_pad_id)).item(),'INPUT-FREEZE-PAD-LABEL-FAIL','padding labels must be -100')
    if 'ngram_feature_ids' in batch:
        nmask=mask.unsqueeze(-1).expand_as(batch['ngram_feature_ids'])
        report.require(torch.all(batch['ngram_feature_ids'][nmask].eq(ngram_pad_id)).item(),'INPUT-FREEZE-PAD-NGRAM-FAIL','padded positions must have zero n-gram IDs')
    return report
