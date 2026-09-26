from __future__ import annotations
from pydantic import Field
from taxmoe.schemas.common import TaxMoEModel
class TrainingLossOutput(TaxMoEModel):
    total_loss: object
    lm_loss: object
    auxiliary_losses: dict[str,object]=Field(default_factory=dict)
    model_config={'arbitrary_types_allowed':True,'extra':'forbid'}
class DenseCausalLMAdapter:
    def compute_loss(self,model,batch):
        if 'ngram_feature_ids' in batch: raise ValueError('DENSE-CPT-NGRAM-INPUT-FORBIDDEN')
        out=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],labels=batch['labels'])
        return TrainingLossOutput(total_loss=out.loss,lm_loss=out.loss)
