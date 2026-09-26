from .run_models import TrainingProgress
from .sampling import effective_passes

def commit_step(progress:TrainingProgress,*,examples:int,tokens:int,supervised_tokens:int,unique_train_tokens:int):
    return progress.model_copy(update={'global_step':progress.global_step+1,'optimizer_step':progress.optimizer_step+1,'consumed_examples':progress.consumed_examples+examples,'consumed_tokens':progress.consumed_tokens+tokens,'supervised_tokens':progress.supervised_tokens+supervised_tokens,'effective_passes':effective_passes(progress.consumed_tokens+tokens,unique_train_tokens)})
