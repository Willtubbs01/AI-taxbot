from taxmoe.training.run_models import TrainingProgress
from taxmoe.training.progress import commit_step
def test_commit():
    p=commit_step(TrainingProgress(),examples=2,tokens=100,supervised_tokens=100,unique_train_tokens=1000); assert p.global_step==1 and p.effective_passes==.1
