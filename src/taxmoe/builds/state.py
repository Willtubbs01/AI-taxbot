from .models import BuildStage, StageStatus


def fresh_stage_state():
    return {stage: StageStatus.PENDING for stage in BuildStage}
