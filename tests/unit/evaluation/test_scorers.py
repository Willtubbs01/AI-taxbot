from taxmoe.evaluation.scorers.sets import set_metrics
from taxmoe.evaluation.scorers.protocol import strict_json_metrics
from taxmoe.evaluation.comparison import paired_counts
def test_set_metrics_empty(): assert set_metrics([],[])['f1']==1
def test_json_strict(): assert strict_json_metrics('{"status":"OK"}',required_keys=['status'])['schema_valid']==1
def test_paired(): assert paired_counts([1,1,0,0],[1,0,1,0])=={'unchanged_correct':1,'regression':1,'fix':1,'unchanged_wrong':1}
