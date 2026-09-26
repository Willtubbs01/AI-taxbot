import pytest
from taxmoe.experiments.ablations import validate_one_variable_change
def test_one_variable(): assert validate_one_variable_change({'a':1,'b':2},{'a':3,'b':2},'a')==['a']
def test_multi_rejected():
    with pytest.raises(ValueError): validate_one_variable_change({'a':1,'b':2},{'a':3,'b':4},'a')
