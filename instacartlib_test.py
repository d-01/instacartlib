import inspect
import pytest


@pytest.mark.skip('unstable')
def test_imports():
    from instacartlib import InstacartDataset
    assert inspect.isclass(InstacartDataset)
