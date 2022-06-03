import inspect
import pytest


def test_imports():
    from instacartlib import InstacartDataset
    assert inspect.isclass(InstacartDataset)
