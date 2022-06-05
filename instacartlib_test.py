import inspect
import pytest


def test_import_InstacartDataset():
    from instacartlib import InstacartDataset
    assert inspect.isclass(InstacartDataset)


def test_imports_FeaturesDataset():
    from instacartlib import FeaturesDataset
    assert inspect.isclass(FeaturesDataset)

