import inspect


def test_imports():
    from instacartlib import InstacartDataset
    assert inspect.isclass(InstacartDataset)
