
from instacartlib.Products import Products

import os
from unittest.mock import patch

import pytest


@pytest.fixture
def products(test_data_dir):
    return Products().from_dir(test_data_dir)


def test_Products_load_from_gdrive():
    trns = Products()
    with patch('gdown.cached_download') as mock_method:
        assert trns.load_from_gdrive('abc') is trns
    args, kwargs = mock_method.call_args
    assert 'path' in kwargs
    assert kwargs['path'] == os.path.join('abc', 'products.csv.zip')


def test_Products_repr(products):
    products_empty = Products()
    expected_1 = '<Products df=None>'
    assert repr(products_empty) == expected_1

    repr_ = repr(products)
    assert repr_.startswith('<Products df=')
    assert repr_.endswith('>')