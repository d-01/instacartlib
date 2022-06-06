
from instacartlib.Products import Products

from .conftest import GdownCachedDownloadIsCalled

import pytest

@pytest.fixture
def products(test_data_dir):
    return Products().from_dir(test_data_dir)


def test_Products_load_from_gdrive(fake_gdown_cached_download):
    with pytest.raises(GdownCachedDownloadIsCalled):
        Products().load_from_gdrive('.')

def test_Products_repr(products):
    products_empty = Products()
    expected_1 = '<Products df=None>'
    assert repr(products_empty) == expected_1

    repr_ = repr(products)
    assert repr_.startswith('<Products df=')
    assert repr_.endswith('>')