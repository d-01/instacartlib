
from instacartlib.Products import Products

from .conftest import GdownCachedDownloadIsCalled

import pytest


def test_Products_load_from_gdrive(fake_gdown_cached_download):
    with pytest.raises(GdownCachedDownloadIsCalled):
        Products().load_from_gdrive('.')
