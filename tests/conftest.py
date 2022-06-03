
from instacartlib.Transactions import read_transactions_csv
from instacartlib.Transactions import _preprocess_raw_columns as trns_preprocess

from instacartlib.Products import read_products_csv
from instacartlib.Products import preprocess_raw_columns as prods_preprocess

import datetime
import random
import shutil
import pathlib

import pytest


CLEANUP_TEMP_DIR = False

class GdownCachedDownloadIsCalled(Exception):
    pass


@pytest.fixture
def has_been_called():
    """ Usage example:
    ```python
    # Scenario A
    has_been_called().call()  # without id
    has_been_called().call()  # without id
    assert has_been_called().times == 2

    # Scenario B
    has_been_called(id='method_A').call()
    has_been_called(id='method_B').call()
    assert has_been_called('method_A').times == 1
    assert has_been_called('method_B').times == 1
    assert has_been_called.total == 2  # no unexpected calls have been made
    ```
    """
    class CallCounter:
        total = 0
        counters = {}  # global storage for counters with id specified
        def __init__(self, id=None):
            self.id = id
            self.counters[id] = self.counters.get(id, 0)
            self.times = self.counters[id]
        def call(self):
            CallCounter.total += 1
            self.counters[self.id] += 1
    return CallCounter


@pytest.fixture
def test_data_dir():
    return pathlib.Path(r'tests/testing_data')


@pytest.fixture
def fake_gdown_cached_download(monkeypatch):
    def ff(id=None, path=None, md5=None, quiet=None):  # fake function
        assert id is not None
        assert path is not None
        assert md5 is not None
        raise GdownCachedDownloadIsCalled()

    monkeypatch.setattr('gdown.cached_download', ff)


@pytest.fixture
def tmp_dir(tmp_path):
    datetime_8_6_6 = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    random_8 = random.randint(0, 0xffff_ffff)
    path = tmp_path / f'{datetime_8_6_6}_{random_8:X}'
    path.mkdir()
    yield path
    if CLEANUP_TEMP_DIR:
        shutil.rmtree(path)


################################################################################
# Transactions
################################################################################

@pytest.fixture
def transactions_csv_path(test_data_dir):
    file_path = test_data_dir / 'transactions.csv'
    assert file_path.exists()
    return file_path


@pytest.fixture
def df_trns_raw(transactions_csv_path):
    return read_transactions_csv(transactions_csv_path)


@pytest.fixture
def df_trns(df_trns_raw):
    return trns_preprocess(df_trns_raw)


@pytest.fixture
def ui_index(df_trns):
    ui = df_trns.drop_duplicates(['uid', 'iid']).set_index(['uid', 'iid'])
    return ui.index


@pytest.fixture
def uids(df_trns):
    return df_trns.uid.unique()


@pytest.fixture
def iids(df_trns):
    return df_trns.iid.unique()

################################################################################
# Products
################################################################################

@pytest.fixture
def products_csv_path(test_data_dir):
    file_path = test_data_dir / 'products.csv'
    assert file_path.exists()
    return file_path


@pytest.fixture
def df_prod_raw(products_csv_path):
    return read_products_csv(products_csv_path)


@pytest.fixture
def df_prod(df_prod_raw):
    return prods_preprocess(df_prod_raw)

