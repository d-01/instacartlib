
from .Transactions import read_transactions_csv
from .Transactions import preprocess_raw_columns

import pathlib

import pytest


class GdownCachedDownloadIsCalled(Exception):
    pass



@pytest.fixture
def test_data_dir():
    return pathlib.Path(r'instacartlib/testing_data')


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
    return preprocess_raw_columns(df_trns_raw)


@pytest.fixture
def uids(df_trns):
    return df_trns.uid.unique()


@pytest.fixture
def iids(df_trns):
    return df_trns.iid.unique()


@pytest.fixture
def fake_gdown_cached_download(monkeypatch):
    def ff(id=None, path=None, md5=None, quiet=None):  # fake function
        assert id is not None
        assert path is not None
        assert md5 is not None
        raise GdownCachedDownloadIsCalled()

    monkeypatch.setattr('gdown.cached_download', ff)
