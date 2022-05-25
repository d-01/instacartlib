
from instacartlib.Transactions import Transactions

from .conftest import GdownCachedDownloadIsCalled

import pytest

import numpy as np
import pandas as pd


@pytest.fixture
def transactions(test_data_dir):
    return Transactions().from_dir(test_data_dir)


def test_Transactions_from_dir(test_data_dir):
    ds = Transactions().from_dir(test_data_dir)
    assert type(ds) == Transactions
    assert type(ds.df) == pd.DataFrame


def test_Transactions_repr(transactions):
    transactions_empty = Transactions()
    expected_1 = '<Transactions df=None>'
    assert repr(transactions_empty) == expected_1

    expected_2 = '<Transactions df=(1468, 9)>'
    assert repr(transactions) == expected_2


def test_Transactions_show_progress(capsys, test_data_dir):
    Transactions(show_progress=True).from_dir(test_data_dir)
    out_1, err_1 = capsys.readouterr()
    assert 'Reading ' in out_1
    assert 'Preprocessing ' in out_1
    assert err_1 == ''

    Transactions(show_progress=False).from_dir(test_data_dir)
    outerr_2 = capsys.readouterr()
    assert outerr_2 == ('', '')


def test_Transactions_load_from_gdrive(fake_gdown_cached_download):
    with pytest.raises(GdownCachedDownloadIsCalled):
        Transactions().load_from_gdrive('.')



