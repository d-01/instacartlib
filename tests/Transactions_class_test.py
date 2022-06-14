
from unittest.mock import patch
from instacartlib.Transactions import Transactions

import os
import numpy as np
import pandas as pd

import pytest


@pytest.fixture
def transactions(test_data_dir):
    return Transactions().read_dir(test_data_dir)


def test_Transactions_from_dir(test_data_dir):
    ds = Transactions().read_dir(test_data_dir)
    assert type(ds) == Transactions
    assert type(ds.df) == pd.DataFrame


def test_Transactions_repr(transactions):
    transactions_empty = Transactions()
    expected_1 = '<Transactions df=None>'
    assert repr(transactions_empty) == expected_1

    repr_ = repr(transactions)
    assert repr_.startswith('<Transactions df=')
    assert repr_.endswith('>')


def test_Transactions_show_progress(capsys, test_data_dir):
    Transactions(show_progress=True).read_dir(test_data_dir)
    out_1, err_1 = capsys.readouterr()
    assert 'Reading ' in out_1
    assert err_1 == ''

    Transactions(show_progress=False).read_dir(test_data_dir)
    outerr_2 = capsys.readouterr()
    assert outerr_2 == ('', '')


def test_Transactions_load_from_gdrive():
    trns = Transactions()
    with patch('gdown.cached_download') as mock_method:
        assert trns.load_from_gdrive('abc') is trns
    args, kwargs = mock_method.call_args
    assert 'path' in kwargs
    assert kwargs['path'] == os.path.join('abc', 'transactions.csv.zip')
