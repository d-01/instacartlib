
from .Transactions import Transactions

import pytest

import numpy as np
import pandas as pd


@pytest.fixture
def ds_train(test_data_dir):
    return Transactions().from_dir(test_data_dir)


def test_Transactions_from_dir(test_data_dir):
    ds = Transactions().from_dir(test_data_dir)
    assert type(ds) == Transactions
    assert type(ds.df_trns) == pd.DataFrame


def test_Transactions_repr(ds_train):
    ds_empty = Transactions()
    expected_1 = '<Transactions df_trns=None>'
    assert repr(ds_empty) == expected_1

    expected_2 = '<Transactions df_trns=(1468, 9)>'
    assert repr(ds_train) == expected_2


def test_Transactions_show_progress(capsys, test_data_dir):
    Transactions(show_progress=True).from_dir(test_data_dir)
    out_1, err_1 = capsys.readouterr()
    assert 'Reading ' in out_1
    assert 'Preprocessing ' in out_1
    assert err_1 == ''

    Transactions(show_progress=False).from_dir(test_data_dir)
    outerr_2 = capsys.readouterr()
    assert outerr_2 == ('', '')




