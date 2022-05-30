
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


def test_Transactions_drop_orders(transactions):
    transactions.drop_orders(keep_n=5)

    df_user1 = transactions.df[transactions.df.uid == 1]

    assert 2539329 not in df_user1.oid.values  #  1st of 10
    assert  431534 not in df_user1.oid.values  #  5th of 10
    assert 3367565 in df_user1.oid.values      #  6th of 10
    assert 2550362 in df_user1.oid.values      # 10th of 10

    # dynamic column is updated automatically
    assert df_user1.iord.drop_duplicates().to_list() == [4, 3, 2, 1, 0]

    df_user2 = transactions.df[transactions.df.uid == 2]

    assert 2168274 not in df_user2.oid.values  #  1st of 13
    assert 1718559 not in df_user2.oid.values  #  8th of 13
    assert 1447487 in df_user2.oid.values      #  9th of 13
    assert  839880 in df_user2.oid.values      # 13th of 13

    # dynamic column is updated automatically
    assert df_user2.iord.drop_duplicates().to_list() == [4, 3, 2, 1, 0]
