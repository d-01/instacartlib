
from unittest.mock import patch
from instacartlib.Transactions import Transactions

from .conftest import GdownCachedDownloadIsCalled

import os
import numpy as np
import pandas as pd

import pytest


@pytest.fixture
def transactions(test_data_dir):
    return Transactions().from_dir(test_data_dir)


def test_Transactions_from_dir(test_data_dir):
    ds = Transactions().from_dir(test_data_dir)
    assert type(ds) == Transactions
    assert type(ds.df) == pd.DataFrame


def test_Transactions_iord_column(transactions):
    orders = transactions.df.drop_duplicates('oid')
    user_1_last_3_orders = orders[orders.uid == 1][-3:]
    assert user_1_last_3_orders.iord.to_list() == [2, 1, 0]

    user_2_last_3_orders = orders[orders.uid == 2][-3:]
    assert user_2_last_3_orders.iord.to_list() == [2, 1, 0]


def test_Transactions_repr(transactions):
    transactions_empty = Transactions()
    expected_1 = '<Transactions df=None>'
    assert repr(transactions_empty) == expected_1

    repr_ = repr(transactions)
    assert repr_.startswith('<Transactions df=')
    assert repr_.endswith('>')


def test_Transactions_show_progress(capsys, test_data_dir):
    Transactions(show_progress=True).from_dir(test_data_dir)
    out_1, err_1 = capsys.readouterr()
    assert 'Reading ' in out_1
    assert 'Preprocessing ' in out_1
    assert err_1 == ''

    Transactions(show_progress=False).from_dir(test_data_dir)
    outerr_2 = capsys.readouterr()
    assert outerr_2 == ('', '')


def test_Transactions_load_from_gdrive():
    trns = Transactions()
    with patch('gdown.cached_download') as mock_method:
        assert trns.load_from_gdrive('abc') is trns
    args, kwargs = mock_method.call_args
    assert 'path' in kwargs
    assert kwargs['path'] == os.path.join('abc', 'transactions.csv.zip')


def test_Transactions_drop_last_orders_n_5(transactions):
    """ First 2 users' orders:
    ```
             oid  uid  iord
    0    2539329    1     9
    5    2398795    1     8
    11    473747    1     7
    16   2254736    1     6
    21    431534    1     5
    29   3367565    1     4
    33    550135    1     3
    38   3108588    1     2
    44   2295261    1     1
    50   2550362    1     0
    59   2168274    2    13
    72   1501582    2    12
    78   1901567    2    11
    83    738281    2    10
    96   1673511    2     9
    109  1199898    2     8
    130  3194192    2     7
    144   788338    2     6
    160  1718559    2     5
    186  1447487    2     4
    195  1402090    2     3
    210  3186735    2     2
    229  3268552    2     1
    238   839880    2     0
    ...
    ```
    """
    transactions.drop_last_orders(n=5)

    df_user1 = transactions.df[transactions.df.uid == 1]

    assert 2539329 in df_user1.oid.values  #  1st of 10
    assert  431534 in df_user1.oid.values  #  5th of 10
    assert 3367565 not in df_user1.oid.values      #  6th of 10
    assert 2550362 not in df_user1.oid.values      # 10th of 10

    # dynamic column is updated automatically
    assert df_user1.iord.drop_duplicates().to_list() == [4, 3, 2, 1, 0]

    df_user2 = transactions.df[transactions.df.uid == 2]

    assert 2168274 in df_user2.oid.values  #  1st of 14
    assert 1718559 in df_user2.oid.values  #  9th of 14
    assert 1447487 not in df_user2.oid.values      # 10th of 14
    assert  839880 not in df_user2.oid.values      # 14th of 14

    # dynamic column is updated automatically
    assert (df_user2.iord.drop_duplicates().to_list()
        == [8, 7, 6, 5, 4, 3, 2, 1, 0])


def test_Transactions_keep_last_orders_n_5(transactions):
    transactions.keep_last_orders(n=5)

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


@pytest.mark.parametrize('n_last_orders,n_trns', [
    (0, 0),
    (1, 90),
    (2, 164),
])
def test_Transactions_get_last_orders(transactions, n_last_orders, n_trns):
    df = transactions.get_last_orders(n_last_orders)
    assert len(df) == n_trns


def test_Transactions_drop_last_orders_n_required(transactions):
    with pytest.raises(TypeError,
            match=r'missing \d+ required positional argument'):
        transactions.drop_last_orders()


def test_Transactions_keep_last_orders_n_required(transactions):
    with pytest.raises(TypeError,
            match=r'missing \d+ required positional argument'):
        transactions.keep_last_orders()
