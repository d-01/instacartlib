
from instacartlib.InstacartDataset import InstacartDataset

from instacartlib.Transactions import Transactions
from instacartlib.Products import Products

import warnings

import pandas as pd

import pytest


@pytest.fixture
def transactions_load_from_gdrive_patch(monkeypatch, has_been_called):
    def load_from_gdrive(self, to_dir):
        has_been_called(id='Transactions.load_from_gdrive').call()
    monkeypatch.setattr(Transactions, 'load_from_gdrive', load_from_gdrive)


@pytest.fixture
def products_load_from_gdrive_patch(monkeypatch, has_been_called):
    def load_from_gdrive(self, to_dir):
        has_been_called(id='Products.load_from_gdrive').call()
    monkeypatch.setattr(Products, 'load_from_gdrive', load_from_gdrive)


def test_InstacartDataset_download(
        transactions_load_from_gdrive_patch,
        products_load_from_gdrive_patch,
        has_been_called):
    InstacartDataset().download()
    assert has_been_called('Transactions.load_from_gdrive').times == 1
    assert has_been_called('Products.load_from_gdrive').times == 1
    assert has_been_called.total == 2


@pytest.fixture
def required_trns_cols():
    return {
        'oid',
        'uid',
        'iord',
        'iid',
        'reord',
        # 'dow',
        # 'hour',
        'days_prev',
        'cart_pos',
    }


@pytest.fixture
def n_trns_cols(required_trns_cols):
    return len(required_trns_cols)


def test_InstacartDataset_train_default_usage(test_data_dir, required_trns_cols,
        n_trns, n_trns_cols):
    icds = InstacartDataset(verbose=1)
    assert type(icds.df_trns) == pd.DataFrame
    assert type(icds.df_prod) == pd.DataFrame
    assert type(icds.df_trns_target) == pd.DataFrame
    assert icds.df_trns.shape == (0, 0)
    assert icds.df_prod.shape == (0, 0)
    assert icds.df_trns_target.shape == (0, 0)
    assert icds.get_dataframes() == {
        'df_trns': icds.df_trns,
        'df_prod': icds.df_prod,
    }
    assert icds.dataframes == {
        'df_trns': icds.df_trns,
        'df_prod': icds.df_prod,
    }

    icds.read_dir(test_data_dir)
    assert type(icds.df_trns) == pd.DataFrame
    assert type(icds.df_prod) == pd.DataFrame
    assert type(icds.df_trns_target) == pd.DataFrame
    assert icds.df_trns_target.shape == (0, 0)
    assert icds.df_trns.shape == (n_trns, n_trns_cols)
    assert required_trns_cols - set(icds.df_trns.columns) == set()
    df_orders_uid_1 = icds.df_trns[icds.df_trns.uid == 1].drop_duplicates('oid')
    assert (df_orders_uid_1.iord.to_list() == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    assert icds.df_prod.shape == (577, 6)
    assert icds.df_prod.columns.to_list() == ['iid', 'dept_id', 'aisle_id',
        'dept', 'aisle', 'prod']
    assert icds.get_dataframes() == {
        'df_trns': icds.df_trns,
        'df_prod': icds.df_prod,
    }
    assert icds.dataframes == {
        'df_trns': icds.df_trns,
        'df_prod': icds.df_prod,
    }


def test_InstacartDataset_train_true_usage(test_data_dir, n_trns, n_trns_cols,
        n_trns_target, required_trns_cols):
    icds = InstacartDataset(train=True, verbose=1)
    assert type(icds.df_trns) == pd.DataFrame
    assert type(icds.df_prod) == pd.DataFrame
    assert type(icds.df_trns_target) == pd.DataFrame
    assert icds.df_trns.shape == (0, 0)
    assert icds.df_prod.shape == (0, 0)
    assert icds.df_trns_target.shape == (0, 0)
    assert icds.get_dataframes() == {
        'df_trns': icds.df_trns,
        'df_prod': icds.df_prod,
        'df_trns_target': icds.df_trns_target,
    }
    assert icds.dataframes == {
        'df_trns': icds.df_trns,
        'df_prod': icds.df_prod,
        'df_trns_target': icds.df_trns_target,
    }

    icds.read_dir(test_data_dir)
    assert type(icds.df_trns) == pd.DataFrame
    assert type(icds.df_prod) == pd.DataFrame
    assert type(icds.df_trns_target) == pd.DataFrame
    assert icds.df_trns.shape == (n_trns - n_trns_target, n_trns_cols)
    assert required_trns_cols - set(icds.df_trns.columns) == set()
    df_orders_uid_1 = icds.df_trns[icds.df_trns.uid == 1].drop_duplicates('oid')
    assert (df_orders_uid_1.iord.to_list() == [8, 7, 6, 5, 4, 3, 2, 1, 0])
    assert icds.df_prod.shape == (577, 6)
    assert icds.df_prod.columns.to_list() == ['iid', 'dept_id', 'aisle_id',
        'dept', 'aisle', 'prod']
    assert icds.df_trns_target.shape == (n_trns_target, n_trns_cols)
    assert required_trns_cols - set(icds.df_trns_target.columns) == set()
    assert (icds.df_trns_target.iord == 0).all()
    assert icds.get_dataframes() == {
        'df_trns': icds.df_trns,
        'df_prod': icds.df_prod,
        'df_trns_target': icds.df_trns_target,
    }
    assert icds.dataframes == {
        'df_trns': icds.df_trns,
        'df_prod': icds.df_prod,
        'df_trns_target': icds.df_trns_target,
    }

def test_InstacartDataset_repr():
    InstacartDataset_repr = repr(InstacartDataset())
    assert InstacartDataset_repr.startswith('<InstacartDataset')
    assert InstacartDataset_repr.endswith('>')
    assert 'transactions=' in InstacartDataset_repr


def test_InstacartDataset_info(capsys):
    inst = InstacartDataset(train=True)
    capsys.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert inst.info() is inst
    out, err = capsys.readouterr()
    assert out != ''
    assert err == ''
