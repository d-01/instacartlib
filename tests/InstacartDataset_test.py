
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


def test_InstacartDataset_usage(test_data_dir):
    icds = InstacartDataset(verbose=1)
    assert type(icds.df_trns) == pd.DataFrame
    assert type(icds.df_prod) == pd.DataFrame
    assert icds.df_trns.shape == (0, 0)
    assert icds.df_prod.shape == (0, 0)
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
    assert icds.df_trns.shape == (1468, 9)
    assert icds.df_trns.columns.to_list() == ['oid', 'uid', 'iord', 'iid',
        'reord', 'dow', 'hour', 'days_prev', 'cart_pos']
    assert icds.df_prod.shape == (577, 6)
    assert icds.df_prod.columns.to_list() == ['iid', 'dept_id', 'aisle_id',
        'dept', 'aisle', 'product']
    assert icds.get_dataframes() == {
        'df_trns': icds.df_trns,
        'df_prod': icds.df_prod,
    }
    assert icds.dataframes == {
        'df_trns': icds.df_trns,
        'df_prod': icds.df_prod,
    }


def test_InstacartDataset_repr():
    InstacartDataset_repr = repr(InstacartDataset())
    assert InstacartDataset_repr.startswith('<InstacartDataset')
    assert InstacartDataset_repr.endswith('>')
    assert 'transactions=' in InstacartDataset_repr


def test_InstacartDataset_info(capsys):
    inst = InstacartDataset()
    capsys.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert inst.info() is inst
    out, err = capsys.readouterr()
    assert out != ''
    assert err == ''
