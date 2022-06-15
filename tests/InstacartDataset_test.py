
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
def required_ord_cols():
    return {
        'order_id',
        'uid',
        'order_n',
        'days_since_prior_order',
    }


@pytest.fixture
def required_trns_cols():
    return {
        'order_id',
        'uid',
        'iid',
        'cart_pos',
    }


@pytest.fixture
def required_prod_cols():
    return {
        'iid',
    }


def test_InstacartDataset_init():
    inst = InstacartDataset()
    dataframes = ['df_ord', 'df_trns', 'df_trns_target', 'df_prod']
    for name in dataframes:
        df = getattr(inst, name)
        assert type(df) == pd.DataFrame
        assert df.shape == (0, 0)


def test_InstacartDataset_train_default_read_dir(test_data_dir,
        required_trns_cols, required_ord_cols, required_prod_cols, n_trns,
        n_trns_target):
    inst = InstacartDataset().read_dir(test_data_dir)
    assert required_ord_cols - set(inst.df_ord.columns) == set()
    assert required_trns_cols - set(inst.df_trns.columns) == set()
    assert required_trns_cols - set(inst.df_trns_target.columns) == set()
    assert required_prod_cols - set(inst.df_prod.columns) == set()

    assert len(inst.df_trns) == n_trns
    assert len(inst.df_trns_target) == 0


def test_InstacartDataset_train_true_read_dir(test_data_dir,
        required_trns_cols, required_ord_cols, required_prod_cols, n_trns,
        n_trns_target):
    inst = InstacartDataset(train=True).read_dir(test_data_dir)
    assert required_ord_cols - set(inst.df_ord.columns) == set()
    assert required_trns_cols - set(inst.df_trns.columns) == set()
    assert required_trns_cols - set(inst.df_trns_target.columns) == set()
    assert required_prod_cols - set(inst.df_prod.columns) == set()

    assert len(inst.df_trns) == n_trns - n_trns_target
    assert len(inst.df_trns_target) == n_trns_target


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


def test_InstacartDataset_print_text_indent(capsys):
    inst = InstacartDataset(verbose=1)
    capsys.readouterr()
    inst._print('abc\nxyz', indent=2)
    out, err = capsys.readouterr()
    assert out == '  abc\n  xyz\n'
    assert err == ''


def test_InstacartDataset_train_default_dataframes():
    inst = InstacartDataset(train=False)
    dataframes = inst.dataframes
    assert type(dataframes) == dict
    assert 'df_ord' in dataframes
    assert 'df_trns' in dataframes
    assert 'df_prod' in dataframes
    assert 'df_trns_target' not in dataframes


def test_InstacartDataset_train_true_dataframes():
    inst = InstacartDataset(train=True)
    dataframes = inst.dataframes
    assert type(dataframes) == dict
    assert 'df_ord' in dataframes
    assert 'df_trns' in dataframes
    assert 'df_prod' in dataframes
    assert 'df_trns_target' in dataframes

