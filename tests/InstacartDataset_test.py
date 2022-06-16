
from instacartlib.InstacartDataset import InstacartDataset

from instacartlib.Transactions import Transactions
from instacartlib.Products import Products

import warnings
import inspect

import pandas as pd

import pytest


def is_reverse_count(seq, start=0):
    lseq = list(seq)
    reverse_count = list(range(start, len(lseq) + start))[::-1]
    return lseq == reverse_count


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


@pytest.fixture
def inst_train_false_loaded(test_data_dir):
    return InstacartDataset(train=False).read_dir(test_data_dir)


@pytest.fixture
def inst_train_loaded(test_data_dir):
    return InstacartDataset(train=True).read_dir(test_data_dir)


@pytest.fixture
def inst_train_n_order_limit_2_loaded(test_data_dir):
    inst = InstacartDataset(train=True, n_orders_limit=2)
    inst.read_dir(test_data_dir)
    return inst


def test_InstacartDataset_init():
    inst = InstacartDataset()
    dataframes = ['df_ord', 'df_trns', 'df_trns_target', 'df_prod']
    for name in dataframes:
        df = getattr(inst, name)
        assert type(df) == pd.DataFrame
        assert df.shape == (0, 0)


def test_InstacartDataset_train_false_dataframes_shapes(
        inst_train_false_loaded, required_trns_cols,
        required_ord_cols, required_prod_cols, n_trns, n_trns_target):
    inst = inst_train_false_loaded
    assert required_ord_cols - set(inst.df_ord.columns) == set()
    assert required_trns_cols - set(inst.df_trns.columns) == set()
    assert required_trns_cols - set(inst.df_trns_target.columns) == set()
    assert required_prod_cols - set(inst.df_prod.columns) == set()

    assert len(inst.df_trns) == n_trns
    assert len(inst.df_trns_target) == 0
    assert len(inst.df_ord) != 0
    assert len(inst.df_prod) != 0


def test_InstacartDataset_train_false_preprocess(inst_train_false_loaded):
    inst = inst_train_false_loaded
    assert (set(inst.df_trns.order_id.unique())
        == set(inst.df_ord.order_id.unique()))

    u1_uid = inst.df_trns.uid.values[0]
    u1_orders = (inst
        .df_trns[inst.df_trns.uid==u1_uid]
        .drop_duplicates('order_id')
    )
    assert is_reverse_count(u1_orders.order_r, start=1)


def test_InstacartDataset_train_true_dataframes_shapes(inst_train_loaded,
        required_trns_cols, required_ord_cols, required_prod_cols, n_trns,
        n_trns_target):
    inst = inst_train_loaded
    assert required_ord_cols - set(inst.df_ord.columns) == set()
    assert required_trns_cols - set(inst.df_trns.columns) == set()
    assert required_trns_cols - set(inst.df_trns_target.columns) == set()
    assert required_prod_cols - set(inst.df_prod.columns) == set()

    assert len(inst.df_trns) == n_trns - n_trns_target
    assert len(inst.df_trns_target) == n_trns_target
    assert len(inst.df_ord) != 0
    assert len(inst.df_prod) != 0


def test_InstacartDataset_n_order_limit(inst_train_n_order_limit_2_loaded,
        n_trns_target):
    inst = inst_train_n_order_limit_2_loaded

    order_ids_two_most_recent = pd.Series(
        [3108588, 2295261, 3186735, 3268552, 3160850, 676467])
    order_ids_most_recent_3rd = pd.Series(
        [550135, 1402090, 3225766])

    assert order_ids_two_most_recent.isin(inst.df_trns.order_id).all() == True
    assert order_ids_most_recent_3rd.isin(inst.df_trns.order_id).any() == False
    assert order_ids_two_most_recent.isin(inst.df_ord.order_id).all() == True
    assert order_ids_most_recent_3rd.isin(inst.df_ord.order_id).any() == False
    assert len(inst.df_trns_target) == n_trns_target
    assert len(inst.df_prod) != 0


def test_InstacartDataset_train_true_preprocess(inst_train_loaded):
    inst = inst_train_loaded
    assert (set(inst.df_trns.order_id.unique())
        == set(inst.df_ord.order_id.unique()))

    u1_uid = inst.df_trns.uid.values[0]
    u1_orders = (inst
        .df_trns[inst.df_trns.uid==u1_uid]
        .drop_duplicates('order_id')
    )
    assert is_reverse_count(u1_orders.order_r, start=1)


def test_InstacartDataset_train_default():
    inst = InstacartDataset()
    assert inst.train == False


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


def test_InstacartDataset_read_dir_reduced_default():
    read_dir_method = InstacartDataset.read_dir
    reduced_default_arg_value = (
        inspect
        .signature(read_dir_method)
        .parameters['reduced']
        .default
    )
    assert reduced_default_arg_value == False
