
from .InstacartDataset import InstacartDataset

import pytest

import numpy as np
import pandas as pd


@pytest.fixture
def ds_train(test_data_dir):
    return InstacartDataset().from_dir(test_data_dir)


def test_InstacartDataset_from_dir_with_inner_ids(test_data_dir):
    ds = InstacartDataset(generate_inner_ids=True).from_dir(test_data_dir)

    assert type(ds) == InstacartDataset
    assert type(ds.df_trns) == pd.DataFrame
    assert np.issubdtype(ds.df_trns.uid.dtype, np.integer)
    assert ds.df_trns.uid.values[0] == 0
    assert np.issubdtype(ds.df_trns.iid.dtype, np.integer)
    assert ds.df_trns.iid.values[0] == 0
    assert type(ds.uid_raw_to_inner) == pd.Series
    assert type(ds.iid_raw_to_inner) == pd.Series
    assert type(ds.uid_inner_to_raw) == pd.Series
    assert type(ds.iid_inner_to_raw) == pd.Series


def test_InstacartDataset_from_dir_with_raw_ids(test_data_dir):
    ds = InstacartDataset().from_dir(test_data_dir)

    assert type(ds) == InstacartDataset
    assert type(ds.df_trns) == pd.DataFrame
    assert ds.uid_raw_to_inner is None
    assert ds.iid_raw_to_inner is None
    assert ds.uid_inner_to_raw is None
    assert ds.iid_inner_to_raw is None


def test_InstacartDataset_repr(ds_train):
    ds_empty = InstacartDataset()
    expected_1 = '<InstacartDataset transactions_data=None>'
    assert repr(ds_empty) == expected_1

    expected_2 = '<InstacartDataset transactions_data=(1468, 9)>'
    assert repr(ds_train) == expected_2


def test_InstacartDataset_show_progress(capsys, test_data_dir):
    InstacartDataset(show_progress=True).from_dir(test_data_dir)
    out_1, err_1 = capsys.readouterr()
    assert 'Reading ' in out_1
    assert 'Preprocessing ' in out_1
    assert err_1 == ''

    InstacartDataset(show_progress=False).from_dir(test_data_dir)
    outerr_2 = capsys.readouterr()
    assert outerr_2 == ('', '')




