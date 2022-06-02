from instacartlib.InstacartDataset import InstacartDataset
from instacartlib.Transactions import Transactions
from instacartlib.Products import Products

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


@pytest.fixture
def inst_after_read_dir():
    inst = InstacartDataset()
    inst.read_dir('tests/testing_data')
    return inst


def test_InstacartDataset_from_dir(inst_after_read_dir):
    type(inst_after_read_dir.df_ui) == pd.DataFrame
    type(inst_after_read_dir.ui_index) == pd.MultiIndex


def test_InstacartDataset_download(
        transactions_load_from_gdrive_patch,
        products_load_from_gdrive_patch,
        has_been_called):
    InstacartDataset().download()
    assert has_been_called('Transactions.load_from_gdrive').times == 1
    assert has_been_called('Products.load_from_gdrive').times == 1
    assert has_been_called.total == 2
