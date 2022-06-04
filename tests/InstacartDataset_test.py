from instacartlib.InstacartDataset import DataNotLoaded
from instacartlib.InstacartDataset import ExtractorCallError
from instacartlib.InstacartDataset import ExtractorInvalidOutputError
from instacartlib.InstacartDataset import ExtractorExistsError
from instacartlib.InstacartDataset import _use_extractor
from instacartlib.InstacartDataset import _assert_extractor_output
from instacartlib.InstacartDataset import _process_extractor_output
from instacartlib.InstacartDataset import _get_feature_cache_path
from instacartlib.InstacartDataset import InstacartDataset

from instacartlib.Transactions import Transactions
from instacartlib.Products import Products

import warnings
import sys
import io

from pathlib import Path
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
def extractor_valid(has_been_called):
    def func(ui_index, df_trns, df_prod):
        has_been_called(id='extractor_valid').call()
        return pd.DataFrame(index=ui_index).assign(feature_A=0)
    return func


@pytest.fixture
def extractor_valid_duplicate():
    def func(ui_index, df_trns, df_prod):
        return pd.DataFrame(index=ui_index).assign(feature_A=1)
    return func


@pytest.fixture
def extractor_broken():
    def func(ui_index, df_trns, df_prod):
        raise Exception('broken extractor')
    return func


@pytest.fixture
def extractor_invalid():
    def func(ui_index, df_trns, df_prod):
        return pd.DataFrame(index=[1, 2, 3]).assign(feature_B=0)
    return func


def test_use_extractor_invalid(extractor_invalid, ui_index, df_trns, df_prod):
    _use_extractor(extractor_invalid, ui_index, df_trns, df_prod)


def test_use_extractor_broken(extractor_broken, ui_index, df_trns, df_prod):
    with pytest.raises(ExtractorCallError, match='broken extractor'):
        _use_extractor(extractor_broken, ui_index, df_trns, df_prod)


def test_assert_extractor_output_valid(extractor_valid, ui_index, df_trns,
        df_prod):
    output = extractor_valid(ui_index, df_trns, df_prod)
    _assert_extractor_output(output, ui_index)


def test_assert_extractor_output_invalid(extractor_invalid, ui_index, df_trns,
        df_prod):
    output = extractor_invalid(ui_index, df_trns, df_prod)
    not_a_dataframe = 0
    with pytest.raises(ExtractorInvalidOutputError):
        _assert_extractor_output(not_a_dataframe, ui_index)
    with pytest.raises(ExtractorInvalidOutputError):
        _assert_extractor_output(output, ui_index)


def test_process_extractor_output():
    extractor_output = pd.DataFrame(columns=list('abc'))
    feature_registry = {'c': 'extr1'}

    output, old_new_name_dict = _process_extractor_output(extractor_output,
        feature_registry)
    assert output.columns.to_list() == ['a', 'b', 'c_1']
    assert old_new_name_dict == {'c': 'c_1'}


def test_get_feature_cache_path():
    output = _get_feature_cache_path('dir', 'name')
    assert isinstance(output, Path)
    assert output.as_posix() == 'dir/name.zip'


def test_InstacartDataset_download(
        transactions_load_from_gdrive_patch,
        products_load_from_gdrive_patch,
        has_been_called):
    InstacartDataset().download()
    assert has_been_called('Transactions.load_from_gdrive').times == 1
    assert has_been_called('Products.load_from_gdrive').times == 1
    assert has_been_called.total == 2


def test_InstacartDataset_usage_no_cache(test_data_dir, extractor_valid,
        has_been_called):
    inst = InstacartDataset(verbose=1)
    assert type(inst.df_trns) == pd.DataFrame
    assert type(inst.df_prod) == pd.DataFrame
    assert type(inst.df_ui) == pd.DataFrame
    assert inst.df_trns.shape == (0, 0)
    assert inst.df_prod.shape == (0, 0)
    assert inst.df_ui.shape == (0, 0)

    with pytest.raises(DataNotLoaded):
        inst._assert_dataframes_initialized()

    inst.read_dir(test_data_dir)
    assert len(inst.df_ui.columns) == 0
    assert len(inst.df_ui.index) == 624

    inst._feature_extractors = {}
    inst.extract_features()
    assert len(inst.df_ui.columns) == 0

    inst.register_feature_extractors({"extractor_1": extractor_valid})
    assert 'extractor_1' in inst._feature_extractors

    with pytest.raises(ExtractorExistsError):
        inst.register_feature_extractors({"extractor_1": 'any'})

    assert inst._feature_registry == {}

    assert has_been_called(id='extractor_valid').times == 0
    inst.extract_features()
    assert has_been_called(id='extractor_valid').times == 1
    assert inst._feature_registry == {'feature_A': 'extractor_1'}
    assert inst.df_ui.columns.to_list() == ['feature_A']
    inst.extract_features()
    assert has_been_called(id='extractor_valid').times == 2
    assert inst._feature_registry == {
        'feature_A': 'extractor_1',
        'feature_A_1': 'extractor_1',
    }
    assert inst.df_ui.columns.to_list() == ['feature_A', 'feature_A_1']


def test_InstacartDataset_usage_with_cache(test_data_dir, tmp_dir,
        extractor_valid, extractor_invalid, extractor_broken,
        extractor_valid_duplicate, has_been_called):
    inst = InstacartDataset(features_cache_dir=tmp_dir)
    inst.read_dir(test_data_dir)
    inst._feature_extractors = {}

    inst.register_feature_extractors({
        "extractor_1": extractor_valid,
        "extractor_2": extractor_invalid,
        "extractor_3": extractor_broken,
        "extractor_4": extractor_valid_duplicate,
    })
    assert (set(inst._feature_extractors) == {"extractor_1", "extractor_2",
        "extractor_3", "extractor_4"})
    assert inst._feature_registry == {}
    assert inst.df_ui.columns.to_list() == []

    assert has_been_called(id='extractor_valid').times == 0
    inst.extract_features()
    assert has_been_called(id='extractor_valid').times == 1
    assert (tmp_dir / 'extractor_1.zip').exists()
    assert (tmp_dir / 'extractor_2.zip').exists()
    assert (tmp_dir / 'extractor_3.zip').exists() == False
    assert (tmp_dir / 'extractor_4.zip').exists()
    assert inst._feature_registry == {
        'feature_A': 'extractor_1',
        'feature_A_1': 'extractor_4',
    }
    assert inst.df_ui.columns.to_list() == ['feature_A', 'feature_A_1']
    inst.extract_features()
    assert has_been_called(id='extractor_valid').times == 1
    assert inst._feature_registry == {
        'feature_A': 'extractor_1',
        'feature_A_1': 'extractor_4',
        'feature_A_2': 'extractor_1',
        'feature_A_3': 'extractor_4',
    }
    assert inst.df_ui.columns.to_list() == ['feature_A', 'feature_A_1',
        'feature_A_2', 'feature_A_3']


def test_InstacartDataset_repr():
    InstacartDataset_repr = repr(InstacartDataset())
    assert InstacartDataset_repr.startswith('<InstacartDataset')
    assert InstacartDataset_repr.endswith('>')
    assert 'transactions=' in InstacartDataset_repr
    assert 'features=' in InstacartDataset_repr


def test_InstacartDataset_info(capsys):
    inst = InstacartDataset()
    capsys.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert inst.info() is inst
    out, err = capsys.readouterr()
    assert out != ''
    assert err == ''
