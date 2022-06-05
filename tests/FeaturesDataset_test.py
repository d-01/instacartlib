


from instacartlib.FeaturesDataset import ExtractorCallError
from instacartlib.FeaturesDataset import ExtractorInvalidOutputError
from instacartlib.FeaturesDataset import ExtractorExistsError
from instacartlib.FeaturesDataset import _use_extractor
from instacartlib.FeaturesDataset import _assert_extractor_output
from instacartlib.FeaturesDataset import _process_extractor_output
from instacartlib.FeaturesDataset import _get_feature_cache_path
from instacartlib.FeaturesDataset import FeaturesDataset


from pathlib import Path
import pandas as pd


import pytest


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


def test_FeatureExtractor_usage_no_cache(df_trns, df_prod, extractor_valid,
        has_been_called):
    fsds = FeaturesDataset(df_trns, df_prod, verbose=1)
    assert type(fsds.df_trns) == pd.DataFrame
    assert type(fsds.df_prod) == pd.DataFrame
    assert type(fsds.df_ui) == pd.DataFrame
    assert len(fsds.df_ui.columns) == 0
    assert len(fsds.df_ui.index) == 624

    fsds._feature_extractors = {}
    fsds.extract_features()
    assert len(fsds.df_ui.columns) == 0

    fsds.register_feature_extractors({"extractor_1": extractor_valid})
    assert 'extractor_1' in fsds._feature_extractors

    with pytest.raises(ExtractorExistsError):
        fsds.register_feature_extractors({"extractor_1": 'any'})

    assert fsds._feature_registry == {}

    assert has_been_called(id='extractor_valid').times == 0
    fsds.extract_features()
    assert has_been_called(id='extractor_valid').times == 1
    assert fsds._feature_registry == {'feature_A': 'extractor_1'}
    assert fsds.df_ui.columns.to_list() == ['feature_A']
    fsds.extract_features()
    assert has_been_called(id='extractor_valid').times == 2
    assert fsds._feature_registry == {
        'feature_A': 'extractor_1',
        'feature_A_1': 'extractor_1',
    }
    assert fsds.df_ui.columns.to_list() == ['feature_A', 'feature_A_1']


def test_FeatureExtractor_usage_with_cache(df_trns, df_prod, tmp_dir,
        extractor_valid, extractor_invalid, extractor_broken,
        extractor_valid_duplicate, has_been_called):
    fsds = FeaturesDataset(df_trns, df_prod, features_cache_dir=tmp_dir)
    fsds._feature_extractors = {}

    fsds.register_feature_extractors({
        "extractor_1": extractor_valid,
        "extractor_2": extractor_invalid,
        "extractor_3": extractor_broken,
        "extractor_4": extractor_valid_duplicate,
    })
    assert (set(fsds._feature_extractors) == {"extractor_1", "extractor_2",
        "extractor_3", "extractor_4"})
    assert fsds._feature_registry == {}
    assert fsds.df_ui.columns.to_list() == []

    assert has_been_called(id='extractor_valid').times == 0
    fsds.extract_features()
    assert has_been_called(id='extractor_valid').times == 1
    assert (tmp_dir / 'extractor_1.zip').exists()
    assert (tmp_dir / 'extractor_2.zip').exists()
    assert (tmp_dir / 'extractor_3.zip').exists() == False
    assert (tmp_dir / 'extractor_4.zip').exists()
    assert fsds._feature_registry == {
        'feature_A': 'extractor_1',
        'feature_A_1': 'extractor_4',
    }
    assert fsds.df_ui.columns.to_list() == ['feature_A', 'feature_A_1']
    fsds.extract_features()
    assert has_been_called(id='extractor_valid').times == 1
    assert fsds._feature_registry == {
        'feature_A': 'extractor_1',
        'feature_A_1': 'extractor_4',
        'feature_A_2': 'extractor_1',
        'feature_A_3': 'extractor_4',
    }
    assert fsds.df_ui.columns.to_list() == ['feature_A', 'feature_A_1',
        'feature_A_2', 'feature_A_3']