from instacartlib.DataFrameFileCache import DataFrameFileCache

import pandas as pd

import pytest


@pytest.fixture
def df():
    return pd.DataFrame([1, 2, 3], dtype='float32')


@pytest.fixture
def fname():
    return 'df.zip'


@pytest.fixture
def call_counter():
    def this():
        this.count += 1
    this.count = 0
    return this


def test_cache_disable_default(tmp_dir, call_counter, df):
    subdir = tmp_dir / 'sub1' / 'sub2'
    assert not subdir.exists()

    cache_file_path = subdir / 'df_cached.zip'

    @DataFrameFileCache(cache_file_path)
    def get_df():
        call_counter()
        return df

    assert call_counter.count == 0

    output_1 = get_df()
    assert call_counter.count == 1
    assert cache_file_path.exists()
    pd.testing.assert_frame_equal(output_1, df)

    output_2 = get_df()
    assert call_counter.count == 1
    pd.testing.assert_frame_equal(output_2, df)

    cache_content = pd.read_pickle(cache_file_path)
    pd.testing.assert_frame_equal(cache_content, df)



def test_cache_disable_true(tmp_dir, call_counter, df):
    cache_file_path = tmp_dir / 'df_cached.zip'
    assert not cache_file_path.exists()

    @DataFrameFileCache(cache_file_path, disable=True)
    def get_df():
        call_counter()
        return df

    assert call_counter.count == 0

    output_1 = get_df()
    assert call_counter.count == 1
    assert not cache_file_path.exists()
    pd.testing.assert_frame_equal(output_1, df)

    output_2 = get_df()
    assert call_counter.count == 2
    pd.testing.assert_frame_equal(output_2, df)


def test_cache_non_existed_subdir(tmp_dir, call_counter, df):
    subdir = tmp_dir / 'sub1' / 'sub2'
    assert not subdir.exists()

    cache_file_path = subdir / 'df_cached.zip'

    @DataFrameFileCache(cache_file_path)
    def get_df():
        call_counter()
        return df

    get_df()