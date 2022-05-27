from instacartlib.DataFrameFileCache import DataFrameFileCache

import datetime
import random
import shutil

import pandas as pd

import pytest

CLEANUP_TEMP_DIR = False


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


@pytest.fixture
def tmp_dir(tmp_path):
    datetime_8_6_6 = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    random_8 = random.randint(0, 0xffff_ffff)
    path = tmp_path / f'{datetime_8_6_6}_{random_8:X}'
    path.mkdir()
    yield path
    if CLEANUP_TEMP_DIR:
        shutil.rmtree(path)


def test_cache_disable_default(tmp_dir, call_counter, df):
    cache_file_path = tmp_dir / 'df_cached.zip'
    assert not cache_file_path.exists()

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


