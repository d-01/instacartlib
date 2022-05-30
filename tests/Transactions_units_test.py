
from instacartlib.Transactions import read_transactions_csv
from instacartlib.Transactions import preprocess_raw_columns
from instacartlib.Transactions import check_df_raw
from instacartlib.Transactions import get_transactions_csv_path
from instacartlib.Transactions import get_iord
from instacartlib.Transactions import update_iord

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def expected_raw_col_types():
    return {
        'order_id': np.dtype('uint32'),
        'user_id': np.dtype('uint32'),
        'order_number': np.dtype('uint8'),
        'order_dow': np.dtype('uint8'),
        'order_hour_of_day': np.dtype('uint8'),
        'days_since_prior_order': np.dtype('float16'),
        'product_id': np.dtype('uint32'),
        'add_to_cart_order': np.dtype('uint8'),
        'reordered': np.dtype('uint8'),
    }


@pytest.fixture
def expected_col_types():
    return {
        'oid': np.dtype('uint32'),
        'uid': np.dtype('uint32'),
        'iord': np.dtype('uint8'),
        'iid': np.dtype('uint32'),
        'reord': np.dtype('uint8'),
        'dow': np.dtype('uint8'),
        'hour': np.dtype('uint8'),
        'days_prev': np.dtype('int8'),
        'in_cart_ord': np.dtype('uint8'),
    }


def test_read_transactions_csv_nrows(transactions_csv_path):
    output = read_transactions_csv(transactions_csv_path, nrows=0)
    assert type(output) == pd.DataFrame
    assert output.shape == (0, 9)

    with pytest.raises(ValueError):
        read_transactions_csv(transactions_csv_path, nrows=-1)


def test_read_transactions_csv_return_dataframe(df_trns_raw):
    assert type(df_trns_raw) == pd.DataFrame


def test_df_trns_raw_col_types(df_trns_raw, expected_raw_col_types):
    assert df_trns_raw.dtypes.to_dict() == expected_raw_col_types


def test_check_df_raw(df_trns_raw):
    check_df_raw(df_trns_raw)


def test_df_trns_no_nans(df_trns):
    assert df_trns.notna().all(axis=None)


def test_df_trns_col_types(df_trns, expected_col_types):
    assert df_trns.dtypes.to_dict() == expected_col_types


def test_preprocess_raw_columns(df_trns_raw):
    output = preprocess_raw_columns(df_trns_raw)
    assert type(output) == pd.DataFrame
    assert output.shape == (1468, 9)


def test_get_transactions_csv_path(test_data_dir):
    path = get_transactions_csv_path(test_data_dir)
    assert path == (test_data_dir / 'transactions.csv')

    with pytest.raises(FileNotFoundError):
        get_transactions_csv_path('__NON-EXISTENT_PATH__')


def test_get_iord():
    test_input = pd.DataFrame([
        ['ord_D', 'user1', 3],
        ['ord_D', 'user1', 3],
        ['ord_A', 'user1', 2],
        ['ord_A', 'user1', 2],
        ['ord_C', 'user2', 1],
        ['ord_C', 'user2', 1],
        ['ord_B', 'user2', 2],
        ['ord_B', 'user2', 2],
    ], columns=['oid', 'uid', 'iord'])

    test_output = get_iord(test_input)

    expected = pd.Series({
        'ord_D': 1,
        'ord_A': 0,
        'ord_C': 1,
        'ord_B': 0,
    })
    pd.testing.assert_series_equal(test_output, expected, check_names=False)


def test_update_iord():
    test_input = pd.DataFrame([
        ['ord_D', 'user1', 3],
        ['ord_D', 'user1', 3],
        ['ord_A', 'user1', 2],
        ['ord_A', 'user1', 2],
        ['ord_C', 'user2', 1],
        ['ord_C', 'user2', 1],
        ['ord_B', 'user2', 2],
        ['ord_B', 'user2', 2],
    ], columns=['oid', 'uid', 'iord'])

    test_output = update_iord(test_input, start_count=1)

    expected = pd.DataFrame([
        ['ord_D', 'user1', 2],
        ['ord_D', 'user1', 2],
        ['ord_A', 'user1', 1],
        ['ord_A', 'user1', 1],
        ['ord_C', 'user2', 2],
        ['ord_C', 'user2', 2],
        ['ord_B', 'user2', 1],
        ['ord_B', 'user2', 1],
    ], columns=['oid', 'uid', 'iord'])

    pd.testing.assert_frame_equal(test_output, expected)