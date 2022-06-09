
from instacartlib.Transactions import read_transactions_csv
from instacartlib.Transactions import _preprocess_raw_columns
from instacartlib.Transactions import check_df_raw
from instacartlib.Transactions import get_transactions_csv_path
from instacartlib.Transactions import _new_iord
from instacartlib.Transactions import _update_iord
from instacartlib.Transactions import get_last_orders
from instacartlib.Transactions import InvalidTransactionsData

import io
import numpy as np
import pandas as pd

import pytest


@pytest.fixture
def required_raw_columns_dtypes():
    return {
        'order_id': np.dtype('uint32'),
        'user_id': np.dtype('uint32'),
        'order_number': np.dtype('uint8'),
        # 'order_dow': np.dtype('uint8'),
        # 'order_hour_of_day': np.dtype('uint8'),
        'days_since_prior_order': np.dtype('float16'),
        'product_id': np.dtype('uint32'),
        'add_to_cart_order': np.dtype('uint8'),
        'reordered': np.dtype('uint8'),
    }


@pytest.fixture
def n_raw_cols(required_raw_columns_dtypes):
    return len(required_raw_columns_dtypes)


@pytest.fixture
def required_columns_dtypes():
    return {
        'oid': np.dtype('uint32'),
        'uid': np.dtype('uint32'),
        'iord': np.dtype('uint8'),
        'iid': np.dtype('uint32'),
        'reord': np.dtype('uint8'),
        # 'dow': np.dtype('uint8'),
        # 'hour': np.dtype('uint8'),
        'days_prev': np.dtype('int8'),
        'cart_pos': np.dtype('uint8'),
    }


@pytest.fixture
def n_cols(required_columns_dtypes):
    return len(required_columns_dtypes)


TRANSACTIONS_CSV = """\
order_id,user_id,order_number,order_dow,order_hour_of_day,days_since_prior_order,product_id,add_to_cart_order,reordered
2539329,1,1,2,8,,196,1.0,0.0
2539329,1,1,2,8,,14084,2.0,0.0
2539329,1,1,2,8,,12427,3.0,0.0
"""

TRANSACTIONS_CSV_REQUIRED_COLUMN_MISSING = """\
order_id,order_number,order_dow,order_hour_of_day,days_since_prior_order,product_id,add_to_cart_order,reordered
2539329,1,2,8,,196,1.0,0.0
2539329,1,2,8,,14084,2.0,0.0
2539329,1,2,8,,12427,3.0,0.0
"""

def test_read_transactions_csv(transactions_csv_path,
        required_raw_columns_dtypes, n_trns, n_raw_cols):
    output_1 = read_transactions_csv(transactions_csv_path)
    assert type(output_1) == pd.DataFrame
    assert output_1.shape == (n_trns, n_raw_cols)
    assert output_1.dtypes.to_dict() == required_raw_columns_dtypes

    output_2 = read_transactions_csv(transactions_csv_path, nrows=0)
    assert type(output_2) == pd.DataFrame
    assert output_2.shape == (0, n_raw_cols)

    output_3 = read_transactions_csv(io.StringIO(TRANSACTIONS_CSV))
    assert type(output_3) == pd.DataFrame
    assert output_3.shape == (3, n_raw_cols)

    with pytest.raises(ValueError):
        read_transactions_csv(transactions_csv_path, nrows=-1)

    with pytest.raises(FileNotFoundError):
        read_transactions_csv('__NON_EXISTENT_FILE__')

    with pytest.raises(InvalidTransactionsData, match='user_id'):
        read_transactions_csv(io.StringIO(
            TRANSACTIONS_CSV_REQUIRED_COLUMN_MISSING))


def test_check_df_raw(df_trns_raw):
    check_df_raw(df_trns_raw)


def test_df_trns_no_nans(df_trns):
    assert df_trns.notna().all(axis=None)


def test_df_trns_col_types(df_trns, required_columns_dtypes):
    assert df_trns.dtypes.to_dict() == required_columns_dtypes


def test_preprocess_raw_columns(df_trns_raw, n_trns, n_cols):
    output = _preprocess_raw_columns(df_trns_raw)
    assert type(output) == pd.DataFrame
    assert output.shape == (n_trns, n_cols)


def test_get_transactions_csv_path(test_data_dir):
    path = get_transactions_csv_path(test_data_dir)
    assert path == (test_data_dir / 'transactions.csv')

    with pytest.raises(FileNotFoundError):
        get_transactions_csv_path('__NON-EXISTENT_PATH__')


def test_new_iord():
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

    test_output = _new_iord(test_input)

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

    test_output = _update_iord(test_input, start_count=1)

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


@pytest.mark.parametrize('n_last_orders,n_trns', [
    (0, 0),
    (1, 90),
    (2, 164),
])
def test_get_last_orders(df_trns, n_last_orders, n_trns):
    df = get_last_orders(df_trns, n_last_orders)
    assert len(df) == n_trns
