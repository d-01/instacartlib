
from instacartlib.Transactions import read_transactions_csv
from instacartlib.Transactions import get_transactions_csv_path
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
        'order_dow': np.dtype('uint8'),
        'order_hour_of_day': np.dtype('uint8'),
        'days_since_prior_order': np.dtype('float16'),
        'product_id': np.dtype('uint32'),
        'add_to_cart_order': np.dtype('uint8'),
        'reordered': np.dtype('uint8'),
    }


@pytest.fixture
def n_raw_cols(required_raw_columns_dtypes):
    return len(required_raw_columns_dtypes)


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


def test_df_trns_no_nans(df_trns):
    assert df_trns.notna().all(axis=None)


def test_get_transactions_csv_path(test_data_dir):
    path = get_transactions_csv_path(test_data_dir)
    assert path == (test_data_dir / 'transactions.csv')

    with pytest.raises(FileNotFoundError):
        get_transactions_csv_path('__NON-EXISTENT_PATH__')
