from instacartlib.transactions_utils import get_df_trns_from_raw
from instacartlib.transactions_utils import split_last_order
from instacartlib.transactions_utils import get_order_days_until_last
from instacartlib.transactions_utils import get_order_days_until_target
from instacartlib.transactions_utils import get_days_until_same_item
from instacartlib.transactions_utils import get_user_days_between_orders_mid
from instacartlib.transactions_utils import get_n_last_orders

import io

import numpy as np
import pandas as pd

import pytest


TRANSACTIONS_CSV = """\
order_id,user_id,order_number,order_dow,order_hour_of_day,days_since_prior_order,product_id,add_to_cart_order,reordered
2539329,1,1,2,8,,196,1.0,0.0
2539329,1,1,2,8,0,14084,2.0,0.0
"""

TRANSACTIONS_CSV_EXTRA_COLUMNS = """\
order_id,user_id,order_number,order_dow,order_hour_of_day,days_since_prior_order,product_id,add_to_cart_order,reordered,extra_col
2539329,1,1,2,8,,196,1.0,0.0,0
2539329,1,1,2,8,0,14084,2.0,0.0,0
"""

TRANSACTIONS_CSV_MISSING_COLUMNS = """\
order_id,user_id,order_number,order_dow,order_hour_of_day,days_since_prior_order,product_id,add_to_cart_order
2539329,1,1,2,8,,196,1.0
2539329,1,1,2,8,0,14084,2.0
"""

DF_TRNS = '''\
order_id  uid  order_n  iid  is_reordered  order_dow  order_hour_of_day  days_since_prior_order  cart_pos  days_until_target  days_until_same_item
ord_2  user_A  1  item_B  0  0  0  -1  1  9  6
ord_2  user_A  1  item_A  0  0  0  -1  2  9  4
ord_3  user_A  2  item_A  1  0  0   4  1  5  2
ord_4  user_A  3  item_A  1  0  0   2  1  3  3
ord_4  user_A  3  item_B  1  0  0   2  2  3  3
ord_5  user_A  4  item_A  1  0  0   3  1  0  0
ord_8  user_B  1  item_B  0  0  0  -1  1  4  4
ord_8  user_B  1  item_C  0  0  0  -1  2  4  3
ord_9  user_B  2  item_C  1  0  0   3  1  1  1
ord_0  user_B  3  item_C  1  0  0   1  1  0  0
'''

DF_ORD = '''\
order_id  uid  order_n  order_dow  order_hour_of_day  days_since_prior_order  days_until_target
ord_2  user_A  1  0  0  -1  9
ord_3  user_A  2  0  0   4  5
ord_4  user_A  3  0  0   2  3
ord_5  user_A  4  0  0   3  0
ord_8  user_B  1  0  0  -1  4
ord_9  user_B  2  0  0   3  1
ord_0  user_B  3  0  0   1  0
'''


@pytest.fixture
def df_trns_all_columns():
    return pd.read_csv(io.StringIO(DF_TRNS), sep=r'\s+')


@pytest.fixture
def df_trns(df_trns_all_columns):
    return df_trns_all_columns[['order_id', 'uid', 'order_n', 'iid',
        'is_reordered', 'order_dow', 'order_hour_of_day',
        'days_since_prior_order', 'cart_pos']]


@pytest.fixture
def df_ord_all_columns():
    return pd.read_csv(io.StringIO(DF_ORD), sep=r'\s+')


@pytest.fixture
def df_ord(df_ord_all_columns):
    return df_ord_all_columns[['order_id', 'uid', 'order_n', 'order_dow',
        'order_hour_of_day', 'days_since_prior_order']]


@pytest.fixture
def df_raw():
    return pd.read_csv(io.StringIO(TRANSACTIONS_CSV))


@pytest.fixture
def df_raw_extra_columns():
    return pd.read_csv(io.StringIO(TRANSACTIONS_CSV_EXTRA_COLUMNS))


@pytest.fixture
def df_raw_missing_columns():
    return pd.read_csv(io.StringIO(TRANSACTIONS_CSV_MISSING_COLUMNS))


@pytest.fixture
def df_trns_required_dtypes():
    return {
        'order_id'               : np.uint32,
        'uid'                    : np.uint32,
        'order_n'                : np.uint8,   # 99
        'iid'                    : np.uint32,
        'is_reordered'           : np.uint8,   # 0, 1
        'order_dow'              : np.uint8,   # 0..6
        'order_hour_of_day'      : np.uint8,   # 0..23
        'days_since_prior_order' : np.int8,    # -1..30
        'cart_pos'               : np.uint8,   # 95
    }


def test_get_df_trns_from_raw(df_raw, df_trns_required_dtypes):
    df_trns = get_df_trns_from_raw(df_raw)

    assert type(df_trns) == pd.DataFrame
    assert df_trns.columns.to_list() == ['order_id', 'uid', 'order_n', 'iid',
        'is_reordered', 'order_dow', 'order_hour_of_day',
        'days_since_prior_order', 'cart_pos']
    assert df_trns.dtypes.to_dict() == df_trns_required_dtypes
    assert df_trns.days_since_prior_order.isna().any() == False


def test_get_df_trns_from_raw_exclude_columns(df_raw):
    df_trns = get_df_trns_from_raw(df_raw, exclude_columns=['is_reordered',
        'order_n'])

    assert 'is_reordered' not in df_trns
    assert 'order_n' not in df_trns


def test_get_df_trns_from_raw_exclude_columns_str_param(df_raw):
    with pytest.raises(ValueError):
        get_df_trns_from_raw(df_raw, exclude_columns='is_reordered')


def test_get_df_trns_from_raw_extra_columns(df_raw_extra_columns):
    df_trns = get_df_trns_from_raw(df_raw_extra_columns)
    assert 'extra_col' not in df_trns


def test_get_df_trns_from_raw_missing_columns(df_raw_missing_columns):
    with pytest.raises(KeyError, match='is_reordered'):
        get_df_trns_from_raw(df_raw_missing_columns)


def test_get_df_trns_from_raw_missing_columns_excluded(df_raw_missing_columns):
    """ No exception if the missign column is not required. """
    get_df_trns_from_raw(df_raw_missing_columns,
        exclude_columns=['is_reordered'])


def test_split_last_order():
    df_trns = pd.read_fwf(io.StringIO('''\
        #  order_id     uid  row_id
        0     ord_2  user_A       0
        1     ord_3  user_A       1
        2     ord_3  user_A       2
        3     ord_4  user_A       3
        4     ord_5  user_A       4
        5     ord_5  user_A       5
        6     ord_8  user_B       6
        7     ord_9  user_B       7
        8     ord_9  user_B       8
        9     ord_0  user_B       9
        ''')).set_index('#')

    output = split_last_order(df_trns)

    assert type(output) == tuple
    assert len(output) == 2
    assert output[0].row_id.to_list() == [0, 1, 2, 3, 6, 7, 8]
    assert output[1].row_id.to_list() == [4, 5, 9]


def test_get_order_days_until_last(df_ord):
    # days_since_prior_order
    # -------------------
    # order_n  1  2  3  4
    # uid
    # user_A  -1  4  2  3
    # user_B  -1  3  1

    # days_until_next
    # -------------------
    # order_n  1  2  3  4
    # uid
    # user_A   4  2  3 -1
    # user_B   3  1 -1

    # days_until_last
    # -------------------
    # order_n  1  2  3  4
    # uid
    # user_A   9  5  3  0
    # user_B   4  1  0
    output = get_order_days_until_last(df_ord)

    assert type(output) == pd.Series

    index = pd.Index(['ord_2', 'ord_3', 'ord_4', 'ord_5', 'ord_8', 'ord_9',
         'ord_0'], name='order_id')
    expected_output = pd.Series([
        9, 5, 3, 0,  # user_A's orders
        4, 1, 0,     # user_B's orders
    ], index=index, name="days_until_last")
    pd.testing.assert_series_equal(output, expected_output, check_dtype=False)


def test_get_order_days_until_target(df_ord):
    output = get_order_days_until_target(df_ord)

    assert type(output) == pd.Series

    index = pd.Index(['ord_2', 'ord_3', 'ord_4', 'ord_5', 'ord_8', 'ord_9',
         'ord_0'], name='order_id')
    expected_output = pd.Series([
        12, 8, 6, 3,  # user_A's orders
         6, 3, 2,     # user_B's orders
    ], index=index, name="days_until_target")
    pd.testing.assert_series_equal(output, expected_output, check_dtype=False)



def test_get_days_until_same_item(df_trns_all_columns):
    # days_since_prior_order
    # -------------------------
    # order_n        1  2  3  4
    # uid    iid
    # user_A item_A -1  4  2  3
    #        item_B -1  .  2  .
    # user_B item_B -1  .  .  .
    #        item_C -1  3  1  .

    # days_until_same_item
    # -------------------------
    # order_n        1  2  3  4
    # uid    iid
    # user_A item_A  4  2  3  0
    #        item_B  6  .  3  .
    # user_B item_B  4  .  .  .
    #        item_C  3  1  0  .
    df_trns = df_trns_all_columns.drop(columns='days_until_same_item')
    output = get_days_until_same_item(df_trns)

    assert type(output) == pd.Series
    pd.testing.assert_series_equal(
        output,
        df_trns_all_columns.days_until_same_item,
        check_dtype=True,
    )


def test_get_user_days_between_orders_mid(df_ord):
    output = get_user_days_between_orders_mid(df_ord)
    expected = pd.Series({
        'user_A': 3,  # median([4, 2, 3])
        'user_B': 2,  # median([3, 1])
    }, dtype='float16', name='days_between_orders_mid')
    expected.index.name = 'uid'

    pd.testing.assert_series_equal(output, expected)


def test_get_n_last_orders_n_default(df_trns):
    with pytest.raises(TypeError):
        get_n_last_orders(df_trns)


def test_get_n_last_orders_n_15():
    df_trns = pd.DataFrame({
        'order_id': range(30),
        'uid': 'user_A',
    })
    output = get_n_last_orders(df_trns, n=15)
    expected = pd.DataFrame({
        'order_id': range(15, 30),
        'uid': 'user_A',
    })
    pd.testing.assert_frame_equal(output, expected)


def test_get_n_last_orders_n_0():
    df_trns = pd.DataFrame({
        'order_id': range(30),
        'uid': 'user_A',
    })
    output = get_n_last_orders(df_trns, n=0)
    expected = pd.DataFrame({
        'order_id': [],
        'uid': [],
    })
    pd.testing.assert_frame_equal(output, expected, check_dtype=False)



