import numpy as np
import pandas as pd


COLUMN_NAMES_DICT = {
    'order_id'               : 'order_id',
    'user_id'                : 'uid',
    'order_number'           : 'order_n',
    'product_id'             : 'iid',
    'reordered'              : 'is_reordered',
    'order_dow'              : 'order_dow',
    'order_hour_of_day'      : 'order_hour_of_day',
    'days_since_prior_order' : 'days_since_prior_order',
    'add_to_cart_order'      : 'cart_pos',
}

COLUMN_DTYPES_DICT = {
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

REQUIRED_COLUMNS = [
    'order_id',
    'uid',
    'order_n',
    'iid',
    'is_reordered',
    'order_dow',
    'order_hour_of_day',
    'days_since_prior_order',
    'cart_pos',
]


def get_df_trns_from_raw(df_raw, exclude_columns=None):
    """
    Create transactions dataframe from raw transactions.

    If `days_since_prior_order` column is required, its dtype is converted to
    integer with NaN values filled with `-1`.

    df_raw: DataFrame
        Required columns (9): order_id, user_id, order_number, order_dow,
            order_hour_of_day, days_since_prior_order, product_id,
            add_to_cart_order, reordered

    exclude_columns: list-like, optional
        Column names to be exluded from output dataframe.

    Return:
    -------
    df: DataFrame
        Available columns (9): order_id, uid, order_n, iid, is_reordered,
            order_dow, order_hour_of_day, days_since_prior_order, cart_pos
    """
    if type(exclude_columns) == str:
        raise ValueError('`exclude_columns` expected to be list of str, '
            'not str')

    required_columns = REQUIRED_COLUMNS
    if exclude_columns:
        required_columns = [name for name in required_columns
            if name not in exclude_columns]

    df = df_raw.rename(columns=COLUMN_NAMES_DICT)
    try:
        df = df.loc[:, required_columns]
    except KeyError as e:
        e.args = ('Missing required columns in raw transactions dataframe',
            *e.args)
        raise

    if 'days_since_prior_order' in df:
        df = df.fillna({'days_since_prior_order': -1})

    column_dtypes = {col: COLUMN_DTYPES_DICT[col]
                     for col in df
                     if col in COLUMN_DTYPES_DICT}
    df = df.astype(column_dtypes)
    return df


def split_last_order(df_trns):
    """
    Split target basket (last order) transactions and the rest into two
    dataframes.

    df_trns: DataFrame
        Requirements:
            1. Required columns (2): uid, order_id.
            2. User's transactions are sorted in temporal order.

    Returns:
    --------
    (df_trns_past, df_trns_target)
    df_trns_past: DataFrame
        Transactions not in the target basket (last order).
    df_trns_target: DataFrame
        Transactions in the target basket (last order).
    """
    last_order_ids = df_trns.drop_duplicates('uid', keep='last').order_id
    is_last_order = df_trns.order_id.isin(last_order_ids)
    return (df_trns[~is_last_order], df_trns[is_last_order])



def _days_since_prior_order_reverse_cumsum(df_trns):
    days_to_prior = (
        df_trns
            .drop_duplicates('order_id')
            .set_index(['order_id', 'uid'])
            .days_since_prior_order
    )
    days_until_next = (
        days_to_prior
            .groupby('uid', sort=False)
            .shift(-1, fill_value=0)
    )
    days_until_last = (
        days_until_next[::-1]
            .groupby('uid', sort=False)
            .cumsum()[::-1]
            .astype('uint16')
            .rename('days_until_last')
    )
    return days_until_last



def get_order_days_until_last(df_trns):
    """
    For each user's order count days until the last order. Value for the last
    order is 0.

    df_trns: DataFrames
        Requirements:
            1. Required columns (3): order_id, uid, days_since_prior_order
            2. User's orders (baskets) are sorted in temporal order.

    Returns
    -------
    days_until_last: pd.Series
        Index: order_id
        Value: uint16
    """
    return _days_since_prior_order_reverse_cumsum(df_trns).droplevel('uid')


def get_order_days_until_target(df_trns):
    """
    For each user's order predict days until the target order (next after the
    most recent one).

    df_trns: DataFrames
        Requirements:
            1. Required columns (3): order_id, uid, days_since_prior_order
            2. User's orders (baskets) are sorted in temporal order.

    Returns
    -------
    days_until_last: pd.Series
        Index: order_id
        Value: uint16
    """
    order_uid_days = _days_since_prior_order_reverse_cumsum(df_trns)
    uid_days_median = get_user_days_between_orders_mid(df_trns)
    order_uid_days += uid_days_median
    return order_uid_days.droplevel('uid')


def get_days_until_same_item(df_trns):
    """
    For each transaction made by user, calculate the number of days the user
    purchaised the same item again or until the last order if the item has not
    been purchaised afterwards.

    df_trns: DataFrames
        Requirements:
            1. Required columns (3): uid, iid, days_until_last
            2. User's transactions are sorted in temporal order.

    Returns
    -------
    days_until_same_item: pd.Series
        Index: order_id
        Value: uint16
    """
    same_item_next_purchase = (
        df_trns
            .groupby(['uid', 'iid'], sort=False)
            .days_until_last.shift(-1, fill_value=0)
    )
    days_until_same_item = (df_trns.days_until_last - same_item_next_purchase
        ).rename('days_until_same_item')
    return days_until_same_item


def get_user_days_between_orders_mid(df_trns, n_most_recent=15):
    """
    Calculate median days between orders for each user.

    Used to predict when the next order after the last one will be made.

    df_trns: DataFrame
        Requirements:
            1. Required columns (3): order_id, uid, days_since_prior_order

    Returns
    -------
    days_between_orders_mid: pd.Series
        Index: uid
        Value: float16
    """
    df_orders = df_trns.drop_duplicates('order_id')
    df_orders_except_initial = df_orders[df_orders.days_since_prior_order != -1]
    return (
        df_orders_except_initial
            .groupby('uid')
            .days_since_prior_order
            .median()
            .astype('float16')
            .rename('days_between_orders_mid')
    )


def get_n_last_orders(df_trns, n):
    """
    Limit transactions to n most recent orders.

    df_trns: DataFrame
        Requirements:
            1. Required columns (3): order_id, uid
            2. User's orders (baskets) are sorted in temporal order.
    n: None or int, required
        Get no more then n most recent orders. If n is None orders will not be
        limited (same dataframe is returned).

    Returns
    -------
    df_trns: DateFrame
        Subset of input dataframe.
    """
    if n is None:
        return df_trns
    last_order_ids = (
        df_trns
            .drop_duplicates('order_id')
            .groupby('uid')
            .tail(n)
            .order_id
    )
    return df_trns[df_trns.order_id.isin(last_order_ids)]
