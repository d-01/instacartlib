"""
Set of functions to read raw data from `transactions.csv` and perform basic preprocessing:
1. Choose appropriate dtypes for columns.
2. Deal with NaNs.
3. Convert user ids and product ids from raw arbitrary type to inner (int, 0-based) type.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import download_from_info

TRANSACTIONS_FILENAME = 'transactions.csv'
TRANSACTIONS_ZIP_FILENAME = 'transactions.csv.zip'

class TRANSACTIONS_DOWNLOAD_INFO:
    NAME = "transactions.csv.zip"  # 154.5MB
    MD5 = "835dc8854d0c8f9d72f90e0f13ae6a2e"
    GDRIVE_ID = "1-2cq6ZrBd57o_m6ixdWYhUbRHL3z43N1"


def get_transactions_csv_path(path_dir):
        for filename in [TRANSACTIONS_FILENAME, TRANSACTIONS_ZIP_FILENAME]:
            path = Path(path_dir) / filename
            if path.exists():
                return path

        abs_path = Path(path_dir).absolute()
        raise FileNotFoundError(
            f'Neither "{TRANSACTIONS_FILENAME}", '
            f'nor "{TRANSACTIONS_ZIP_FILENAME}.zip" '
            f'was not found at "{abs_path}".')


def download_transactions_csv(path_dir, show_progress=False):
    download_from_info(TRANSACTIONS_DOWNLOAD_INFO, path_dir, show_progress)


def read_transactions_csv(csv_path, nrows=None):
    """
    Read `transactions.csv` file into DataFrame.

    csv_path: str or pathlib.Path
        Path to `transactions.csv` file.
    nrows: None or int
        Limit the number of rows to read from the file (header row not included).
    """
    dtype = {
        "order_id": np.uint32,
        "user_id": np.uint32,
        "order_number": np.uint8,  # 99
        "order_dow": np.uint8,
        "order_hour_of_day": np.uint8,
        "days_since_prior_order": np.float16,  # 30.0, NaN
        "product_id": np.uint32,
        "add_to_cart_order": np.uint8,  # 95
        "reordered": np.uint8,  # 0, 1
    }
    df_raw = pd.read_csv(csv_path, dtype=dtype, nrows=nrows)
    return df_raw


def check_df_raw(df_raw):
    """
    Check assumptions are valid:
    1. `order_number` per user is in arithmetic progression: 1, 2, 3, 4, etc.
    2. `add_to_cart_order` per order has no duplicates and is monotonically
       increasing.
    3. Rows per user is ordered by [order_number, add_to_cart_order].
    """
    order_number_grouped_by_user = df_raw.groupby('user_id').order_number
    if order_number_grouped_by_user.is_monotonic_increasing.all() == False:
        raise ValueError('`order_number` expected to be sorted in icreasing '
                         'order per user_id.')

    min_order_number_per_user = order_number_grouped_by_user.min()
    if (min_order_number_per_user == 1).all() == False:
        raise ValueError(
            'Lowest `order_number` is expected to be 1 per user_id.')

    max_order_number_per_user = order_number_grouped_by_user.max()
    num_orders_per_user = (df_raw
        .drop_duplicates(['user_id', 'order_number'])
        .value_counts('user_id', sort=False)
    )
    if (max_order_number_per_user == num_orders_per_user).all() == False:
        raise ValueError('Max `order_number` is expected to be equal to the '
                         'number of user\'s orders.')

    if df_raw.duplicated(['order_id', 'add_to_cart_order']).any():
        raise ValueError('`add_to_cart_order` should be unique within order.')

    cart_order_grouped_by_order = df_raw.groupby('order_id').add_to_cart_order
    if cart_order_grouped_by_order.is_monotonic_increasing.all() == False:
        raise ValueError('`add_to_cart_order` expected to be sorted in '
                         'increasing order per order_id.')


def preprocess_raw_columns(df_raw):
    """
    df_raw: DataFrame
        Columns (9): order_id, user_id, order_number, order_dow,
            order_hour_of_day, days_since_prior_order, product_id,
            add_to_cart_order, reordered

    Return:
    -------
    df: DataFrame
        Columns (9): oid, uid, iord, iid, reord, dow, hour, days_prev,
            in_cart_ord
    """
    iord = (df_raw.drop_duplicates('order_id')
                .set_index('order_id')
                .groupby('user_id')
                .cumcount(ascending=False)
                .astype(np.uint8)
                .rename('iord'))

    df = pd.DataFrame({
        'oid'         : df_raw.order_id,
        'uid'         : df_raw.user_id,
        'iord'        : df_raw.order_id.map(iord),
        'iid'         : df_raw.product_id,
        'reord'       : df_raw.reordered,
        'dow'         : df_raw.order_dow,
        'hour'        : df_raw.order_hour_of_day,
        'days_prev'   : df_raw.days_since_prior_order.fillna(-1).astype('int8'),
        'in_cart_ord' : df_raw.add_to_cart_order,
    })
    return df


# Probable improvement: verify `iord` column has right numbering format.
def drop_orders(df_trns, keep_n=None):
    """
    For each user drop orders except n most recent ones.

    df_trns: DataFrame
        DataFrame with users' transactions, where orders are numbered
        `0, 1, 2, ...` from the most recent to the oldest.
        `iord=0` is the last (most recent) order.
    n_most_recent: None or int
        Number of the most recent orders in user's history to keep. If None
        keep all orders (no-op).
    """
    if keep_n is not None:
        df_trns = df_trns[df_trns.iord < keep_n]
    return df_trns