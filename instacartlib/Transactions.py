"""
Transactions API.

Capabilities:
* Load raw transactions data from `*.csv` or `*.zip` files into DataFrame with
  appropriate dtype.
* Deal with NaNs.
* Limit number of orders to N most recent (per user).
"""

from .utils import download_from_info
from .utils import dummy_contextmanager
from .utils import timer_contextmanager
from .utils import get_df_info

from pathlib import Path

import numpy as np
import pandas as pd


TRANSACTIONS_FILENAME = 'transactions.csv'
TRANSACTIONS_ZIP_FILENAME = 'transactions.csv.zip'
REDUCED_DATASET_N_ROWS = 1_571_044  # 6000 user ids
EXCLUDE_COLUMNS = []
RAW_COLUMNS_DTYPES = {
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


class TRANSACTIONS_DOWNLOAD_INFO:
    NAME = "transactions.csv.zip"  # 154.5MB
    MD5 = "835dc8854d0c8f9d72f90e0f13ae6a2e"
    GDRIVE_ID = "1-2cq6ZrBd57o_m6ixdWYhUbRHL3z43N1"


class InvalidTransactionsData(ValueError):
    """ Transactions table missing required columns. """


def get_transactions_csv_path(path_dir):
        for filename in [TRANSACTIONS_FILENAME, TRANSACTIONS_ZIP_FILENAME]:
            path = Path(path_dir) / filename
            if path.exists():
                return path

        abs_path = Path(path_dir).absolute()
        raise FileNotFoundError(
            f'Neither "{TRANSACTIONS_FILENAME}", '
            f'nor "{TRANSACTIONS_ZIP_FILENAME}" '
            f'was not found at "{abs_path}".')


def read_transactions_csv(filepath_or_buffer, nrows=None, exclude_columns=None):
    """
    Read `transactions.csv` file into DataFrame.

    Assumptions about `transactions` table:
    1. `order_number` per user is in arithmetic progression: 1, 2, 3, 4, etc.
    2. `add_to_cart_order` per order has no duplicates and is monotonically
       increasing.
    3. Rows per user is ordered by [order_number, add_to_cart_order].

    Parameters
    ----------
    filepath_or_buffer: str, pathlib.Path or buffer
        Path to csv file (`*.csv`), zipped csv file (`*.zip`) or buffer with
        csv-formatted string (io.StringIO).
    nrows: None or int
        Limit the number of rows to read from the file (header row not
        included).
    """
    if exclude_columns is None:
        exclude_columns = []

    required_columns_dtypes = {
        col: dtype
        for col, dtype
        in RAW_COLUMNS_DTYPES.items()
        if col not in exclude_columns
    }

    try:
        df_raw = pd.read_csv(filepath_or_buffer,
                             usecols=required_columns_dtypes,
                             dtype=required_columns_dtypes,
                             nrows=nrows)
    except ValueError as e:
        raise InvalidTransactionsData(
            f'Transactions file "{filepath_or_buffer}"', e.args[0])
    return df_raw


class Transactions:
    """
    Transactions data manipulator.

    Transactions dataframe:
    - oid - order id (raw)
    - uid - user id
    - iord - order (basket) index in reverse temporal order
      * Example: [4, 3, 2, 1, 0], means 0 is the newest, 4 is the oldest.
      * Conveniet for taking target basket: `df[df.iord==0]` or limiting history
        to N most recent orders, like: `df[df.iord > N]`.
    - iid - item id (raw product id)
    - reord - was item ordered previously? 1=yes, 0=no
    - dow - day of week (0-6) [currently not used]
    - hour - hour of day (0-23) [currently not used]
    - days_prev - number of days passed since previous order (-1 = for the
      initial order)
    - cart_pos - add to cart order (1-based)

    show_progress: {False, True}
        Print messages with progress information for long operations.
    """
    def __init__(self, iord_start_count=0, show_progress=False):
        self.iord_start_count = iord_start_count
        self.show_progress = show_progress

        self.df = None


    def __repr__(self):
        class_name = self.__class__.__name__
        df_info = 'None' if self.df is None else get_df_info(self.df)
        return f'<{class_name} df={df_info}>'


    def _timer(self, message):
        if self.show_progress:
            return timer_contextmanager(message)
        else:
            return dummy_contextmanager()


    def read_dir(self, path_dir='.', reduced=False):
        """
        Read files with raw data from given local directory into DataFrame.

        path_dir: str or pathlib.Path
            Path to directory with `transactions.csv` or `transactions.csv.zip`
            files.
        reduced: {False, True}
            Read transactions for the first 6000 users.
        """
        transactions_csv_path = get_transactions_csv_path(path_dir)
        n_rows = REDUCED_DATASET_N_ROWS if reduced else None
        with self._timer(f'Reading "{transactions_csv_path.name}" ...'):
            self.df = read_transactions_csv(transactions_csv_path, n_rows)
        return self


    def load_from_gdrive(self, path_dir='.'):
        """
        Download files `transactions.csv` and `products.csv` from gdrive
        using url or id to current path or given directory.
        """
        download_from_info(TRANSACTIONS_DOWNLOAD_INFO, path_dir,
            self.show_progress)
        return self
