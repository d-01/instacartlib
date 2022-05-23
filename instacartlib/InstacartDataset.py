"""
InstacartDataset API.

Capabilities:
* Load raw transactions data from `*.csv` or `*.zip` files into DataFrame
* Limit number of orders to N most recent (per user)

Capabilities (planned):
* Generate train/test datasets for classification model
"""

import numpy as np
import pandas as pd

from .transactions import get_transactions_csv_path
from .transactions import download_transactions_csv
from .transactions import read_transactions_csv
from .transactions import preprocess_raw_columns

from .utils import dummy_contextmanager
from .utils import timer_contextmanager


REDUCED_DATASET_N_ROWS = 1_571_044  # 6000 user ids


# move to products.py
class PRODUCTS_CSV:
    NAME = "products.csv.zip"      # 896KB
    MD5 = "35b4246bfeaec5fb6217ffc85975a6db"
    GDRIVE_ID = "130QJGLbv265WYgRtjXHm4FFY7QCMRXtv"


class InstacartDataset:
    """
    generate_inner_ids: {False, True}
        Replace values in user_id and product_id columns with integer ids. If 
        True: `uid_raw_to_inner, iid_raw_to_inner, uid_inner_to_raw,
        iid_inner_to_raw` attributes will be populated after calling 
        `from_dir()` method.
    show_progress: {False, True}
        Print messages with progress information.

    Transactions table:
    - oid - order id (raw)
    - uid - user id (inner, 0-based)
    - iord - order (basket) index in reverse temporal order
      * Example: [4, 3, 2, 1, 0], means 0 is the newest, 4 is the oldest.
      * Conveniet for taking target basket: `df[df.iord==0]` or limiting history
        to N most recent orders, like: `df[df.iord > N]`.
    - iid - item id (inner, 0-based)
    - reord - was item ordered previously? 1=yes, 0=no
    - dow - day of week (0-6)
    - hour - hour of day (0-23)
    - days_prev - number of days passed since previous order (-1 = unknown
      value for the initial order)
    - in_cart_ord - add to cart order (1-based)
    """
    def __init__(self, show_progress=False):
        self.show_progress = show_progress

        self.df_trns = None


    def _info(self, message):
        if self.show_progress:
            print(message)


    def _timer(self, message):
        if self.show_progress:
            return timer_contextmanager(message)
        else:
            return dummy_contextmanager()


    def from_dir(self, path_dir='.', reduced=False):
        """
        Read files `transactions.csv` and `products.csv` with raw data from
        current path or given local directory into DataFrame.

        path_dir: str or pathlib.Path
            Path to directory with `transactions.csv` or `transactions.csv.zip` files.
        reduced: {False, True}
            Read transactions for the first 6000 users.
        """
        transaction_csv_path = get_transactions_csv_path(path_dir)
        n_rows = REDUCED_DATASET_N_ROWS if reduced else None
        with self._timer('Reading `transactions.csv` ...'):
            df_raw = read_transactions_csv(transaction_csv_path, n_rows)
        with self._timer('Preprocessing columns ...'):
            self.df_trns = preprocess_raw_columns(df_raw)
        return self


    def load_from_gdrive(self, path_dir='.'):
        """
        Download files `transactions.csv` and `products.csv` from gdrive
        using url or id to current path or given directory.
        """
        download_transactions_csv(path_dir, self.show_progress)


    def limit_train_orders(self, n_most_recent=None):
        """
        For each user drop orders except n most recent ones.

        n_most_recent: None or int
            Number of the most recent orders in user's history to keep. If None
            keep all orders (no-op).
        """
        if n_most_recent is not None:
            self.df_trns = self.df_trns[self.df_trns.iord < n_most_recent]
        return self


    def __repr__(self):
        class_name = self.__class__.__name__
        df_train_shape = None if self.df_trns is None else self.df_trns.shape
        return f'<{class_name} transactions_data={df_train_shape}>'


