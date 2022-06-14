"""
Prepares data for feature extraction.


Capabilities:
* Download csv files with raw data.
* Load raw data and preprocess.
* Make train dataset.
* Make test dataset.
"""

from .Transactions import Transactions
from .transactions_utils import get_df_trns_from_raw
from .transactions_utils import split_last_order
from .Products import Products
from .Products import _preprocess_raw_products
from .utils import get_df_info, format_size, get_df_size_bytes

from typing import NamedTuple

import numpy as np
import pandas as pd


COLUMNS_INCLUDE_INTO_TRNS = [
    'order_id',
    'uid',
    'iid',
    'cart_pos',
    'is_reordered',
]

COLUMNS_INCLUDE_INTO_ORDERS = [
    'order_id',
    'order_number',
    'uid',
    'order_dow',
    'order_hour_of_day',
    'days_since_prior_order',
]


def _preprocess_raw_transactions(df_raw, create_target=False, verbose=0):
    df_trns = get_df_trns_from_raw(df_raw)
    df_ord = (df_trns
        .drop_duplicates('order_id')
        .reset_index(drop=True)
        .filter(COLUMNS_INCLUDE_INTO_ORDERS))
    df_trns = df_trns.filter(COLUMNS_INCLUDE_INTO_TRNS)

    df_trns_target = df_trns[:0]
    if create_target:
        df_trns, df_trns_target = split_last_order(df_trns)

    return (df_ord, df_trns, df_trns_target)


class TestDataframes(NamedTuple):
    ord: pd.DataFrame
    trns: pd.DataFrame
    prod: pd.DataFrame


class TrainDataframes(NamedTuple):
    ord: pd.DataFrame
    trns: pd.DataFrame
    trns_target: pd.DataFrame
    prod: pd.DataFrame


class InstacartDataset:
    """
    train : {False, True}
        Set aside most recent order as target for target basket
        prediction task. These orders will be available as `df_trns_target`.
    """
    def __init__(self, train=False, verbose=0):
        self.train = train
        self.verbose = verbose

        self._transactions = Transactions(show_progress=self.verbose > 0)
        self._products = Products(show_progress=self.verbose > 0)

        self.df_ord = pd.DataFrame()
        self.df_trns = pd.DataFrame()
        self.df_trns_target = pd.DataFrame()
        self.df_prod = pd.DataFrame()

        self.n_ord_user_max = 0
        self.n_users = 0
        self.n_items = 0
        self.n_prod_items = 0
        self.n_aisles = 0
        self.n_departments = 0
        self.n_users_target = 0
        self.n_items_target = 0


    def get_dataframes(self):
        if self.train:
            return TrainDataframes(
                self.df_ord,
                self.df_trns,
                self.df_trns_target,
                self.df_prod,
            )
        else:
            return TestDataframes(
                self.df_ord,
                self.df_trns,
                self.df_prod,
            )


    @property
    def dataframes(self):
        return self.get_dataframes()


    def download(self, to_dir='instacart_temp/raw_data'):
        self._transactions.load_from_gdrive(to_dir)
        self._products.load_from_gdrive(to_dir)
        return self


    def _update_stats(self):
        self.n_ord_user_max = self.df_ord.groupby('uid').size().max()
        self.n_users = self.df_trns.uid.nunique()
        self.n_items = self.df_trns.iid.nunique()
        self.n_aisles = self.df_prod.aisle_id.nunique()
        self.n_departments = self.df_prod.department_id.nunique()
        self.n_prod_items = self.df_prod.iid.nunique()
        self.n_users_target = self.df_trns_target.uid.nunique()
        self.n_items_target = self.df_trns_target.iid.nunique()


    def _print(self, message, indent=0):
        if self.verbose > 0:
            message = str(message)
            if indent > 0:
                pad = ' ' * indent
                message = pad + message.replace('\n', '\n' + pad)
            print(message)


    def read_dir(self, path_dir='instacart_temp/raw_data', reduced=False):
        self._transactions.read_dir(path_dir=path_dir, reduced=reduced)
        self._products.read_dir(path_dir=path_dir, reduced=reduced)
        self._print('Transactions preprocessing ...', indent=2)
        self._preprocess_raw_transactions()
        self._print('Products preprocessing ...', indent=2)
        self._preprocess_raw_products()

        self._update_stats()
        return self


    def _preprocess_raw_transactions(self):
        frames = _preprocess_raw_transactions(self._transactions.df,
            create_target=self.train, verbose=self.verbose)
        (self.df_ord, self.df_trns, self.df_trns_target) = frames


    def _preprocess_raw_products(self):
        self.df_prod = _preprocess_raw_products(self._products.df,
            verbose=self.verbose)


    def info(self):
        total_memory = sum([
            get_df_size_bytes(self.df_ord),
            get_df_size_bytes(self.df_trns),
            get_df_size_bytes(self.df_trns_target),
            get_df_size_bytes(self.df_prod),
        ])

        info_message = [
            f'Raw data:                                ',
            f'    df_ord: {get_df_info(self.df_ord)} ',
            f'        max order history: {self.n_ord_user_max}',
            f'    df_trns: {get_df_info(self.df_trns)} ',
            f'        users: {self.n_users}            ',
            f'        items: {self.n_items}            ',
            f'    df_prod: {get_df_info(self.df_prod)} ',
            f'        departments: {self.n_departments}',
            f'        aisles: {self.n_aisles}          ',
            f'        products: {self.n_prod_items}    ',
            f'    df_trns_target: {get_df_info(self.df_trns_target)} ',
            f'        users: {self.n_users_target}            ',
            f'        items: {self.n_items_target}            ',
            f'                                         ',
            f'Total memory:                            ',
            f'    {format_size(total_memory)}          ',
        ]
        print(*info_message, sep='\n')
        return self


    def __repr__(self):
        class_name = self.__class__.__name__
        return (f'<{class_name} '
                f'transactions={len(self.df_trns)}>')
