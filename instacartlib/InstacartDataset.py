"""
Prepares data for feature extraction.


Capabilities:
* Download csv files with raw data.
* Load raw data and preprocess.
* Make train dataset.
* Make test dataset.
"""

from .Transactions import Transactions
from .Products import Products
from .utils import get_df_info, format_size, get_df_size_bytes

import numpy as np
import pandas as pd


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

        self.df_trns = pd.DataFrame()
        self.df_prod = pd.DataFrame()
        self.df_trns_target = pd.DataFrame()

        self.n_users = 0
        self.n_items = 0
        self.n_prod_items = 0
        self.n_aisles = 0
        self.n_departments = 0
        self.n_users_target = 0
        self.n_items_target = 0


    def get_dataframes(self):
        if self.train:
            return {
                'df_trns': self.df_trns,
                'df_prod': self.df_prod,
                'df_trns_target': self.df_trns_target,
            }
        else:
            return {
                'df_trns': self.df_trns,
                'df_prod': self.df_prod,
            }


    @property
    def dataframes(self):
        return self.get_dataframes()


    def download(self, to_dir='instacart_temp/raw_data'):
        self._transactions.load_from_gdrive(to_dir)
        self._products.load_from_gdrive(to_dir)
        return self


    def _update_stats(self):
        self.n_users = self.df_trns.uid.nunique()
        self.n_items = self.df_trns.iid.nunique()
        self.n_aisles = self.df_prod.aisle_id.nunique()
        self.n_departments = self.df_prod.dept_id.nunique()
        self.n_prod_items = self.df_prod.iid.nunique()
        if self.train:
            self.n_users_target = self.df_trns_target.uid.nunique()
            self.n_items_target = self.df_trns_target.iid.nunique()


    def read_dir(self, path_dir='instacart_temp/raw_data', reduced=False):
        self._transactions.read_dir(path_dir=path_dir, reduced=reduced)
        if self.train:
            self._create_target()
        self.df_trns = self._transactions.df

        self._products.read_dir(path_dir=path_dir, reduced=reduced)
        self.df_prod = self._products.df
        self._update_stats()
        return self


    def _create_target(self):
        self.df_trns_target = self._transactions.get_last_orders(1)
        self._transactions.drop_last_orders(1)


    def info(self):
        total_memory = (get_df_size_bytes(self.df_trns)
                        + get_df_size_bytes(self.df_prod))
        df_trns_target_message = []
        if self.train:
            total_memory += get_df_size_bytes(self.df_trns_target)
            df_trns_target_message = [
                f'    df_trns_target: {get_df_info(self.df_trns_target)} ',
                f'        users: {self.n_users_target}            ',
                f'        items: {self.n_items_target}            ',
            ]

        info_message = [
            f'Raw data:                                ',
            f'    df_trns: {get_df_info(self.df_trns)} ',
            f'        users: {self.n_users}            ',
            f'        items: {self.n_items}            ',
            f'    df_prod: {get_df_info(self.df_prod)} ',
            f'        departments: {self.n_departments}',
            f'        aisles: {self.n_aisles}          ',
            f'        products: {self.n_prod_items}    ',
            *df_trns_target_message,
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
