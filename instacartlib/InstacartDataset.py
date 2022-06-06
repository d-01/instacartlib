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
from .utils import get_df_info, format_size

import numpy as np
import pandas as pd


class InstacartDataset:
    def __init__(self, verbose=0):
        self.verbose = verbose

        self._transactions = Transactions(show_progress=self.verbose > 0)
        self._products = Products(show_progress=self.verbose > 0)

        self.df_trns = pd.DataFrame()
        self.df_prod = pd.DataFrame()

        self.n_users = 0
        self.n_items = 0
        self.n_prod_items = 0
        self.n_aisles = 0
        self.n_departments = 0


    def get_dataframes(self):
        dataframes = dict(df_trns=self.df_trns, df_prod=self.df_prod)
        if hasattr(self, 'df_trns_target'):
            dataframes['df_trns_target'] = self.df_trns_target
        return dataframes


    @property
    def dataframes(self):
        return self.get_dataframes()


    def download(self, to_dir='instacart_temp/raw_data'):
        self._transactions.load_from_gdrive(to_dir)
        self._products.load_from_gdrive(to_dir)
        return self


    def read_dir(self, path_dir='instacart_temp/raw_data', reduced=False):
        self._transactions.from_dir(path_dir=path_dir, reduced=reduced)
        self.df_trns = self._transactions.df
        self.n_users = self.df_trns.uid.nunique()
        self.n_items = self.df_trns.iid.nunique()

        self._products.from_dir(path_dir=path_dir, reduced=reduced)
        self.df_prod = self._products.df
        self.n_aisles = self.df_prod.aisle_id.nunique()
        self.n_departments = self.df_prod.dept_id.nunique()
        self.n_prod_items = self.df_prod.iid.nunique()

        return self


    def info(self):
        total_memory = (self.df_trns.memory_usage().sum()
                        + self.df_prod.memory_usage().sum())
        info_message = [
            f'Raw data:                                  ',
            f'    df_trns: {get_df_info(self.df_trns)}   ',
            f'        users: {self.n_users}',
            f'        items: {self.n_items}',
            f'    df_prod: {get_df_info(self.df_prod)}   ',
            f'        departments: {self.n_departments}',
            f'        aisles: {self.n_aisles}',
            f'        products: {self.n_prod_items}',
            f'                                           ',
            f'Total memory:                              ',
            f'    {format_size(total_memory)}            ',
        ]
        print(*info_message, sep='\n')
        return self


    def __repr__(self):
        class_name = self.__class__.__name__
        return (f'<{class_name} '
                f'transactions={len(self.df_trns)}>')
