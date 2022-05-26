"""
Main API to manage dataset generation process.

API draft:
```
from instacartlib import InstacartDataset

inst = InstacartDataset(**config)

inst.download(to_dir='abc')
inst.read_dir('abc')
inst.info()

inst.make_features([...], cache='path to dir or None')
inst.add_negative_samples()  # how many

(train, val) = inst.get_dataset(val_split=0.2)
x_train, y_train = train[:, -1:], train[:, -1]
x_val, y_val = val[:, -1:], val[:, -1]

inst.make_features([...], cache='path to dir or None')
x = inst.get_dataset()
```

Capabilities:
1. Download csv files with raw data.
1. Load raw data and preprocess.

Capabilities (planned):
1. Generate (and cache) features for classification task.
1. Make train dataset.
1. Make test dataset.
"""

from .Transactions import Transactions
from .Products import Products
from .utils import get_df_info

import numpy as np
import pandas as pd


INFO_TEMPLATE = """\
Raw data:
  df_trns: {df_trns_info}
  df_prod: {df_prod_info}

Features:

"""


class DataNotLoaded(Exception):
    pass


class InstacartDataset:
    def __init__(self, features_cache_dir=None, show_progress=False):
        self.features_cache_dir = features_cache_dir
        self.show_progress = show_progress

        self._transactions = Transactions(show_progress=self.show_progress)
        self._products = Products(show_progress=self.show_progress)

        self.df_trns = None
        self.df_prod = None
        self.dataset = None

        self._feature_manager = None


    def download(self, to_dir='instacart_temp/raw_data'):
        self._transactions.load_from_gdrive(to_dir)
        self._products.load_from_gdrive(to_dir)
        return self


    def read_dir(self, path_dir='instacart_temp/raw_data', reduced=False):
        self._transactions.from_dir(path_dir=path_dir, reduced=reduced)
        self.df_trns = self._transactions.df

        self._products.from_dir(path_dir=path_dir, reduced=reduced)
        self.df_prod = self._products.df

        return self


    def _assert_dataframes_initialized(self):
        if self.df_trns is None:
            raise DataNotLoaded('Transactions data not loaded (call '
                '`read_dir` to load).')
        if self.df_prod is None:
            raise DataNotLoaded('Products data not loaded (call '
                '`read_dir` to load).')


    def _assert_dataset_initialized(self):
        self._assert_dataframes_initialized()
        if self.dataset is None:
            ui_index = (self.df_trns
                .set_index(['uid', 'iid']).index
                .drop_duplicates())
            self.dataset = pd.DataFrame(index=ui_index)


    def info(self):
        self._assert_dataframes_initialized()
        
        print(INFO_TEMPLATE.format(
            df_trns_info=get_df_info(self.df_trns),
            df_prod_info=get_df_info(self.df_prod),
        ))
        return self




