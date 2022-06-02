"""
Main API to manage dataset generation process.

Generation process: extract features from transactions and products dataframes, 
and cache them to files.


Types of features:
1. UI - user-item features (interactions). Indexed by (uid, iid) pairs.
   * Example: `User A` has purchased `Item B` N times.
2. U - user features, not related to particular item. Indexed by (uid).
   * Example: `User A` has N orders total.
3. I - item features, not related to particular user. Indexed by (iid).
   * Example: `Item B` purchased by % users at least once.


API draft:
```
import instacartlib.UserItemDataset as UserItemDataset
import extractors

ds = UserItemDataset(cache_dir='./cached_features', verbose=1, **config)

ds.download(to_dir='abc')
ds.read_dir('abc')
ds.info()

ds.register_feature_extractors({
    **extractors,
    'custom_feature': extractor_function,
})

ds.extract([...])  # long
ds.add_negative_samples()  # how many

(train, val) = ds.get_dataset(val_split=0.2)
x_train, y_train = train[:, -1:], train[:, -1]
x_val, y_val = val[:, -1:], val[:, -1]
```

Capabilities:
* Download csv files with raw data.
* Load raw data and preprocess.
* Register feature extractors.

Capabilities (planned):
* Generate (and cache) features for classification task.
* Combine dataset from extracted features.
* Make train dataset.
* Make test dataset.
* (maybe) Manage common and specific parameters (settings) for feature extractors.
"""

from .Transactions import Transactions
from .Products import Products
from .feature_extractors import exports as feature_extractors
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
        self.df_ui = None
        self.ui_index = None

        self._feature_extractors = {}


    def download(self, to_dir='instacart_temp/raw_data'):
        self._transactions.load_from_gdrive(to_dir)
        self._products.load_from_gdrive(to_dir)
        return self


    def read_dir(self, path_dir='instacart_temp/raw_data', reduced=False):
        self._transactions.from_dir(path_dir=path_dir, reduced=reduced)
        self.df_trns = self._transactions.df

        self._products.from_dir(path_dir=path_dir, reduced=reduced)
        self.df_prod = self._products.df

        self._assert_df_ui_initialized()
        return self


    def register_feature_extractors(self, feature_extractors: dict):
        already_exist = (
            set(self._feature_extractors) & set(feature_extractors))
        if len(already_exist) > 0:
            raise ValueError(
                f"Feature extractors already registered: {already_exist}.")
        self._feature_extractors.update(feature_extractors)
        return self


    def _assert_dataframes_initialized(self):
        if self.df_trns is None:
            raise DataNotLoaded('Transactions data not loaded (call '
                '`read_dir` to load).')
        if self.df_prod is None:
            raise DataNotLoaded('Products data not loaded (call '
                '`read_dir` to load).')


    def _assert_df_ui_initialized(self):
        self._assert_dataframes_initialized()
        if self.df_ui is None:
            self.ui_index = (self.df_trns
                .set_index(['uid', 'iid'])
                .index.drop_duplicates())
            self.df_ui = pd.DataFrame(index=self.ui_index)


    def info(self):
        self._assert_dataframes_initialized()
        
        print(INFO_TEMPLATE.format(
            df_trns_info=get_df_info(self.df_trns),
            df_prod_info=get_df_info(self.df_prod),
        ))
        return self


