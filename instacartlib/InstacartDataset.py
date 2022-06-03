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

# ds.add_negative_samples()  # ?

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

from .DataFrameFileCache import DataFrameFileCache
from .Transactions import Transactions
from .Products import Products
from .feature_extractors import exports as feature_extractors
from .utils import get_df_info, format_size

from pathlib import Path

import numpy as np
import pandas as pd


class DataNotLoaded(Exception):
    """ Trying to use method that requires data before the data has been read.
    """

class ExtractorCallError(Exception):
    """ Calling an extractor function raises an exception. """

class ExtractorInvalidOutputError(Exception):
    """ The output from an extractor function is not valid. """

class ExtractorExistsError(Exception):
    """ Extractor with the same name has been already registered. """


def _use_extractor(function, ui_index, df_trns, df_prod):
    try:
        return function(ui_index, df_trns, df_prod)
    except Exception as e:
        info = '  ' + str(e).replace('\n', '\n  ')
        raise ExtractorCallError(f'Extractor failed with error:\n{info}')


def _assert_extractor_output(extractor_output, expected_index):
    if type(extractor_output) != pd.DataFrame:
        raise ExtractorInvalidOutputError(
            f'Extractor\'s output expected to be DataFrame, '
            f'got: {type(extractor_output)}')
    try:
        pd.testing.assert_index_equal(extractor_output.index, expected_index)
    except AssertionError as e:
        raise ExtractorInvalidOutputError(e)


def _process_extractor_output(output, _feature_registry):
    duplicated_features = set(_feature_registry) & set(output.columns)
    if duplicated_features:
        output = output.drop(columns=duplicated_features)
    return (output, duplicated_features)


def _get_feature_cache_path(features_cache_dir, name):
    return Path(features_cache_dir) / f'{name}.zip'


class InstacartDataset:
    def __init__(self, features_cache_dir=None, verbose=0):
        self.features_cache_dir = features_cache_dir
        self.verbose = verbose

        self._transactions = Transactions(show_progress=self.verbose > 0)
        self._products = Products(show_progress=self.verbose > 0)

        self.df_trns = pd.DataFrame()
        self.df_prod = pd.DataFrame()
        self.df_ui = pd.DataFrame()

        self.n_users = 0
        self.n_items = 0
        self.n_prod_items = 0
        self.n_aisles = 0
        self.n_departments = 0

        self._feature_extractors = {}
        self._feature_registry = {}

        self.cache_enabled = self.features_cache_dir is not None
        if self.features_cache_dir is not None:
            self.features_cache_dir = Path(self.features_cache_dir)

        self.register_feature_extractors(feature_extractors)


    def _print(self, *args, indent=0, **kwargs):
        if self.verbose > 0:
            if indent > 0:
                pad = ' ' * indent
                args = [str(arg).replace('\n', '\n' + pad) for arg in args]
            print(*args, **kwargs)


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

        self._init_df_ui()
        return self


    def _init_df_ui(self):
        self._assert_dataframes_initialized()
        ui_index = (self.df_trns
            .set_index(['uid', 'iid'])
            .index.drop_duplicates())
        self.df_ui = pd.DataFrame(index=ui_index)


    def _assert_dataframes_initialized(self):
        if len(self.df_trns.columns) == 0:
            raise DataNotLoaded('Transactions data not loaded (call '
                '`read_dir` to load).')
        if len(self.df_trns.columns) == 0:
            raise DataNotLoaded('Products data not loaded (call '
                '`read_dir` to load).')


    def register_feature_extractors(self, feature_extractors: dict):
        already_exist = (
            set(self._feature_extractors) & set(feature_extractors))
        if len(already_exist) > 0:
            raise ExtractorExistsError(
                f"Feature extractors already registered: {already_exist}.")

        if self.cache_enabled:
            for name, function in feature_extractors.items():
                path = _get_feature_cache_path(self.features_cache_dir, name)
                wrapper = DataFrameFileCache(path, verbose=self.verbose - 1)
                self._feature_extractors[name] = wrapper(function)
        else:
            self._feature_extractors.update(feature_extractors)
        return self


    def extract_features(self):
        if len(self._feature_extractors) == 0:
            self._print(
                'No feature extractors have been registered yet. '
                'Use `register_feature_extractors()` method first.')

        for extractor_name, function in self._feature_extractors.items():
            self._print(f'Using extractor: "{extractor_name}"')

            try:
                output = _use_extractor(function,
                                        self.df_ui.index,
                                        self.df_trns,
                                        self.df_prod)
                _assert_extractor_output(output, self.df_ui.index)
            except Exception as e:
                self._print(e, indent=2)
                continue

            self._print(f'Extracted features: {list(output.columns)}', indent=2)

            output, duplicated_features = (
                _process_extractor_output(output, self._feature_registry))
            self._warn_duplicated_features(duplicated_features)
            self._add_features_to_registry(extractor_name, output.columns)

            self.df_ui = self.df_ui.join(output)
        return self


    def _warn_duplicated_features(self, feature_names):
        for feature_name in feature_names:
            self._print(
                f'"{feature_name}" has been already extracted using '
                f'"{self._feature_registry[feature_name]}"',
                indent=2)


    def _add_features_to_registry(self, extractor_name, feature_names):
        for feature_name in feature_names:
            self._feature_registry[feature_name] = extractor_name


    def info(self):
        total_memory = (self.df_trns.memory_usage().sum()
                        + self.df_prod.memory_usage().sum()
                        + self.df_ui.memory_usage().sum())
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
            f'Features:                                  ',
            f'    df_ui: {get_df_info(self.df_ui)}       ',
            f'    columns: {self.df_ui.columns.to_list()}',
            f'                                           ',
            f'Total memory:                              ',
            f'    {format_size(total_memory)}            ',
        ]
        print(*info_message, sep='\n')
        return self


    def __repr__(self):
        class_name = self.__class__.__name__
        return (f'<{class_name} '
                f'transactions={len(self.df_trns)} '
                f'features={len(self.df_ui.columns)}>')
