"""
API for managing dataset generation process.

Generation process: extract features from transactions and products dataframes,
and cache them to files.

Capabilities:
* Register feature extractors.
* Generate (and cache) features for classification task.
* Combine dataset from extracted features.

Types of features:
1. UI - user-item features (interactions). Indexed by (uid, iid) pairs.
   * Example: `User A` has purchased `Item B` N times.
2. U - user features, not related to particular item. Indexed by (uid).
   * Example: `User A` has N orders total.
3. I - item features, not related to particular user. Indexed by (iid).
   * Example: `Item B` purchased by % users at least once.

```python
    fsds = FeaturesDataset(verbose=1)
    df_features = fsds.extract_features(instacart_dataset_train)
```
"""

from .feature_extractors import exports as feature_extractors
from .DataFrameFileCache import DataFrameFileCache
from .utils import get_df_info, increment_counter_suffix

from pathlib import Path

import numpy as np
import pandas as pd


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


def _make_unique_suffix(name: str, existing_names: set) -> str:
    while name in existing_names:
        name = increment_counter_suffix(name)
    return name


def _process_extractor_output(output, _feature_registry):
    existing_names = set(_feature_registry)
    duplicated_names = existing_names & set(output.columns)
    if duplicated_names == set():
        return (output, {})

    old_new_names = {
        old_name: _make_unique_suffix(old_name, existing_names)
        for old_name
        in duplicated_names
    }
    output = output.rename(columns=old_new_names)
    return (output, old_new_names)


def _get_feature_cache_path(features_cache_dir, name):
    return Path(features_cache_dir) / f'{name}.zip'


class FeaturesDataset:
    def __init__(self, df_trns, df_prod, features_cache_dir=None, verbose=0):
        self.features_cache_dir = features_cache_dir
        self.verbose = verbose

        self.df_trns = df_trns
        self.df_prod = df_prod

        ### move to `extract` method
        ui_index = (self.df_trns
            .set_index(['uid', 'iid'])
            .index.drop_duplicates())
        self.df_ui = pd.DataFrame(index=ui_index)
        ###

        self._feature_extractors = {}
        self._feature_registry = {}

        self.cache_enabled = self.features_cache_dir is not None
        if self.features_cache_dir is not None:
            self.features_cache_dir = Path(self.features_cache_dir)

        self.register_feature_extractors(feature_extractors)


    def _print(self, message, indent=0):
        if self.verbose > 0:
            if indent > 0:
                pad = ' ' * indent
                message = pad + message.replace('\n', '\n' + pad)
            print(message)


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

            self._print(f'Extracted features: {list(output.columns)}',
                indent=2)

            output, old_new_names_dict = (
                _process_extractor_output(output, self._feature_registry))
            self._warn_renamed_features(old_new_names_dict)
            self._add_features_to_registry(extractor_name, output.columns)

            self.df_ui = self.df_ui.join(output)
        return self


    def _warn_renamed_features(self, old_new_names_dict):
        for old_name, new_name in old_new_names_dict.items():
            self._print(
                f'Feature has been renamed "{old_name}" -> "{new_name}", '
                f'because it has been already extracted by extractor '
                f'"{self._feature_registry[old_name]}".',
                indent=2)


    def _add_features_to_registry(self, extractor_name, feature_names):
        for feature_name in feature_names:
            self._feature_registry[feature_name] = extractor_name


    def info(self):
        info_message = [
            f'Features:                                  ',
            f'    df_ui: {get_df_info(self.df_ui)}       ',
            f'    columns: {self.df_ui.columns.to_list()}',
        ]
        print(*info_message, sep='\n')
        return self