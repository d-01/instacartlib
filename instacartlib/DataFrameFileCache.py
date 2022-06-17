"""
Capabilities:
1. If path doesn't exist, call function and save result to path.
2. Read DataFrame from path and return.
"""

from .utils import get_df_info

from pathlib import Path

import pandas as pd


class DataFrameFileCache:
    """
    path: str or pathlib.Path
        If the file exists read DataFrame from this file instead of
        calling the wrapped function.
        If the file doesn't exist call wrapped function and write the output
        DataFrame to this file.
    disable: {False, True}
        If False wrapper has no effect (pass-through).
    verbose: int
        If verbose > 0 print additional information.
    """
    def __init__(self, path, disable=False, verbose=0):
        self.path = Path(path).resolve()
        self.disable = disable
        self.verbose = verbose


    def __call__(self, __wrapped__):
        if self.disable:
            return __wrapped__
        else:
            self.__wrapped__ = __wrapped__
            return self.wrapper


    def _print(self, *args, **kwargs):
        if self.verbose > 0:
            print(*args, **kwargs)


    def wrapper(self, *args, **kwargs):
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            self._print(f'Waiting result from {self.__wrapped__} ...')
            result = self.__wrapped__(*args, **kwargs)
            if type(result) != pd.DataFrame:
                raise TypeError('Wrapped function expected to return '
                    f'pandas.DataFrame, got: {type(result)}')
            df_info = get_df_info(result)
            self._print(f'  ... writing {df_info} to "{self.path}".')
            result.to_pickle(self.path)
        else:
            self._print(f'Reading from "{self.path}" ...')
            result = pd.read_pickle(self.path)
            if type(result) != pd.DataFrame:
                raise TypeError(f'File "{self.path}" expected to contain '
                    f'pandas.DataFrame, got: {type(result)}')
            df_info = get_df_info(result)
            self._print(f'  ... {df_info} has been read.')
        return result



