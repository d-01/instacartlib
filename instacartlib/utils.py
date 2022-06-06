import contextlib
import time

from pathlib import Path
import pandas as pd

import gdown

@contextlib.contextmanager
def timer_contextmanager(message):
    print(message, end='')
    t0 = time.time()
    yield
    elapsed_sec = time.time() - t0
    print(' ({:.3}s)'.format(elapsed_sec))


@contextlib.contextmanager
def dummy_contextmanager():
    yield


def download_from_info(download_info, path_dir='.', show_progress=False):
    path = gdown.cached_download(
        id=download_info.GDRIVE_ID,
        path=str(Path(path_dir) / download_info.NAME),
        md5=download_info.MD5,
        quiet=(not show_progress),
    )
    return path


def format_size(size_bytes):
    if size_bytes < 0:
        raise ValueError(f'size expected to be >= 0, got: {size_bytes}')

    size_kb = 1 if 0 < size_bytes < 1024 else round(size_bytes / 1024)
    if size_kb < 1000:
        return f'{size_kb} KB'

    size_mb = round(size_kb / 1024)
    if size_mb < 1000:
        return f'{size_mb} MB'

    size_gb = round(size_mb / 1024)
    return f'{size_gb} GB'


def get_df_size_bytes(df: pd.DataFrame, deep=True) -> int:
    return df.memory_usage(deep=deep).sum()


def get_df_info(df: pd.DataFrame) -> str:
    if type(df) != pd.DataFrame:
        raise TypeError(f'DataFrame expected, got: {type(df)}')
    memory_usage = get_df_size_bytes(df)
    return (f'<{type(df).__name__} '
            f'shape={df.shape} '
            f'memory=\'{format_size(memory_usage)}\'>')


def split_counter_suffix(string, sep='_'):
    parts = string.rsplit(sep, maxsplit=1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts
    return [string, None]


def increment_counter_suffix(string, sep='_'):
    (base, counter_str) = split_counter_suffix(string, sep=sep)
    if counter_str is None:
        counter_str = '0'
    counter_incremented_str = str(int(counter_str) + 1)
    return f'{base}{sep}{counter_incremented_str.zfill(len(counter_str))}'