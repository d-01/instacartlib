import contextlib
import time

from pathlib import Path

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


def get_df_info(df):
    memory_usage = df.memory_usage(deep=True).sum()
    return (f'<{type(df).__name__} '
            f'shape={df.shape} '
            f'memory={format_size(memory_usage)}>')
