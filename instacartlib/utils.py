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


