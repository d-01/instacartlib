from .utils import download_from_info
from .utils import dummy_contextmanager
from .utils import timer_contextmanager

from pathlib import Path

import numpy as np
import pandas as pd


PRODUCTS_FILENAME = 'products.csv'
PRODUCTS_ZIP_FILENAME = 'products.csv.zip'


class PRODUCTS_DOWNLOAD_INFO:
    NAME = "products.csv.zip"      # 896KB
    MD5 = "35b4246bfeaec5fb6217ffc85975a6db"
    GDRIVE_ID = "130QJGLbv265WYgRtjXHm4FFY7QCMRXtv"


def get_products_csv_path(path_dir):
        for filename in [PRODUCTS_FILENAME, PRODUCTS_ZIP_FILENAME]:
            path = Path(path_dir) / filename
            if path.exists():
                return path

        abs_path = Path(path_dir).absolute()
        raise FileNotFoundError(
            f'Neither "{PRODUCTS_FILENAME}", '
            f'nor "{PRODUCTS_ZIP_FILENAME}.zip" '
            f'was not found at "{abs_path}".')


def read_products_csv(csv_path):
    """
    Read `products.csv` file into DataFrame.

    csv_path: str or pathlib.Path
        Path to csv file (`*.csv`) or zipped csv file (`*.zip`).
    """
    dtype = {
        'product_id': np.uint32,
        'aisle_id': np.uint32,
        'department_id': np.uint32,
    }
    df_raw = pd.read_csv(csv_path, dtype=dtype)
    return df_raw


def preprocess_raw_columns(df_raw):
    """
    df_raw: DataFrame
        Columns (6): product_id, product_name, aisle_id, department_id, aisle, 
        department

    Return:
    -------
    df: DataFrame
        Columns (6): iid, dept_id, aisle_id, dept, aisle, product
    """
    df = pd.DataFrame({
        'iid': df_raw.product_id,
        'dept_id': df_raw.department_id,
        'aisle_id': df_raw.aisle_id,
        'dept': df_raw.department,
        'aisle': df_raw.aisle,
        'product': df_raw.product_name,
    })
    return df


class Products:
    """
    Products data manipulator.

    Products dataframe:
    - iid - product id (raw)
    - dept_id - department id (raw)
    - aisle_id - aisle id (raw)
    - dept - department name
    - aisle - aisle name
    - product - product name
    
    show_progress: {False, True}
        No op. Left for consistency with Transactions API.
    """
    def __init__(self, show_progress=False):
        self.show_progress = show_progress

        self.df = None


    def _timer(self, message):
        if self.show_progress:
            return timer_contextmanager(message)
        else:
            return dummy_contextmanager()


    def load_from_gdrive(self, path_dir='.'):
        """
        Download files `transactions.csv` and `products.csv` from gdrive
        using url or id to current path or given directory.
        """
        download_from_info(PRODUCTS_DOWNLOAD_INFO, path_dir,
            self.show_progress)


    def from_dir(self, path_dir='.', reduced=False):
        """
        Read `products.csv` with raw data from current path or given local 
        directory into DataFrame.

        path_dir: str or pathlib.Path
            Path to directory with `transactions.csv` or `transactions.csv.zip` files.
        show_progress: {False, True}
            No op. Left for consistency with Transactions API.
        """
        products_csv_path = get_products_csv_path(path_dir)
        with self._timer(f'Reading "{products_csv_path.name}" ...'):
            df_raw = read_products_csv(products_csv_path)
        with self._timer('  Preprocessing columns ...'):
            self.df = preprocess_raw_columns(df_raw)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        df_shape = None if self.df is None else self.df.shape
        return f'<{class_name} df={df_shape}>'