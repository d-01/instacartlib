from instacartlib.Products import get_products_csv_path
from instacartlib.Products import preprocess_raw_columns

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def expected_raw_col_types():
    return {
        'product_id': np.dtype('uint32'),
        'product_name': np.dtype('object'),
        'aisle_id': np.dtype('uint32'),
        'department_id': np.dtype('uint32'),
        'aisle': np.dtype('object'),
        'department': np.dtype('object'),
    }


@pytest.fixture
def expected_col_types():
    return {
        'iid': np.dtype('uint32'),
        'dept_id': np.dtype('uint32'),
        'aisle_id': np.dtype('uint32'),
        'dept': np.dtype('object'),
        'aisle': np.dtype('object'),
        'product': np.dtype('object'),
    }


def test_read_products_csv_return_dataframe(df_prod_raw):
    assert type(df_prod_raw) == pd.DataFrame


def test_df_prod_raw_col_types(df_prod_raw, expected_raw_col_types):
    assert df_prod_raw.dtypes.to_dict() == expected_raw_col_types


def test_df_prod_no_nans(df_prod):
    assert df_prod.notna().all(axis=None)


def test_df_prod_col_types(df_prod, expected_col_types):
    assert df_prod.dtypes.to_dict() == expected_col_types


def test_preprocess_raw_columns(df_prod_raw):
    output = preprocess_raw_columns(df_prod_raw)
    assert type(output) == pd.DataFrame
    assert output.shape == (49688, 6)


def test_get_products_csv_path(test_data_dir):
    path = get_products_csv_path(test_data_dir)
    assert path == (test_data_dir / 'products.csv.zip')

    with pytest.raises(FileNotFoundError):
        get_products_csv_path('__NON-EXISTENT_PATH__')
