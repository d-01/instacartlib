from instacartlib.Products import get_products_csv_path
from instacartlib.Products import read_products_csv
from instacartlib.Products import preprocess_raw_columns
from instacartlib.Products import InvalidProductsData

import io
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
        'prod': np.dtype('object'),
    }


PRODUCTS_CSV = """\
product_id,product_name,aisle_id,department_id,aisle,department
23,Organic Turkey Burgers,49,12,packaged poultry,meat seafood
79,Wild Albacore Tuna No Salt Added,95,15,canned meat seafood,canned goods
196,Soda,77,7,soft drinks,beverages
"""

PRODUCTS_CSV_MISSING_COLUMN = """\
product_id,aisle_id,department_id,aisle,department
23,49,12,packaged poultry,meat seafood
79,95,15,canned meat seafood,canned goods
196,77,7,soft drinks,beverages
"""


def test_read_products_csv_return_dataframe(products_csv_path,
        expected_raw_col_types):
    output_1 = read_products_csv(products_csv_path)
    assert type(output_1) == pd.DataFrame
    assert output_1.shape == (577, 6)
    assert output_1.dtypes.to_dict() == expected_raw_col_types

    output_2 = read_products_csv(io.StringIO(PRODUCTS_CSV))
    assert type(output_2) == pd.DataFrame
    assert output_2.shape == (3, 6)

    with pytest.raises(FileNotFoundError):
        read_products_csv('__NON_EXISTENT_FILE__')

    with pytest.raises(InvalidProductsData, match='product_name'):
        read_products_csv(io.StringIO(PRODUCTS_CSV_MISSING_COLUMN))


def test_df_prod_no_nans(df_prod):
    assert df_prod.notna().all(axis=None)


def test_df_prod_col_types(df_prod, expected_col_types):
    assert df_prod.dtypes.to_dict() == expected_col_types


def test_preprocess_raw_columns(df_prod_raw):
    output = preprocess_raw_columns(df_prod_raw)
    assert type(output) == pd.DataFrame
    assert output.shape == (577, 6)


def test_get_products_csv_path(test_data_dir):
    path = get_products_csv_path(test_data_dir)
    assert path == (test_data_dir / 'products.csv')

    with pytest.raises(FileNotFoundError):
        get_products_csv_path('__NON-EXISTENT_PATH__')
