
from instacartlib.feature_extractors import exports as feature_extractors

import io
import pandas as pd

import pytest


TRANSACTIONS_CSV = '''\
  oid     uid  iord     iid  reord  dow  hour  days_prev  cart_pos
ord_A  user_A     1  item_B      0    1    16          0         1
ord_A  user_A     1  item_A      0    1    16          0         2
ord_E  user_A     0  item_A      1    4     8         30         1
ord_E  user_A     0  item_C      0    4     8         30         2
ord_C  user_B     1  item_C      0    4    11         30         1
ord_C  user_B     1  item_D      0    4    11         30         2
ord_D  user_B     0  item_C      0    3    10         13         1
ord_D  user_B     0  item_A      1    3    10         13         2
'''

PRODUCTS_CSV = '''\
   iid  dept_id  aisle_id    dept    aisle       product
item_A        1         1  dept A  aisle A  product name
item_B        1         1  dept A  aisle A  product name
item_C        1         2  dept A  aisle B  product name
item_D        2         3  dept B  aisle C  product name
item_E        2         4  dept B  aisle D  product name
'''


@pytest.fixture
def fxtr_freq():
    return feature_extractors['000_ui_freq.freq']


@pytest.fixture
def fxtr_avg_cart_pos():
    return feature_extractors['001_ui_avg_cart_pos.avg_cart_pos']


@pytest.fixture
def df_trns_1():
    return pd.read_fwf(io.StringIO(TRANSACTIONS_CSV))


@pytest.fixture
def df_prod_1():
    return pd.read_fwf(io.StringIO(PRODUCTS_CSV))


@pytest.fixture
def ui_index_1():
    return pd.read_fwf(io.StringIO('''
           uid     iid
        user_A  item_A
        user_A  item_B
        user_A  item_C
        user_A  item_D
        user_A  item_E
        user_B  item_A
        user_B  item_B
        user_B  item_C
        user_B  item_D
        user_B  item_E
    ''')).set_index(['uid', 'iid']).index

@pytest.mark.nondestructive
@pytest.mark.parametrize("extractor_name", feature_extractors)
def test_feature_extractors_output_valid(extractor_name,
        ui_index_1, df_trns_1, df_prod_1):
    function = feature_extractors[extractor_name]
    test_output = function(ui_index_1, df_trns_1, df_prod_1)
    pd.testing.assert_index_equal(test_output.index, ui_index_1)
    assert test_output.isna().values.sum() == 0


def test_ui_freq(fxtr_freq, ui_index, df_trns, uids, iids):
    test_output = fxtr_freq(ui_index, df_trns, None)
    assert type(test_output) == pd.DataFrame
    assert test_output.columns == ['freq']
    assert (test_output.freq.loc[uids[0], iids[[0, 1, 2]]].to_list()
        == [10, 1, 10])


def test_ui_freq__1(fxtr_freq, ui_index_1, df_trns_1):
    test_output = fxtr_freq(ui_index_1, df_trns_1, None)
    expected = pd.read_fwf(io.StringIO('''
           uid     iid  freq
        user_A  item_A     2
        user_A  item_B     1
        user_A  item_C     1
        user_A  item_D     0
        user_A  item_E     0
        user_B  item_A     1
        user_B  item_B     0
        user_B  item_C     2
        user_B  item_D     1
        user_B  item_E     0
    ''')).set_index(['uid', 'iid'])
    pd.testing.assert_frame_equal(test_output, expected)


def test_avg_cart_pos(fxtr_avg_cart_pos, ui_index, df_trns, uids, iids):
    out = fxtr_avg_cart_pos(ui_index, df_trns, None)
    assert type(out) == pd.DataFrame
    assert 'avg_cart_pos' in out.columns
    assert (out['avg_cart_pos'].loc[uids[0], iids[[0, 1, 2]]].to_list() 
        == [1.4, 2.0, 3.3])


def test_avg_cart_pos__1(fxtr_avg_cart_pos, ui_index_1, df_trns_1, df_prod_1):
    test_output = fxtr_avg_cart_pos(ui_index_1, df_trns_1, df_prod_1)
    expected = pd.read_fwf(io.StringIO('''
           uid     iid  avg_cart_pos
        user_A  item_A           1.5
        user_A  item_B           1.0
        user_A  item_C           2.0
        user_A  item_D         999.0
        user_A  item_E         999.0
        user_B  item_A           2.0
        user_B  item_B         999.0
        user_B  item_C           1.0
        user_B  item_D           2.0
        user_B  item_E         999.0
    ''')).set_index(['uid', 'iid'])
    pd.testing.assert_frame_equal(test_output, expected)
