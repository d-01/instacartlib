
from instacartlib.plugins import plugins

import io
import pandas as pd

import pytest


@pytest.fixture
def freq():
    return plugins['000_ui_freq.freq']


@pytest.fixture
def avg_cart_pos():
    return plugins['001_ui_avg_cart_pos.avg_cart_pos']


@pytest.fixture
def df_trns_1():
    return pd.read_fwf(io.StringIO('''
          oid     uid  iord     iid  reord  dow  hour  days_prev  in_cart_ord
        ord_A  user_A     1  item_B      0    1    16          0            1
        ord_A  user_A     1  item_A      0    1    16          0            2
        ord_E  user_A     0  item_A      1    4     8         30            1
        ord_E  user_A     0  item_C      0    4     8         30            2
        ord_C  user_B     1  item_C      0    4    11         30            1
        ord_C  user_B     1  item_D      0    4    11         30            2
        ord_D  user_B     0  item_C      0    3    10         13            1
        ord_D  user_B     0  item_A      1    3    10         13            2
    '''))


def test_ui_freq(freq, df_trns, uids, iids):
    out = freq(df_trns)
    assert type(out) == pd.DataFrame
    assert out.columns == ['freq']
    assert out.freq.loc[uids[0], iids[[0, 1, 2]]].to_list() == [10, 1, 10]


def test_ui_freq__1(freq, df_trns_1):
    test_output = freq(df_trns_1)

    expected = pd.read_fwf(io.StringIO('''
           uid     iid  freq
        user_A  item_B     1
        user_A  item_A     2
        user_A  item_C     1
        user_B  item_C     2
        user_B  item_D     1
        user_B  item_A     1
    ''')).set_index(['uid', 'iid'])
    pd.testing.assert_frame_equal(test_output, expected)


def test_avg_cart_pos(avg_cart_pos, df_trns, uids, iids):
    out = avg_cart_pos(df_trns)
    assert type(out) == pd.DataFrame
    assert 'avg_cart_pos' in out.columns
    assert (out['avg_cart_pos'].loc[uids[0], iids[[0, 1, 2]]].to_list() 
        == [1.4, 2.0, 3.3])


def test_avg_cart_pos__1(avg_cart_pos, df_trns_1):
    test_output = avg_cart_pos(df_trns_1)

    expected = pd.read_fwf(io.StringIO('''
           uid     iid  avg_cart_pos
        user_A  item_B           1.0
        user_A  item_A           1.5
        user_A  item_C           2.0
        user_B  item_C           1.0
        user_B  item_D           2.0
        user_B  item_A           2.0
    ''')).set_index(['uid', 'iid'])
    pd.testing.assert_frame_equal(test_output, expected)
