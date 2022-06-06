
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
def df_trns_1():
    return pd.read_fwf(io.StringIO(TRANSACTIONS_CSV))


@pytest.fixture
def df_trns_target_1(df_trns_1):
    return df_trns_1[df_trns_1.iord == 0]


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


@pytest.fixture
def data_frames_1(df_trns_1, df_prod_1):
    return dict(
        df_trns=df_trns_1,
        df_prod=df_prod_1,
    )


@pytest.fixture
def data_frames_target_1(df_trns_1, df_prod_1, df_trns_target_1):
    return dict(
        df_trns=df_trns_1,
        df_trns_target=df_trns_target_1,
        df_prod=df_prod_1,
    )



@pytest.mark.parametrize("extractor_name", feature_extractors.keys())
def test_feature_extractors_output_valid(extractor_name, ui_index_1,
        data_frames_1):
    function = feature_extractors[extractor_name]
    test_output = function(ui_index_1, **data_frames_1)
    pd.testing.assert_index_equal(test_output.index, ui_index_1)
    assert test_output.isna().values.sum() == 0

    with pytest.raises(TypeError,
            match=r'missing \d+ required positional argument'):
        test_output = function()

    extra_data_frames = {'unused_1': 1, 'unused_2': 2, **data_frames_1}
    test_output = function(ui_index_1, **extra_data_frames)



@pytest.mark.skipif(
    '000_ui_freq.freq' not in feature_extractors,
    reason="feature extractor was not registered",
)
def test_ui_freq(ui_index, data_frames, uids, iids):
    freq = feature_extractors['000_ui_freq.freq']

    test_output = freq(ui_index, **data_frames)
    assert type(test_output) == pd.DataFrame
    assert test_output.columns == ['freq']
    assert (test_output.freq.loc[uids[0], iids[[0, 1, 2]]].to_list()
        == [10, 1, 10])
    pd.testing.assert_index_equal(test_output.index, ui_index)


@pytest.mark.skipif(
    '000_ui_freq.freq' not in feature_extractors,
    reason="feature extractor was not registered",
)
def test_ui_freq__1(ui_index_1, data_frames_1):
    freq = feature_extractors['000_ui_freq.freq']

    test_output = freq(ui_index_1, **data_frames_1)
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


@pytest.mark.skipif(
    '001_ui_avg_cart_pos.avg_cart_pos' not in feature_extractors,
    reason="feature extractor was not registered",
)
def test_avg_cart_pos(ui_index, data_frames, uids, iids):
    avg_cart_pos = feature_extractors['001_ui_avg_cart_pos.avg_cart_pos']

    out = avg_cart_pos(ui_index, **data_frames)
    assert type(out) == pd.DataFrame
    assert 'avg_cart_pos' in out.columns
    assert (out['avg_cart_pos'].loc[uids[0], iids[[0, 1, 2]]].to_list()
        == [1.4, 2.0, 3.3])
    pd.testing.assert_index_equal(out.index, ui_index)


@pytest.mark.skipif(
    '001_ui_avg_cart_pos.avg_cart_pos' not in feature_extractors,
    reason="feature extractor was not registered",
)
def test_avg_cart_pos__1(ui_index_1, data_frames_1):
    avg_cart_pos = feature_extractors['001_ui_avg_cart_pos.avg_cart_pos']

    test_output = avg_cart_pos(ui_index_1, **data_frames_1)
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
